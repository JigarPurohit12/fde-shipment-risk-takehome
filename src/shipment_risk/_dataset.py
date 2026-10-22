"""Offline point-in-time training rows (README "Required work" 1).

build_dataset() returns:
  rows   - one row per unique decision, sorted by (as_of, entity_id)
  report - counts of everything accepted, ignored or censored, so nothing
           disappears silently (printed by the demo, copied into RESULTS.md)

How a row is made (one decision = shipment + as_of):
  1. as_of after observation_end   -> censored "decision_after_cutoff",
                                      features and label unknown (None)
  2. features = compute_features(store.known_at(shipment, as_of))
                                      (same two calls as live scoring)
  3. label not yet mature           -> censored "label_window_immature",
                                      label None, EVEN IF an incident is visible
  4. otherwise                      -> eligible, label 1 iff an incident in
                                      (as_of, as_of + 6h] was available by the cutoff

Input handling:
  * Malformed record anywhere -> InvalidRecordError naming it, e.g. events[17].
    A broken export is a pipeline bug to fix, not something to train around.
  * Events received after observation_end and labels available after it are
    ignored and counted: they did not exist yet for this export.
  * Event conflicts keep the FIRST version and are counted, exactly like the
    live engine will, so offline rows equal what the engine would compute.
  * Label conflicts raise ConflictError: there is no live path to mirror, and
    two different incident times cannot be resolved automatically.
  * Duplicate decisions (same shipment and instant) become one row.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from ._features import FEATURE_SCHEMA_VERSION, compute_features
from ._labels import HORIZON, LABEL_REPORTING_LAG, incidents_in_window, is_mature, label_mature_at
from ._records import Label, validate_decision, validate_event, validate_label
from ._store import ACCEPTED, DUPLICATE, SUPERSEDED_ON_ARRIVAL, EventStore
from ._time import format_timestamp, require_aware_datetime
from .errors import ConflictError, InvalidArgumentError, InvalidRecordError

# Censor reasons (row.metadata.censor_reason)
DECISION_AFTER_CUTOFF = "decision_after_cutoff"
LABEL_WINDOW_IMMATURE = "label_window_immature"

_HOUR = timedelta(hours=1)


@dataclass(frozen=True)
class Dataset:
    rows: list
    report: dict


def build_dataset(events, labels, decisions, observation_end) -> Dataset:
    cutoff = require_aware_datetime(observation_end, "observation_end")
    store, event_counts = _load_events(events, cutoff)
    labels_by_entity, label_counts = _load_labels(labels, cutoff)
    decision_copies = _load_decisions(decisions)

    rows = [
        _make_row(entity_id, as_of, copies, store, labels_by_entity, cutoff)
        for (as_of, entity_id), copies in sorted(decision_copies.items())
    ]
    return Dataset(rows=rows, report=_report(cutoff, event_counts, label_counts, decision_copies, rows))


# --------------------------------------------------------------------------
# Loading inputs
# --------------------------------------------------------------------------

def _load_events(events, cutoff: datetime):
    store = EventStore()
    counts = Counter({key: 0 for key in
                      ("input", ACCEPTED, DUPLICATE, SUPERSEDED_ON_ARRIVAL, "conflict", "received_after_cutoff")})
    # Iteration order = delivery order, so "first version kept" means the same
    # thing here as in the live engine.
    for index, record in _numbered(events, "events"):
        event = _validated(validate_event, record, "events", index)
        counts["input"] += 1
        if event.received_at > cutoff:
            counts["received_after_cutoff"] += 1
            continue
        try:
            counts[store.add(event).status] += 1
        except ConflictError:
            counts["conflict"] += 1
    return store, counts


def _load_labels(labels, cutoff: datetime):
    counts = Counter({key: 0 for key in
                      ("input", "used", DUPLICATE, "available_after_cutoff", "beyond_reporting_lag")})
    seen: dict[tuple[str, str], Label] = {}
    for index, record in _numbered(labels, "labels"):
        label = _validated(validate_label, record, "labels", index)
        counts["input"] += 1
        if label.label_available_at > cutoff:
            counts["available_after_cutoff"] += 1
            continue
        key = (label.entity_id, label.incident_id)
        existing = seen.get(key)
        if existing is None:
            seen[key] = label
        elif existing == label:
            counts[DUPLICATE] += 1
        else:
            raise ConflictError(
                "label_conflict",
                f"labels[{index}]: incident {label.incident_id!r} of shipment "
                f"{label.entity_id!r} appears twice with different content",
            )

    labels_by_entity: dict[str, list[Label]] = defaultdict(list)
    for label in seen.values():
        labels_by_entity[label.entity_id].append(label)
        # A label that took longer than the promised lag means the maturity
        # rule may be hiding other late positives: a monitoring signal.
        if label.label_available_at - label.incident_at > LABEL_REPORTING_LAG:
            counts["beyond_reporting_lag"] += 1
    counts["used"] = len(seen)
    return labels_by_entity, counts


def _load_decisions(decisions) -> Counter:
    """(as_of, entity_id) -> number of copies. Normalized, so offsets do not matter."""
    copies: Counter = Counter()
    for index, record in _numbered(decisions, "decisions"):
        decision = _validated(validate_decision, record, "decisions", index)
        copies[(decision.as_of, decision.entity_id)] += 1
    return copies


def _numbered(value, name: str):
    # A str, bytes or single mapping is iterable but is not a list of records.
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise InvalidArgumentError(
            "not_iterable", f"{name} must be an iterable of records, got {type(value).__name__}"
        )
    return enumerate(value)


def _validated(validator, record, name: str, index: int):
    try:
        return validator(record)
    except InvalidRecordError as exc:
        raise InvalidRecordError(exc.code, f"{name}[{index}]: {exc.detail}") from None


# --------------------------------------------------------------------------
# One row
# --------------------------------------------------------------------------

def _make_row(entity_id, as_of, copies, store, labels_by_entity, cutoff) -> dict:
    metadata = {
        "censor_reason": None,
        "data_reasons": [],
        "duplicate_decisions": copies - 1,
        "eligible": False,
        "feature_digest": None,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "incidents": [],
        "label_mature_at": format_timestamp(label_mature_at(as_of)),
        "label_window_end": format_timestamp(as_of + HORIZON),
        "observation_end": format_timestamp(cutoff),
    }
    row = {
        "entity_id": entity_id,
        "as_of": format_timestamp(as_of),
        "features": None,
        "label": None,
        "metadata": metadata,
    }

    if as_of > cutoff:
        # Records after the cutoff were dropped, so features "as of" a later
        # moment would be incomplete. Report unknown instead of guessing.
        metadata["censor_reason"] = DECISION_AFTER_CUTOFF
        return row

    features = compute_features(store.known_at(entity_id, as_of))
    row["features"] = dict(features.values)
    metadata["feature_digest"] = features.digest
    metadata["data_reasons"] = list(features.reasons)

    if not is_mature(as_of, cutoff):
        metadata["censor_reason"] = LABEL_WINDOW_IMMATURE
        return row

    incidents = incidents_in_window(labels_by_entity.get(entity_id, ()), as_of, known_by=cutoff)
    metadata["eligible"] = True
    metadata["incidents"] = [label.to_record() for label in incidents]
    row["label"] = 1 if incidents else 0
    return row


def _report(cutoff, event_counts, label_counts, decision_copies, rows) -> dict:
    censored = Counter({DECISION_AFTER_CUTOFF: 0, LABEL_WINDOW_IMMATURE: 0})
    for row in rows:
        reason = row["metadata"]["censor_reason"]
        if reason is not None:
            censored[reason] += 1
    eligible = [row for row in rows if row["metadata"]["eligible"]]
    return {
        "observation_end": format_timestamp(cutoff),
        "horizon_hours": HORIZON / _HOUR,
        "label_reporting_lag_hours": LABEL_REPORTING_LAG / _HOUR,
        "events": dict(sorted(event_counts.items())),
        "labels": dict(sorted(label_counts.items())),
        "decisions": {
            "input": sum(decision_copies.values()),
            "rows": len(rows),
            "duplicates_merged": sum(decision_copies.values()) - len(rows),
            "eligible": len(eligible),
            "positive": sum(row["label"] for row in eligible),
            "censored": dict(sorted(censored.items())),
        },
    }
