from __future__ import annotations
import threading
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from ._canonical import canonical_json
from ._dataset import build_dataset
from ._features import FEATURE_SCHEMA_VERSION, compute_features
from ._io import parse_json_object, write_atomic
from ._model import load_artifact
from ._records import validate_event
from ._store import ACCEPTED, SUPERSEDED_ON_ARRIVAL, EventStore
from ._time import format_timestamp, require_aware_datetime
from ._training import train_model
from .contracts import Result
from .errors import (ArtifactError, ConflictError, InvalidArgumentError, RetentionError,
                     ShipmentRiskError, SnapshotError)

MAX_REVISIONS_PER_SHIPMENT = 1000     # per-shipment memory cap (customer note 4)
PROBABILITY_DECIMALS = 6              # wire output; hides last-bit float noise across machines
SNAPSHOT_FORMAT = "shipment-risk-snapshot-v1"
UNKNOWN_SHIPMENT = "unknown_shipment"  # score reason: nothing was ever ingested for this shipment

Record = Mapping[str, object]

def build_training_rows(events: Iterable[Record], labels: Iterable[Record], decisions: Iterable[Record], observation_end: datetime) -> list[dict]:
    """Return point-in-time rows, documenting excluded/censored rows.

    One row per unique decision, sorted by (as_of, entity_id). Censored rows
    are returned (not dropped) with label None and metadata.censor_reason.
    Rules and the counts report: see _dataset.build_dataset.
    """
    return build_dataset(events, labels, decisions, observation_end).rows

def train(rows: Sequence[Record], artifact_dir: Path) -> Mapping[str, object]:
    """Persist a fresh-process-loadable artifact and return actual evaluation metrics.

    Writes artifact_dir/model.json (the model) and returns the evaluation
    report. Censored rows are ignored and counted. Split rules,
    baselines and metrics: see _training.train_model.
    """
    return train_model(rows, artifact_dir)


class RiskEngine:
    """Working prototype of the live engine (see DECISIONS.md for what is not done yet).

    It reuses the exact offline code path, so online and offline agree:
        ingest: validate_event -> EventStore.add            (steps 1-2)
        score:  EventStore.known_at -> compute_features -> model.predict   (steps 2, 3, 5)

    Customer note 3: no valid model -> the constructor raises (never "zero risk");
                     a failed reload_model returns False and keeps the old model.
    Customer note 4: memory is bounded - at most max_entities shipments (least
                     recently ingested is evicted) and MAX_REVISIONS_PER_SHIPMENT
                     stored revisions each. Records or questions for an evicted
                     shipment are rejected, never answered from partial history.
    Concurrency: one lock around all state; a reload swaps (model, version) together.
    """

    def __init__(self, artifact_dir: Path, max_entities: int = 10000):
        if type(max_entities) is not int or max_entities < 1:          # rejects True, 0, 2.5
            raise InvalidArgumentError("invalid_max_entities", f"max_entities must be a positive int, got {max_entities!r}")
        self._model, self._model_version = load_artifact(_path(artifact_dir, "artifact_dir"))
        self._max_entities = max_entities
        self._store = EventStore()
        self._recency = OrderedDict()   # shipment -> stored revisions; first = least recently ingested
        self._evicted = OrderedDict()   # evicted shipments (bounded to max_entities), oldest first
        self._counters = Counter()
        self._last_reload_error = None
        self._lock = threading.Lock()

    def ingest(self, record: Record) -> bool:
        """True iff the record changes what some as-of question would see."""
        try:
            event = validate_event(record)                 # raises InvalidRecordError
        except ShipmentRiskError:
            self._count("rejected_invalid")
            raise
        entity_id = event.entity_id
        with self._lock:
            if entity_id in self._evicted:                 # e.g. a late correction after eviction
                self._counters["rejected_entity_evicted"] += 1
                raise RetentionError("entity_evicted", f"shipment {entity_id!r} was evicted; its history is gone")
            if self._recency.get(entity_id, 0) >= MAX_REVISIONS_PER_SHIPMENT:
                self._counters["rejected_history_full"] += 1
                raise RetentionError("history_full", f"shipment {entity_id!r} already stores {MAX_REVISIONS_PER_SHIPMENT} revisions")
            if entity_id not in self._recency and len(self._recency) >= self._max_entities:
                self._evict_least_recent()
            try:
                result = self._store.add(event)
            except ConflictError:                          # first version kept
                self._counters["conflict"] += 1
                raise
            stored = result.status in (ACCEPTED, SUPERSEDED_ON_ARRIVAL)
            self._recency[entity_id] = self._recency.get(entity_id, 0) + int(stored)
            self._recency.move_to_end(entity_id)
            self._counters[result.status] += 1
            return result.changed

    def score(self, entity_id: str, as_of: datetime) -> Result:
        """Probability of an excursion in (as_of, as_of + 6h], from data received by as_of."""
        if not isinstance(entity_id, str) or not entity_id or entity_id != entity_id.strip():
            raise InvalidArgumentError("invalid_entity_id", f"entity_id must be a non-empty string, got {entity_id!r}")
        as_of = require_aware_datetime(as_of, "as_of")
        with self._lock:                                   # consistent view, then release
            if entity_id in self._evicted:
                self._counters["rejected_entity_evicted"] += 1
                raise RetentionError("entity_evicted", f"shipment {entity_id!r} was evicted; its history is gone")
            known = self._store.known_at(entity_id, as_of)
            is_known = entity_id in self._recency
            model, model_version = self._model, self._model_version
        features = compute_features(known)
        reasons = sorted(set(features.reasons) | ({UNKNOWN_SHIPMENT} if not is_known else set()))
        self._count("scores")
        if reasons:
            self._count("degraded_scores")
        payload = {
            "probability": round(model.predict(features.values), PROBABILITY_DECIMALS) + 0.0,
            "model_version": model_version,
            "feature_digest": features.digest,
            "degraded": bool(reasons),
            "reasons": reasons,
            "feature_schema": FEATURE_SCHEMA_VERSION,   # extra fields, for tracing and audits
            "features": dict(features.values),
        }
        return Result(entity_id=entity_id, as_of=format_timestamp(as_of), payload=payload)

    def snapshot(self, destination: Path) -> None:
        """Write all state as canonical JSON, atomically. Same state -> same bytes."""
        path = _path(destination, "destination")
        with self._lock:
            by_entity = {}
            for event in self._store.events():             # sorted by identity
                by_entity.setdefault(event.entity_id, []).append(event.to_record())
            body = {
                "format": SNAPSHOT_FORMAT,
                "max_entities": self._max_entities,
                "entities": [{"entity_id": e, "events": by_entity[e]} for e in self._recency],
                "evicted": list(self._evicted),
                "counters": dict(sorted(self._counters.items())),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, canonical_json(body) + b"\n")

    @classmethod
    def restore(cls, artifact_dir: Path, snapshot: Path):
        """New engine: model from artifact_dir, state rebuilt by re-ingesting the snapshot."""
        path = _path(snapshot, "snapshot")
        try:
            body = parse_json_object(path.read_text(encoding="utf-8"), path.name)
            if body.get("format") != SNAPSHOT_FORMAT or len(body["entities"]) > body["max_entities"]:
                raise ValueError("not a valid engine snapshot")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise SnapshotError("invalid_snapshot", f"{path}: {exc}") from None
        engine = cls(artifact_dir, body["max_entities"])    # ArtifactError if the model is bad
        try:
            for entry in body["entities"]:                 # least recently ingested first
                for record in entry["events"]:
                    engine.ingest(record)                  # same validation as live data
            engine._evicted = OrderedDict.fromkeys(body["evicted"])
            engine._counters = Counter(body["counters"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotError("invalid_snapshot", f"{path}: {exc}") from None
        return engine

    def reload_model(self, artifact_dir: Path) -> bool:
        """True: new model active. False: load failed, the previous model keeps serving."""
        try:
            model, version = load_artifact(_path(artifact_dir, "artifact_dir"))   # outside the lock
        except (ArtifactError, OSError) as exc:
            with self._lock:
                self._counters["reload_failed"] += 1
                self._last_reload_error = getattr(exc, "code", "artifact_unreadable")
            return False
        with self._lock:
            self._model, self._model_version = model, version   # one swap of the pair
            self._counters["reload_succeeded"] += 1
            self._last_reload_error = None
        return True

    def stats(self) -> Mapping[str, object]:
        with self._lock:
            return {
                "resident_entities": len(self._recency),
                "max_entities": self._max_entities,
                "evicted_remembered": len(self._evicted),
                "model_version": self._model_version,
                "last_reload_error": self._last_reload_error,
                "counters": dict(sorted(self._counters.items())),
            }

    def _evict_least_recent(self) -> None:          # caller holds the lock
        entity_id, _ = self._recency.popitem(last=False)
        self._store.forget(entity_id)
        self._evicted[entity_id] = None
        if len(self._evicted) > self._max_entities:      # the evicted list is bounded too
            self._evicted.popitem(last=False)
        self._counters["evicted"] += 1

    def _count(self, key: str) -> None:
        with self._lock:
            self._counters[key] += 1


def _path(value, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise InvalidArgumentError("not_a_path", f"{name} must be a path, got {type(value).__name__}")
    return Path(value)
