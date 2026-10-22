"""Record validation: untrusted mappings in, immutable normalized objects out.

Rules that hold for every validator here:
  * The caller's mapping is only read, never written (README: "Input
    mappings must not be mutated").
  * The result is a frozen dataclass holding our own normalized copy, so
    later changes to the caller's dict cannot change what we stored.
  * Normalization makes equal content compare equal:
      - every timestamp is converted to UTC,
      - value 4 and value 4.0 both become 4.0,
      - -0.0 becomes 0.0 (it would otherwise serialize differently).
    Later, "same identity with different content" is decided by comparing
    these normalized objects, not raw strings.
  * Unknown fields are rejected, not ignored: a new field such as
    "unit": "F" could change the meaning of value.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from ._time import format_timestamp, parse_timestamp
from .errors import InvalidRecordError

SUPPORTED_KIND = "temperature_c"
ABSOLUTE_ZERO_C = -273.15   # a Celsius reading below this is physically impossible
MAX_TEXT_LENGTH = 128       # bound on identifiers and carrier names

EVENT_FIELDS = frozenset(
    {"event_id", "revision", "entity_id", "event_at",
     "received_at", "kind", "value", "carrier"}
)
LABEL_FIELDS = frozenset({"incident_id", "entity_id", "incident_at", "label_available_at"})
DECISION_FIELDS = frozenset({"entity_id", "as_of"})


@dataclass(frozen=True, slots=True)
class Event:
    """One validated sensor reading (one revision of one event_id)."""

    entity_id: str
    event_id: str
    revision: int
    event_at: datetime      # device clock, UTC
    received_at: datetime   # platform receipt time, UTC
    kind: str
    value: float
    carrier: str

    @property
    def identity(self) -> tuple[str, str, int]:
        """The brief's identity (event_id, revision), scoped to its shipment."""
        return (self.entity_id, self.event_id, self.revision)

    def to_record(self) -> dict[str, object]:
        """JSON-ready dict in canonical form (UTC 'Z' timestamps)."""
        return {
            "carrier": self.carrier,
            "entity_id": self.entity_id,
            "event_at": format_timestamp(self.event_at),
            "event_id": self.event_id,
            "kind": self.kind,
            "received_at": format_timestamp(self.received_at),
            "revision": self.revision,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class Label:
    """One validated audit label (an incident and when it became known)."""

    incident_id: str
    entity_id: str
    incident_at: datetime
    label_available_at: datetime

    def to_record(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "incident_at": format_timestamp(self.incident_at),
            "incident_id": self.incident_id,
            "label_available_at": format_timestamp(self.label_available_at),
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """One validated historical decision point (shipment, moment)."""

    entity_id: str
    as_of: datetime

    def to_record(self) -> dict[str, object]:
        return {"as_of": format_timestamp(self.as_of), "entity_id": self.entity_id}


# --------------------------------------------------------------------------
# Public validators
# --------------------------------------------------------------------------

def validate_event(record: object) -> Event:
    """Validate one events.jsonl record. Raises InvalidRecordError."""
    fields = _check_fields(record, EVENT_FIELDS, "event")
    return Event(
        entity_id=_text(fields, "entity_id", "event"),
        event_id=_text(fields, "event_id", "event"),
        revision=_revision(fields, "event"),
        event_at=parse_timestamp(fields["event_at"], "event.event_at"),
        received_at=parse_timestamp(fields["received_at"], "event.received_at"),
        kind=_kind(fields, "event"),
        value=_temperature(fields, "event"),
        carrier=_text(fields, "carrier", "event"),
    )


def validate_label(record: object) -> Label:
    """Validate one labels.jsonl record. Raises InvalidRecordError."""
    fields = _check_fields(record, LABEL_FIELDS, "label")
    incident_at = parse_timestamp(fields["incident_at"], "label.incident_at")
    available_at = parse_timestamp(fields["label_available_at"], "label.label_available_at")
    if available_at < incident_at:
        raise InvalidRecordError(
            "label_before_incident",
            "label.label_available_at is earlier than label.incident_at",
        )
    return Label(
        incident_id=_text(fields, "incident_id", "label"),
        entity_id=_text(fields, "entity_id", "label"),
        incident_at=incident_at,
        label_available_at=available_at,
    )


def validate_decision(record: object) -> Decision:
    """Validate one decisions.jsonl record. Raises InvalidRecordError."""
    fields = _check_fields(record, DECISION_FIELDS, "decision")
    return Decision(
        entity_id=_text(fields, "entity_id", "decision"),
        as_of=parse_timestamp(fields["as_of"], "decision.as_of"),
    )


# --------------------------------------------------------------------------
# Field helpers
# --------------------------------------------------------------------------

def _check_fields(record: object, allowed: frozenset[str], where: str) -> Mapping[str, object]:
    """Require a mapping with exactly the allowed keys (none missing, none extra)."""
    if not isinstance(record, Mapping):
        raise InvalidRecordError(
            "not_a_mapping",
            f"{where} record must be a mapping, got {type(record).__name__}",
        )
    keys = set(record)
    missing = sorted(allowed - keys)          # sorted -> deterministic message
    if missing:
        raise InvalidRecordError(
            "missing_field", f"{where} record is missing {', '.join(missing)}"
        )
    unknown = keys - allowed
    if unknown:
        # repr() because a key might not even be a string, and sorted() cannot
        # compare mixed types.
        names = ", ".join(sorted(repr(key) for key in unknown))
        raise InvalidRecordError(
            "unknown_field", f"{where} record has unexpected field(s) {names}"
        )
    return record


def _text(fields: Mapping[str, object], name: str, where: str) -> str:
    """Identifiers and names: non-empty, bounded, no hidden whitespace/control chars."""
    value = fields[name]
    if not isinstance(value, str):
        raise InvalidRecordError(
            "invalid_string",
            f"{where}.{name} must be a string, got {type(value).__name__}",
        )
    if not 1 <= len(value) <= MAX_TEXT_LENGTH:
        raise InvalidRecordError(
            "invalid_string", f"{where}.{name} must be 1-{MAX_TEXT_LENGTH} characters"
        )
    # " e-1" and "e-1" would otherwise be two different shipments.
    if value != value.strip() or not value.isprintable():
        raise InvalidRecordError(
            "invalid_string",
            f"{where}.{name}={value!r} has surrounding whitespace or control characters",
        )
    return str(value)


def _revision(fields: Mapping[str, object], where: str) -> int:
    value = fields["revision"]
    # type(...) is int (not isinstance) rejects True/False, because bool is a
    # subclass of int in Python, and rejects 1.0 and "1".
    if type(value) is not int or value < 1:
        raise InvalidRecordError(
            "invalid_revision",
            f"{where}.revision must be a positive integer, got {value!r}",
        )
    return value


def _kind(fields: Mapping[str, object], where: str) -> str:
    value = _text(fields, "kind", where)
    if value != SUPPORTED_KIND:
        raise InvalidRecordError(
            "unsupported_kind",
            f"{where}.kind={value!r}; only {SUPPORTED_KIND!r} is supported",
        )
    return value


def _temperature(fields: Mapping[str, object], where: str) -> float:
    value = fields["value"]
    # Exact type check rejects bool (an int subclass), numeric strings and None.
    if type(value) not in (int, float):
        raise InvalidRecordError(
            "invalid_number",
            f"{where}.value must be a number, got {type(value).__name__}",
        )
    try:
        number = float(value)
    except OverflowError:  # an int too large for a float, e.g. 10**400
        raise InvalidRecordError("non_finite_value", f"{where}.value is too large") from None
    if not math.isfinite(number):  # NaN, inf, -inf
        raise InvalidRecordError("non_finite_value", f"{where}.value must be finite")
    if number < ABSOLUTE_ZERO_C:
        raise InvalidRecordError(
            "below_absolute_zero", f"{where}.value={number} is below {ABSOLUTE_ZERO_C} C"
        )
    return number + 0.0  # -0.0 + 0.0 == +0.0, so equal readings serialize equally
