"""Event store: remembers every revision of every reading and answers
"what was known about this shipment at time T?".

Rules (README "Supplied data and semantics"):
  1. Identity is (entity_id, event_id, revision).
  2. Same identity + same content    -> duplicate, ignored.
     Same identity + different content -> ConflictError, the first one is kept.
  3. A revision is known at T  iff  received_at <= T   (inclusive).
  4. Among the known revisions of one event_id, the HIGHEST revision wins,
     no matter in which order they were delivered.
  5. Clock policy: if the winning revision's device clock (event_at) is more
     than CLOCK_SKEW_TOLERANCE later than its received_at, the reading is not
     used. It stays stored and is counted as "future_device_clock".

Nothing here depends on delivery order: query results are always sorted.

How later steps use this module:
  step 3  features are computed from KnownReadings
  step 4  build_training_rows fills one EventStore and calls known_at()
  step 6  RiskEngine.ingest returns AddResult.changed;
          RiskEngine.score calls known_at()  -> same code path as step 4
  step 7  snapshots save events(); eviction/retention are added here
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

from ._records import Event
from .errors import ConflictError

# Normal clock drift between a sensor and the platform. A device time further
# in the future than this (relative to receipt) means the device clock is wrong.
CLOCK_SKEW_TOLERANCE = timedelta(minutes=5)

# AddResult statuses (also used as stats counters in step 6).
ACCEPTED = "accepted"                            # stored, can win at some time
SUPERSEDED_ON_ARRIVAL = "superseded_on_arrival"  # stored, can never win
DUPLICATE = "duplicate"                          # identical copy, ignored

# Reasons a known reading is not used.
FUTURE_DEVICE_CLOCK = "future_device_clock"


@dataclass(frozen=True, slots=True)
class AddResult:
    status: str     # ACCEPTED, SUPERSEDED_ON_ARRIVAL or DUPLICATE
    changed: bool   # True iff some as-of query can now return a different answer


@dataclass(frozen=True, slots=True)
class KnownReadings:
    """Everything known about one shipment at one moment."""

    entity_id: str
    as_of: datetime
    readings: tuple[Event, ...]              # usable winners, sorted by (event_at, event_id)
    excluded: tuple[tuple[str, int], ...]    # (reason, count) pairs, sorted by reason


def clock_problem(event: Event) -> str | None:
    """Return a reason code if this reading's device time cannot be trusted."""
    if event.event_at > event.received_at + CLOCK_SKEW_TOLERANCE:
        return FUTURE_DEVICE_CLOCK
    return None


class EntityHistory:
    """All stored revisions of all readings of ONE shipment."""

    def __init__(self, entity_id: str) -> None:
        self.entity_id = entity_id
        # event_id -> {revision -> Event}
        self._revisions: dict[str, dict[int, Event]] = {}

    def add(self, event: Event) -> AddResult:
        if event.entity_id != self.entity_id:
            raise ValueError("event routed to the wrong shipment history")  # programming error

        revisions = self._revisions.setdefault(event.event_id, {})
        existing = revisions.get(event.revision)
        if existing is not None:
            if existing == event:              # normalized comparison (step 1)
                return AddResult(DUPLICATE, changed=False)
            raise ConflictError("conflict", _conflict_detail(existing, event))

        # Can this revision ever be the winner? Only if no HIGHER revision had
        # already been received by the moment this one was received. If one
        # had, that higher revision beats it at every time from its own
        # receipt onward, so no as-of answer changes.
        superseded = any(
            other.revision > event.revision and other.received_at <= event.received_at
            for other in revisions.values()
        )
        revisions[event.revision] = event      # always stored (audit, conflicts, snapshots)
        if superseded:
            return AddResult(SUPERSEDED_ON_ARRIVAL, changed=False)
        return AddResult(ACCEPTED, changed=True)

    def known_at(self, as_of: datetime) -> KnownReadings:
        """Point-in-time view. as_of must be an aware UTC datetime."""
        usable: list[Event] = []
        excluded: Counter[str] = Counter()
        for revisions in self._revisions.values():
            winner = _winning_revision(revisions, as_of)
            if winner is None:
                continue                       # not received yet at as_of
            # Revision first, clock second: if the newest known revision has a
            # bad clock, the reading is unusable; we do NOT fall back to the
            # older revision it replaced.
            problem = clock_problem(winner)
            if problem is None:
                usable.append(winner)
            else:
                excluded[problem] += 1
        usable.sort(key=lambda e: (e.event_at, e.event_id))
        return KnownReadings(
            entity_id=self.entity_id,
            as_of=as_of,
            readings=tuple(usable),
            excluded=tuple(sorted(excluded.items())),
        )

    def events(self) -> list[Event]:
        """Every stored revision, sorted by identity (for snapshots and tests)."""
        stored = [e for revisions in self._revisions.values() for e in revisions.values()]
        return sorted(stored, key=lambda e: e.identity)


class EventStore:
    """Histories for all shipments. Bounds and eviction are added in step 7."""

    def __init__(self) -> None:
        self._entities: dict[str, EntityHistory] = {}

    def add(self, event: Event) -> AddResult:
        history = self._entities.get(event.entity_id)
        if history is None:
            history = self._entities[event.entity_id] = EntityHistory(event.entity_id)
        return history.add(event)

    def known_at(self, entity_id: str, as_of: datetime) -> KnownReadings:
        """A shipment with no stored readings simply has nothing known."""
        history = self._entities.get(entity_id)
        if history is None:
            return KnownReadings(entity_id=entity_id, as_of=as_of, readings=(), excluded=())
        return history.known_at(as_of)

    def forget(self, entity_id: str) -> None:
        """Drop a shipment's whole history (used by the engine's eviction)."""
        self._entities.pop(entity_id, None)

    def entity_ids(self) -> list[str]:
        return sorted(self._entities)

    def entity_count(self) -> int:
        return len(self._entities)

    def events(self) -> list[Event]:
        """Every stored revision of every shipment, sorted by identity."""
        return [e for entity_id in self.entity_ids() for e in self._entities[entity_id].events()]


def _winning_revision(revisions: dict[int, Event], as_of: datetime) -> Event | None:
    """Highest revision among those received at or before as_of."""
    best: Event | None = None
    for event in revisions.values():
        if event.received_at <= as_of and (best is None or event.revision > best.revision):
            best = event
    return best


def _conflict_detail(existing: Event, new: Event) -> str:
    old, incoming = existing.to_record(), new.to_record()
    fields = ", ".join(name for name in sorted(old) if old[name] != incoming[name])
    entity_id, event_id, revision = new.identity
    return (
        f"event_id={event_id!r} revision={revision} of shipment {entity_id!r} "
        f"arrived again with different {fields}; the first version is kept"
    )
