"""Step 2: event store, revisions, duplicates, conflicts, device clocks."""
import itertools
import random
import unittest
from datetime import timedelta
from pathlib import Path

from _factories import make_event

from shipment_risk._io import read_jsonl
from shipment_risk._records import validate_decision, validate_event
from shipment_risk._store import (
    ACCEPTED,
    CLOCK_SKEW_TOLERANCE,
    DUPLICATE,
    FUTURE_DEVICE_CLOCK,
    SUPERSEDED_ON_ARRIVAL,
    EventStore,
)
from shipment_risk._time import format_timestamp, parse_timestamp
from shipment_risk.errors import ConflictError

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def at(text):
    return parse_timestamp(text, "t")


def reading(event_id="reading-1", revision=1, received="2026-01-01T08:00:00Z",
            event_at=None, value=4.0, entity_id="ship-1"):
    """Shorthand: device time defaults to 2 minutes before receipt."""
    if event_at is None:
        event_at = format_timestamp(at(received) - timedelta(minutes=2))
    return make_event(event_id=event_id, revision=revision, received_at=received,
                      event_at=event_at, value=value, entity_id=entity_id)


def values(known):
    return [e.value for e in known.readings]


class VisibilityTests(unittest.TestCase):
    def test_known_from_the_receipt_instant_inclusive(self):
        store = EventStore()
        store.add(reading(received="2026-01-01T08:00:00Z"))
        self.assertEqual(len(store.known_at("ship-1", at("2026-01-01T08:00:00Z")).readings), 1)
        self.assertEqual(len(store.known_at("ship-1", at("2026-01-01T07:59:59.999999Z")).readings), 0)

    def test_unknown_shipment_has_nothing_known(self):
        known = EventStore().known_at("never-seen", at("2026-01-01T08:00:00Z"))
        self.assertEqual((known.readings, known.excluded), ((), ()))

    def test_readings_sorted_by_device_time_then_event_id(self):
        store = EventStore()
        store.add(reading("r-b", event_at="2026-01-01T03:00:00Z", received="2026-01-01T03:01:00Z"))
        store.add(reading("r-c", event_at="2026-01-01T01:00:00Z", received="2026-01-01T05:00:00Z"))
        store.add(reading("r-a", event_at="2026-01-01T03:00:00Z", received="2026-01-01T03:02:00Z"))
        known = store.known_at("ship-1", at("2026-01-01T06:00:00Z"))
        self.assertEqual([e.event_id for e in known.readings], ["r-c", "r-a", "r-b"])


class DuplicateAndConflictTests(unittest.TestCase):
    def test_exact_duplicate_is_ignored(self):
        store = EventStore()
        self.assertEqual(store.add(reading()).status, ACCEPTED)
        result = store.add(reading())
        self.assertEqual((result.status, result.changed), (DUPLICATE, False))
        self.assertEqual(len(store.events()), 1)

    def test_duplicate_written_with_another_offset_is_still_a_duplicate(self):
        store = EventStore()
        store.add(reading(received="2026-01-01T08:00:00Z", event_at="2026-01-01T07:58:00Z"))
        again = reading(received="2026-01-01T13:30:00+05:30", event_at="2026-01-01T13:28:00+05:30")
        self.assertEqual(store.add(again).status, DUPLICATE)

    def test_conflict_is_rejected_and_first_version_kept(self):
        store = EventStore()
        store.add(reading(value=4.0))
        before = store.events()
        with self.assertRaises(ConflictError) as ctx:
            store.add(reading(value=9.0))
        self.assertEqual(ctx.exception.code, "conflict")
        self.assertIn("value", ctx.exception.detail)
        self.assertEqual(store.events(), before)
        self.assertEqual(values(store.known_at("ship-1", at("2026-01-02T00:00:00Z"))), [4.0])

    def test_different_receipt_time_is_also_a_conflict(self):
        store = EventStore()
        store.add(reading(received="2026-01-01T08:00:00Z", event_at="2026-01-01T07:58:00Z"))
        with self.assertRaises(ConflictError):
            store.add(reading(received="2026-01-01T09:00:00Z", event_at="2026-01-01T07:58:00Z"))

    def test_same_event_id_in_two_shipments_are_separate_readings(self):
        store = EventStore()
        self.assertEqual(store.add(reading(entity_id="ship-1", value=4.0)).status, ACCEPTED)
        self.assertEqual(store.add(reading(entity_id="ship-2", value=9.0)).status, ACCEPTED)
        moment = at("2026-01-02T00:00:00Z")
        self.assertEqual(values(store.known_at("ship-1", moment)), [4.0])
        self.assertEqual(values(store.known_at("ship-2", moment)), [9.0])


class RevisionTests(unittest.TestCase):
    def test_correction_after_a_decision_does_not_rewrite_it(self):
        store = EventStore()
        store.add(reading(revision=1, received="2026-01-01T11:00:00Z", value=3.9))
        decision_time = at("2026-01-01T14:00:00Z")
        before = store.known_at("ship-1", decision_time)

        result = store.add(reading(revision=2, received="2026-01-01T19:00:00Z", value=1.9))
        self.assertEqual((result.status, result.changed), (ACCEPTED, True))
        self.assertEqual(store.known_at("ship-1", decision_time), before)       # history unchanged
        self.assertEqual(values(store.known_at("ship-1", at("2026-01-01T19:00:00Z"))), [1.9])

    def test_reversed_delivery_with_consistent_receipt_times(self):
        # Flavor (a): rev 2 is DELIVERED first but was RECEIVED later.
        store = EventStore()
        store.add(reading(revision=2, received="2026-01-01T19:00:00Z", value=1.9))
        result = store.add(reading(revision=1, received="2026-01-01T11:00:00Z", value=3.9))
        self.assertEqual((result.status, result.changed), (ACCEPTED, True))
        self.assertEqual(values(store.known_at("ship-1", at("2026-01-01T12:00:00Z"))), [3.9])
        self.assertEqual(values(store.known_at("ship-1", at("2026-01-01T20:00:00Z"))), [1.9])

    def test_truly_reversed_receipt_never_lets_the_lower_revision_win(self):
        # Flavor (b): rev 2 was RECEIVED before rev 1.
        store = EventStore()
        store.add(reading(revision=2, received="2026-01-01T08:10:00Z", value=1.9))
        result = store.add(reading(revision=1, received="2026-01-01T11:00:00Z", value=3.9))
        self.assertEqual((result.status, result.changed), (SUPERSEDED_ON_ARRIVAL, False))
        self.assertEqual(values(store.known_at("ship-1", at("2026-01-01T08:05:00Z"))), [])
        self.assertEqual(values(store.known_at("ship-1", at("2026-01-01T12:00:00Z"))), [1.9])
        self.assertEqual(len(store.events()), 2)    # still stored for audit/snapshots

    def test_same_receipt_instant_highest_revision_wins(self):
        for order in ((1, 2), (2, 1)):
            store = EventStore()
            for revision in order:
                store.add(reading(revision=revision, received="2026-01-01T10:00:00Z", value=float(revision)))
            with self.subTest(order=order):
                self.assertEqual(values(store.known_at("ship-1", at("2026-01-01T10:00:00Z"))), [2.0])

    def test_revision_gap_and_correction_without_original(self):
        store = EventStore()
        store.add(reading(revision=3, received="2026-01-01T12:00:00Z", value=3.0))
        self.assertEqual(values(store.known_at("ship-1", at("2026-01-01T12:00:00Z"))), [3.0])
        store.add(reading(revision=1, received="2026-01-01T09:00:00Z", value=1.0))
        self.assertEqual(values(store.known_at("ship-1", at("2026-01-01T10:00:00Z"))), [1.0])
        self.assertEqual(values(store.known_at("ship-1", at("2026-01-01T12:00:00Z"))), [3.0])


class ClockPolicyTests(unittest.TestCase):
    def test_small_skew_is_tolerated(self):
        received = at("2026-01-01T08:00:00Z")
        store = EventStore()
        store.add(reading(received=format_timestamp(received),
                          event_at=format_timestamp(received + CLOCK_SKEW_TOLERANCE)))
        known = store.known_at("ship-1", received)
        self.assertEqual((len(known.readings), known.excluded), (1, ()))

    def test_future_device_clock_is_excluded_and_counted(self):
        received = at("2026-01-01T08:00:00Z")
        for ahead in (CLOCK_SKEW_TOLERANCE + timedelta(microseconds=1), timedelta(hours=4)):
            store = EventStore()
            store.add(reading(received=format_timestamp(received),
                              event_at=format_timestamp(received + ahead)))
            known = store.known_at("ship-1", received + timedelta(hours=8))
            with self.subTest(ahead=ahead):
                self.assertEqual(known.readings, ())
                self.assertEqual(known.excluded, ((FUTURE_DEVICE_CLOCK, 1),))

    def test_bad_clock_correction_does_not_fall_back_to_older_revision(self):
        store = EventStore()
        store.add(reading(revision=1, received="2026-01-01T08:00:00Z", value=4.0))
        store.add(reading(revision=2, received="2026-01-01T09:00:00Z",
                          event_at="2026-01-01T13:00:00Z", value=5.0))
        self.assertEqual(values(store.known_at("ship-1", at("2026-01-01T08:30:00Z"))), [4.0])
        known = store.known_at("ship-1", at("2026-01-01T09:00:00Z"))
        self.assertEqual((known.readings, known.excluded), ((), ((FUTURE_DEVICE_CLOCK, 1),)))

    def test_correction_that_fixes_the_clock_makes_the_reading_usable(self):
        store = EventStore()
        store.add(reading(revision=1, received="2026-01-01T08:00:00Z",
                          event_at="2026-01-01T12:00:00Z", value=4.0))
        store.add(reading(revision=2, received="2026-01-01T09:00:00Z", value=4.5))
        self.assertEqual(values(store.known_at("ship-1", at("2026-01-01T08:30:00Z"))), [])
        self.assertEqual(values(store.known_at("ship-1", at("2026-01-01T09:00:00Z"))), [4.5])

    def test_very_old_device_clock_is_kept_here(self):
        # A sensor reset to 1970 is usable at the store level; step 3's
        # look-back window is what keeps it out of the features.
        store = EventStore()
        store.add(reading(event_at="1970-01-01T00:00:00Z"))
        self.assertEqual(len(store.known_at("ship-1", at("2026-01-01T09:00:00Z")).readings), 1)


def assert_changed_matches_oracle(test, events_in_order):
    """`changed` must be True exactly when some as-of answer changes.

    Answers can only change at receipt instants, so the oracle compares the
    shipment's answers at every receipt instant before and after each add.
    """
    probes = {}
    for e in events_in_order:
        probes.setdefault(e.entity_id, set()).add(e.received_at)
    store = EventStore()
    for event in events_in_order:
        times = sorted(probes[event.entity_id])
        before = [store.known_at(event.entity_id, t) for t in times]
        result = store.add(event)
        after = [store.known_at(event.entity_id, t) for t in times]
        test.assertEqual(result.changed, before != after, msg=f"{event} -> {result}")
    return store


class ChangedFlagOracleTests(unittest.TestCase):
    def test_every_delivery_order_of_a_tricky_set(self):
        tricky = [
            reading(revision=1, received="2026-01-01T11:00:00Z", value=3.9),
            reading(revision=2, received="2026-01-01T19:00:00Z", value=1.9),
            reading(revision=3, received="2026-01-01T15:00:00Z",
                    event_at="2026-01-01T19:00:00Z", value=7.0),          # bad clock
            make_event(event_id="reading-1", revision=1, value=3.9,           # duplicate,
                       received_at="2026-01-01T16:30:00+05:30",               # other offset
                       event_at="2026-01-01T16:28:00+05:30"),
            reading(event_id="reading-2", received="2026-01-01T10:00:00Z"),
        ]
        final_states = set()
        for order in itertools.permutations(tricky):
            store = assert_changed_matches_oracle(self, list(order))
            final_states.add(tuple(store.events()))
        self.assertEqual(len(final_states), 1)     # delivery order never matters


@unittest.skipUnless((DATA_DIR / "events.jsonl").exists(), "run tools/generate_dataset.py first")
class SampleDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = [validate_event(r) for r in read_jsonl(DATA_DIR / "events.jsonl")]
        cls.decisions = [validate_decision(r) for r in read_jsonl(DATA_DIR / "decisions.jsonl")]

    def build(self, events):
        store = EventStore()
        statuses = [store.add(e).status for e in events]
        self.assertEqual(len(statuses), len(events))
        return store

    def test_delivery_order_does_not_change_any_answer(self):
        shuffled = list(self.events)
        random.Random(0).shuffle(shuffled)
        stores = [self.build(self.events), self.build(self.events[::-1]), self.build(shuffled)]
        reference = stores[0]
        for other in stores[1:]:
            self.assertEqual(other.events(), reference.events())
            for d in self.decisions:
                self.assertEqual(other.known_at(d.entity_id, d.as_of),
                                 reference.known_at(d.entity_id, d.as_of))

    def test_nothing_received_after_as_of_is_ever_used(self):
        store = self.build(self.events)
        for d in self.decisions:
            for e in store.known_at(d.entity_id, d.as_of).readings:
                self.assertLessEqual(e.received_at, d.as_of)

    def test_changed_flag_matches_oracle_on_real_deliveries(self):
        assert_changed_matches_oracle(self, self.events)


if __name__ == "__main__":
    unittest.main()
