"""Step 3: the shared feature function and feature digest."""
import math
import unittest
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from unittest import mock

from _factories import make_event

from shipment_risk import _features
from shipment_risk._canonical import canonical_json, sha256_digest
from shipment_risk._features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    LOOKBACK,
    NO_READINGS_IN_WINDOW,
    compute_features,
)
from shipment_risk._io import read_jsonl
from shipment_risk._records import validate_decision, validate_event
from shipment_risk._store import CLOCK_SKEW_TOLERANCE, FUTURE_DEVICE_CLOCK, EventStore
from shipment_risk._time import format_timestamp, parse_timestamp

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
AS_OF = parse_timestamp("2026-01-01T11:30:00Z", "t")


def at(text):
    return parse_timestamp(text, "t")


def features_for(*readings, as_of=AS_OF):
    """readings: (event_id, event_at text, value[, carrier]); received 2 min after event_at."""
    store = EventStore()
    for item in readings:
        event_id, event_at, value = item[:3]
        carrier = item[3] if len(item) > 3 else "coast"
        received = format_timestamp(at(event_at) + timedelta(minutes=2))
        store.add(make_event(event_id=event_id, event_at=event_at, received_at=received,
                             value=value, carrier=carrier))
    return compute_features(store.known_at("ship-1", as_of))


class FeatureValueTests(unittest.TestCase):
    def test_names_are_fixed(self):
        self.assertEqual(set(features_for().values), set(FEATURE_NAMES))

    def test_basic_values(self):
        fs = features_for(("r-9", "2026-01-01T09:00:00Z", 4.0),
                          ("r-10", "2026-01-01T10:00:00Z", 5.0),
                          ("r-11", "2026-01-01T11:00:00Z", 6.0))
        self.assertEqual(fs.values, {
            "carrier": "coast",
            "hours_since_latest_reading": 0.5,
            "latest_temp_c": 6.0,
            "max_temp_c": 6.0,
            "mean_temp_c": 5.0,
            "reading_count": 3,
            "trend_c_per_hour": 1.0,
        })
        self.assertEqual(fs.reasons, ())

    def test_window_start_is_exclusive(self):
        edge = format_timestamp(AS_OF - LOOKBACK)
        just_inside = format_timestamp(AS_OF - LOOKBACK + timedelta(microseconds=1))
        self.assertEqual(features_for(("r-1", edge, 4.0)).values["reading_count"], 0)
        self.assertEqual(features_for(("r-1", just_inside, 4.0)).values["reading_count"], 1)

    def test_no_readings_are_none_not_invented_numbers(self):
        fs = features_for()
        self.assertEqual(fs.values["reading_count"], 0)
        self.assertEqual(fs.values["hours_since_latest_reading"], LOOKBACK / timedelta(hours=1))
        for name in ("latest_temp_c", "max_temp_c", "mean_temp_c", "trend_c_per_hour", "carrier"):
            self.assertIsNone(fs.values[name], name)
        self.assertEqual(fs.reasons, (NO_READINGS_IN_WINDOW,))

    def test_only_old_readings_still_give_a_carrier(self):
        fs = features_for(("r-1", "2026-01-01T01:00:00Z", 4.0, "north"))
        self.assertEqual(fs.values["reading_count"], 0)
        self.assertEqual(fs.values["carrier"], "north")
        self.assertEqual(fs.reasons, (NO_READINGS_IN_WINDOW,))

    def test_trend_needs_two_distinct_times(self):
        self.assertIsNone(features_for(("r-1", "2026-01-01T10:00:00Z", 4.0)).values["trend_c_per_hour"])
        same_time = features_for(("r-1", "2026-01-01T10:00:00Z", 4.0),
                                 ("r-2", "2026-01-01T10:00:00Z", 6.0))
        self.assertIsNone(same_time.values["trend_c_per_hour"])
        self.assertEqual(same_time.values["latest_temp_c"], 6.0)   # event_id breaks the tie

    def test_flat_temperatures_give_positive_zero_trend(self):
        trend = features_for(("r-1", "2026-01-01T09:00:00Z", 4.0),
                             ("r-2", "2026-01-01T10:00:00Z", 4.0)).values["trend_c_per_hour"]
        self.assertEqual(math.copysign(1.0, trend), 1.0)

    def test_device_clock_far_in_the_past_is_ignored_by_the_window(self):
        fs = features_for(("r-old", "1970-01-01T00:00:00Z", 30.0),
                          ("r-1", "2026-01-01T11:00:00Z", 4.0))
        self.assertEqual((fs.values["reading_count"], fs.values["max_temp_c"]), (1, 4.0))

    def test_small_clock_skew_reading_counts_with_zero_age(self):
        store = EventStore()
        store.add(make_event(event_at=format_timestamp(AS_OF + CLOCK_SKEW_TOLERANCE),
                             received_at=format_timestamp(AS_OF)))
        fs = compute_features(store.known_at("ship-1", AS_OF))
        self.assertEqual((fs.values["reading_count"], fs.values["hours_since_latest_reading"]), (1, 0.0))

    def test_future_clock_exclusion_becomes_a_reason(self):
        store = EventStore()
        store.add(make_event(event_at="2026-01-01T15:00:00Z", received_at="2026-01-01T10:00:00Z"))
        fs = compute_features(store.known_at("ship-1", AS_OF))
        self.assertEqual(fs.reasons, (FUTURE_DEVICE_CLOCK, NO_READINGS_IN_WINDOW))

    def test_latest_carrier_wins_and_unseen_carriers_are_fine(self):
        fs = features_for(("r-1", "2026-01-01T09:00:00Z", 4.0, "coast"),
                          ("r-2", "2026-01-01T10:00:00Z", 4.0, "arctic-express"))
        self.assertEqual(fs.values["carrier"], "arctic-express")


class DigestTests(unittest.TestCase):
    def test_digest_covers_schema_version_and_values(self):
        fs = features_for(("r-1", "2026-01-01T10:00:00Z", 4.0))
        self.assertEqual(fs.digest, sha256_digest({"schema": FEATURE_SCHEMA_VERSION, "features": fs.values}))

    def test_equal_features_equal_digest_and_any_change_changes_it(self):
        a = features_for(("r-1", "2026-01-01T10:00:00Z", 4.0))
        b = features_for(("r-1", "2026-01-01T10:00:00Z", 4.0))
        c = features_for(("r-1", "2026-01-01T10:00:00Z", 4.001))
        self.assertEqual(a.digest, b.digest)
        self.assertNotEqual(a.digest, c.digest)

    def test_new_feature_definitions_change_the_digest(self):
        before = features_for(("r-1", "2026-01-01T10:00:00Z", 4.0)).digest
        with mock.patch.object(_features, "FEATURE_SCHEMA_VERSION", "features-v2"):
            after = features_for(("r-1", "2026-01-01T10:00:00Z", 4.0)).digest
        self.assertNotEqual(before, after)

    def test_values_are_canonical_json_safe(self):
        fs = features_for(("r-1", "2026-01-01T10:00:00Z", 4.0))
        canonical_json(fs.values)   # raises on NaN/inf or non-JSON types


@unittest.skipUnless((DATA_DIR / "events.jsonl").exists(), "run tools/generate_dataset.py first")
class SampleDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = [validate_event(r) for r in read_jsonl(DATA_DIR / "events.jsonl")]
        cls.decisions = [validate_decision(r) for r in read_jsonl(DATA_DIR / "decisions.jsonl")]
        cls.store = EventStore()
        for e in cls.events:
            cls.store.add(e)
        cls.by_entity = defaultdict(list)
        for e in cls.events:
            cls.by_entity[e.entity_id].append(e)

    def test_records_received_after_as_of_never_change_features(self):
        # Full store (knows the future) vs a store fed ONLY what had been
        # received by as_of: the features must be byte-identical.
        for d in self.decisions:
            past_only = EventStore()
            for e in self.by_entity[d.entity_id]:
                if e.received_at <= d.as_of:
                    past_only.add(e)
            full = compute_features(self.store.known_at(d.entity_id, d.as_of))
            honest = compute_features(past_only.known_at(d.entity_id, d.as_of))
            self.assertEqual(full.digest, honest.digest, msg=f"{d}")

    def test_every_decision_gets_complete_json_safe_features(self):
        for d in self.decisions:
            fs = compute_features(self.store.known_at(d.entity_id, d.as_of))
            self.assertEqual(set(fs.values), set(FEATURE_NAMES))
            canonical_json(fs.values)


if __name__ == "__main__":
    unittest.main()
