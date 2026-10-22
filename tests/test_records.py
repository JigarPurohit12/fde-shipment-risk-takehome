"""Step 1: event, label and decision validation."""
import copy
import dataclasses
import json
import math
import unittest
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

from _factories import decision_record, event_record, label_record

from shipment_risk._io import read_jsonl
from shipment_risk._records import (
    EVENT_FIELDS,
    validate_decision,
    validate_event,
    validate_label,
)
from shipment_risk._time import UTC
from shipment_risk.errors import InvalidRecordError

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class EventValidationTests(unittest.TestCase):
    def assert_code(self, record, code):
        with self.assertRaises(InvalidRecordError) as ctx:
            validate_event(record)
        self.assertEqual(ctx.exception.code, code)

    def test_valid_event_is_normalized(self):
        event = validate_event(event_record(event_at="2026-01-01T05:30:00+05:30", value=4))
        self.assertEqual(event.event_at, datetime(2026, 1, 1, tzinfo=UTC))
        self.assertIs(type(event.value), float)
        self.assertEqual(event.value, 4.0)
        self.assertEqual(event.identity, ("ship-1", "reading-1", 1))

    def test_negative_zero_becomes_positive_zero(self):
        event = validate_event(event_record(value=-0.0))
        self.assertEqual(math.copysign(1.0, event.value), 1.0)

    def test_equivalent_offsets_give_equal_events(self):
        a = validate_event(event_record(event_at="2026-01-01T05:30:00+05:30"))
        b = validate_event(event_record(event_at="2026-01-01T00:00:00Z"))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_to_record_round_trips(self):
        event = validate_event(event_record(received_at="2026-01-01T03:02:00+03:00"))
        record = event.to_record()
        self.assertEqual(record["received_at"], "2026-01-01T00:02:00Z")
        self.assertEqual(validate_event(record), event)

    def test_input_mapping_is_not_mutated(self):
        record = event_record(event_at="2026-01-01T05:30:00+05:30", value=4)
        before = copy.deepcopy(record)
        validate_event(record)
        self.assertEqual(record, before)

    def test_read_only_mapping_is_accepted(self):
        event = validate_event(MappingProxyType(event_record()))
        self.assertEqual(event.value, 4.25)

    def test_result_is_frozen_and_independent_of_caller(self):
        record = event_record()
        event = validate_event(record)
        record["value"] = 99.0
        self.assertEqual(event.value, 4.25)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            event.value = 1.0

    def test_non_mappings_are_rejected(self):
        for value in (None, [], "text", 5, [("event_id", "x")]):
            with self.subTest(value=value):
                self.assert_code(value, "not_a_mapping")

    def test_each_missing_field_is_rejected(self):
        for name in sorted(EVENT_FIELDS):
            record = event_record()
            del record[name]
            with self.subTest(missing=name):
                self.assert_code(record, "missing_field")

    def test_unknown_fields_are_rejected(self):
        self.assert_code(event_record(unit="F"), "unknown_field")
        record = event_record()
        record[1] = "non-string key"
        self.assert_code(record, "unknown_field")

    def test_bad_revisions_are_rejected(self):
        for value in (0, -1, True, False, 1.0, "1", None):
            with self.subTest(revision=value):
                self.assert_code(event_record(revision=value), "invalid_revision")

    def test_good_revisions_are_accepted(self):
        for value in (1, 2, 1000):
            self.assertEqual(validate_event(event_record(revision=value)).revision, value)

    def test_bad_values_are_rejected(self):
        cases = [
            (True, "invalid_number"),
            ("4.2", "invalid_number"),
            (None, "invalid_number"),
            ([4.2], "invalid_number"),
            (float("nan"), "non_finite_value"),
            (float("inf"), "non_finite_value"),
            (float("-inf"), "non_finite_value"),
            (10**400, "non_finite_value"),
            (-273.16, "below_absolute_zero"),
        ]
        for value, code in cases:
            with self.subTest(value=value):
                self.assert_code(event_record(value=value), code)

    def test_boundary_values_are_accepted(self):
        for value in (-273.15, 0, 19.0, 1e6):
            with self.subTest(value=value):
                self.assertEqual(validate_event(event_record(value=value)).value, float(value))

    def test_other_kinds_are_rejected(self):
        for kind in ("humidity_pct", "Temperature_C", "temperature_f"):
            with self.subTest(kind=kind):
                self.assert_code(event_record(kind=kind), "unsupported_kind")

    def test_bad_text_fields_are_rejected(self):
        bad_values = ("", " ship", "ship ", "ship\n1", "ship\x00", "a" * 129, 5, None)
        for name in ("entity_id", "event_id", "carrier", "kind"):
            for value in bad_values:
                with self.subTest(field=name, value=value):
                    self.assert_code(event_record(**{name: value}), "invalid_string")

    def test_longest_allowed_text_is_accepted(self):
        self.assertEqual(len(validate_event(event_record(entity_id="a" * 128)).entity_id), 128)

    def test_bad_timestamps_report_their_codes(self):
        self.assert_code(event_record(event_at="2026-01-01T00:00:00"), "missing_utc_offset")
        self.assert_code(event_record(received_at="yesterday"), "invalid_timestamp")

    def test_future_device_clock_is_not_a_validation_error(self):
        # event_at later than received_at is a clock problem handled by the
        # feature policy (next step), not a malformed record.
        event = validate_event(event_record(event_at="2026-01-01T04:00:00Z"))
        self.assertGreater(event.event_at, event.received_at)


class LabelValidationTests(unittest.TestCase):
    def test_valid_label(self):
        label = validate_label(label_record(incident_at="2026-01-01T23:30:00+05:30"))
        self.assertEqual(label.incident_at, datetime(2026, 1, 1, 18, 0, tzinfo=UTC))
        self.assertEqual(validate_label(label.to_record()), label)

    def test_available_at_same_instant_as_incident_is_allowed(self):
        validate_label(label_record(label_available_at="2026-01-01T18:00:00Z"))

    def test_label_available_before_incident_is_rejected(self):
        with self.assertRaises(InvalidRecordError) as ctx:
            validate_label(label_record(label_available_at="2026-01-01T17:59:59Z"))
        self.assertEqual(ctx.exception.code, "label_before_incident")

    def test_missing_and_unknown_fields(self):
        record = label_record()
        del record["incident_id"]
        with self.assertRaises(InvalidRecordError) as ctx:
            validate_label(record)
        self.assertEqual(ctx.exception.code, "missing_field")
        with self.assertRaises(InvalidRecordError) as ctx:
            validate_label(label_record(severity="high"))
        self.assertEqual(ctx.exception.code, "unknown_field")


class DecisionValidationTests(unittest.TestCase):
    def test_valid_decision(self):
        decision = validate_decision(decision_record(as_of="2026-01-01T19:30:00+05:30"))
        self.assertEqual(decision.as_of, datetime(2026, 1, 1, 14, 0, tzinfo=UTC))
        self.assertEqual(validate_decision(decision.to_record()), decision)

    def test_bad_decisions(self):
        cases = [
            (decision_record(as_of="2026-01-01T14:00:00"), "missing_utc_offset"),
            (decision_record(entity_id=""), "invalid_string"),
            (decision_record(horizon_hours=6), "unknown_field"),
            ({"entity_id": "ship-1"}, "missing_field"),
        ]
        for record, code in cases:
            with self.subTest(record=record):
                with self.assertRaises(InvalidRecordError) as ctx:
                    validate_decision(record)
                self.assertEqual(ctx.exception.code, code)


@unittest.skipUnless((DATA_DIR / "MANIFEST.json").exists(), "run tools/generate_dataset.py first")
class SampleDataTests(unittest.TestCase):
    def test_every_supplied_record_validates(self):
        manifest = json.loads((DATA_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
        validators = {
            "events.jsonl": validate_event,
            "labels.jsonl": validate_label,
            "decisions.jsonl": validate_decision,
        }
        for name, validate in validators.items():
            with self.subTest(file=name):
                rows = read_jsonl(DATA_DIR / name)
                # Count comes from the manifest, never hard-coded.
                self.assertEqual(len(rows), manifest["files"][name])
                for row in rows:
                    validate(row)


if __name__ == "__main__":
    unittest.main()
