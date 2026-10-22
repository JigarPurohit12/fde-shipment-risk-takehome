"""Step 4: label window, maturity, reporting lag, and MANIFEST.json."""
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from _factories import label_record

from shipment_risk._io import read_manifest
from shipment_risk._labels import (
    HORIZON,
    LABEL_REPORTING_LAG,
    check_reporting_lag,
    incidents_in_window,
    is_mature,
)
from shipment_risk._records import validate_label
from shipment_risk._time import format_timestamp, parse_timestamp
from shipment_risk.errors import InvalidRecordError

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
AS_OF = parse_timestamp("2026-01-01T14:00:00Z", "t")
US = timedelta(microseconds=1)


def label_at(incident_at, available_at=None, incident_id="inc-1"):
    incident_text = format_timestamp(incident_at)
    available_text = format_timestamp(available_at or incident_at + timedelta(hours=1))
    return validate_label(label_record(incident_at=incident_text,
                                       label_available_at=available_text,
                                       incident_id=incident_id))


class WindowTests(unittest.TestCase):
    far_future = AS_OF + timedelta(days=30)

    def found(self, incident_at, known_by=None):
        return incidents_in_window([label_at(incident_at)], AS_OF, known_by or self.far_future)

    def test_window_is_open_at_as_of_and_closed_at_horizon(self):
        self.assertEqual(self.found(AS_OF), [])                          # already happening
        self.assertEqual(len(self.found(AS_OF + US)), 1)
        self.assertEqual(len(self.found(AS_OF + HORIZON)), 1)            # exactly 6 h counts
        self.assertEqual(self.found(AS_OF + HORIZON + US), [])

    def test_label_must_be_available_by_the_cutoff(self):
        incident = AS_OF + timedelta(hours=4)
        label = label_at(incident, available_at=incident + timedelta(hours=20))
        self.assertEqual(incidents_in_window([label], AS_OF, incident + timedelta(hours=20)), [label])
        self.assertEqual(incidents_in_window([label], AS_OF, incident + timedelta(hours=20) - US), [])

    def test_results_are_sorted(self):
        later = label_at(AS_OF + timedelta(hours=5), incident_id="inc-b")
        earlier = label_at(AS_OF + timedelta(hours=1), incident_id="inc-a")
        found = incidents_in_window([later, earlier], AS_OF, self.far_future)
        self.assertEqual([l.incident_id for l in found], ["inc-a", "inc-b"])


class MaturityTests(unittest.TestCase):
    def test_mature_exactly_at_horizon_plus_lag(self):
        boundary = AS_OF + HORIZON + LABEL_REPORTING_LAG
        self.assertTrue(is_mature(AS_OF, boundary))
        self.assertFalse(is_mature(AS_OF, boundary - US))

    def test_reporting_lag_must_match_the_manifest(self):
        check_reporting_lag(30)
        with self.assertRaises(InvalidRecordError) as ctx:
            check_reporting_lag(48)
        self.assertEqual(ctx.exception.code, "label_lag_mismatch")


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "MANIFEST.json"
        self.good = {"files": {"events.jsonl": 3}, "label_reporting_lag_hours": 30,
                     "observation_end": "2026-02-11T12:00:00Z", "schema_version": 1,
                     "seed": 1, "size": 1}

    def write(self, data=None, text=None):
        self.path.write_text(text if text is not None else json.dumps(data, indent=2), encoding="utf-8")

    def assert_code(self, code):
        with self.assertRaises(InvalidRecordError) as ctx:
            read_manifest(self.path)
        self.assertEqual(ctx.exception.code, code)

    def test_valid_manifest(self):
        self.write(self.good)
        manifest = read_manifest(self.path)
        self.assertEqual(manifest.observation_end, parse_timestamp("2026-02-11T12:00:00Z", "t"))
        self.assertEqual((manifest.label_reporting_lag_hours, manifest.files), (30, {"events.jsonl": 3}))

    def test_bad_manifests(self):
        cases = [
            (dict(self.good, schema_version=2), "unsupported_schema_version"),
            (dict(self.good, schema_version=True), "unsupported_schema_version"),
            (dict(self.good, observation_end="2026-02-11T12:00:00"), "missing_utc_offset"),
            (dict(self.good, label_reporting_lag_hours="30"), "invalid_label_lag"),
            (dict(self.good, label_reporting_lag_hours=None), "invalid_label_lag"),
            (dict(self.good, files=[]), "invalid_manifest"),
        ]
        for data, code in cases:
            with self.subTest(code=code, data=data):
                self.write(data)
                self.assert_code(code)

    def test_strict_json_rules_apply(self):
        self.write(text='{"schema_version": 1, "schema_version": 1}')
        self.assert_code("duplicate_json_key")
        self.write(text='{"schema_version": NaN}')
        self.assert_code("non_finite_value")

    @unittest.skipUnless((DATA_DIR / "MANIFEST.json").exists(), "run tools/generate_dataset.py first")
    def test_supplied_manifest_matches_the_code(self):
        manifest = read_manifest(DATA_DIR / "MANIFEST.json")
        check_reporting_lag(manifest.label_reporting_lag_hours)
        self.assertEqual(manifest.observation_end.utcoffset(), timedelta(0))


if __name__ == "__main__":
    unittest.main()
