"""Step 5: train() - purged temporal split, baselines, metrics, artifact."""
import random
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from _factories import training_row

from shipment_risk._canonical import canonical_json
from shipment_risk._io import read_jsonl, read_manifest
from shipment_risk._labels import HORIZON, LABEL_REPORTING_LAG
from shipment_risk._model import ARTIFACT_FILE, load_artifact
from shipment_risk._time import format_timestamp, parse_timestamp
from shipment_risk._training import _eligible_rows, _temporal_split, train_model
from shipment_risk.errors import InvalidArgumentError, InvalidRecordError
from shipment_risk.solution import build_training_rows, train

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
START = parse_timestamp("2026-01-01T00:00:00Z", "t")
MATURITY = HORIZON + LABEL_REPORTING_LAG


def synthetic_rows(count=120):
    """One decision every 4 h from a new shipment; every 6th one warms up and is positive."""
    rows = []
    for i in range(count):
        positive = i % 6 == 0
        rows.append(training_row(entity_id=f"ship-{i:03d}",
                                 as_of=format_timestamp(START + timedelta(hours=4 * i)),
                                 latest=8.5 if positive else 4.0 + (i % 5) * 0.2,
                                 label=int(positive), carrier=["coast", "north", "central"][i % 3]))
    return rows


class TrainCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def split_of(self, rows):
        eligible, _ = _eligible_rows(rows)
        return _temporal_split(eligible)


class SplitTests(TrainCase):
    def test_purged_temporal_and_shipment_grouped(self):
        split = self.split_of(synthetic_rows())
        self.assertTrue(split.train and split.test)
        for r in split.train:
            self.assertLessEqual(r.as_of + MATURITY, split.split_time)    # mature at split time
        for r in split.test:
            self.assertGreaterEqual(r.as_of, split.split_time)
        self.assertFalse({r.entity_id for r in split.train} & {r.entity_id for r in split.test})
        self.assertGreater(split.purged, 0)
        self.assertEqual(len(split.train) + len(split.test) + split.purged + split.grouped_out, 120)

    def test_shipment_seen_in_test_is_removed_from_train(self):
        rows = synthetic_rows()
        rows.append(training_row(entity_id="ship-119", as_of=format_timestamp(START)))  # early row, test shipment
        split = self.split_of(rows)
        self.assertEqual(split.grouped_out, 1)
        self.assertNotIn("ship-119", {r.entity_id for r in split.train})

    def test_train_labels_use_only_what_was_known_at_split_time(self):
        rows = synthetic_rows()
        # Positive whose audit label arrived long after the split: unknown at
        # split time, so it trains as a negative and is counted.
        rows[0] = training_row(entity_id="ship-000", as_of=rows[0]["as_of"], latest=8.5, label=1,
                               label_available_at="2026-12-31T00:00:00Z")
        split = self.split_of(rows)
        self.assertEqual(split.relabelled, 1)
        self.assertEqual(next(r.label for r in split.train if r.entity_id == "ship-000"), 0)

    def test_censored_rows_are_ignored_and_counted(self):
        rows = synthetic_rows() + [training_row(entity_id="late", as_of="2026-03-01T00:00:00Z", eligible=False)]
        report = train_model(rows, self.dir)
        self.assertEqual(report["split"]["censored_rows_ignored"], 1)
        self.assertEqual(report["split"]["eligible_rows"], 120)


class ReportAndArtifactTests(TrainCase):
    def test_report_has_every_required_part(self):
        report = train_model(synthetic_rows(), self.dir)
        self.assertEqual(set(report["test_metrics"]),
                         {"logistic_regression", "prevalence_baseline", "threshold_rule_8c"})
        for metrics in report["test_metrics"].values():
            self.assertEqual(set(metrics), {"rows", "positives", "average_precision", "brier"})
        self.assertEqual(set(report["slices"]), {"carrier", "freshness"})
        self.assertIn("split_time", report["split"])
        canonical_json(report)                                          # JSON-safe, no NaN

    def test_artifact_matches_the_report(self):
        report = train_model(synthetic_rows(), self.dir)
        _, version = load_artifact(self.dir)
        self.assertEqual(version, report["model_version"])

    def test_same_rows_in_any_order_give_byte_identical_files(self):
        rows = synthetic_rows()
        first = train_model(rows, self.dir)
        model_bytes = (self.dir / ARTIFACT_FILE).read_bytes()
        shuffled = list(rows)
        random.Random(3).shuffle(shuffled)
        other = Path(self._tmp.name) / "other"
        second = train_model(shuffled, other)
        self.assertEqual(second, first)
        self.assertEqual((other / ARTIFACT_FILE).read_bytes(), model_bytes)

    def test_public_train_is_the_same_function(self):
        self.assertEqual(train(synthetic_rows(), self.dir), train_model(synthetic_rows(), self.dir / "b"))


class BadInputTests(TrainCase):
    def assert_error(self, rows, error, code, artifact_dir=None):
        with self.assertRaises(error) as ctx:
            train_model(rows, artifact_dir if artifact_dir is not None else self.dir)
        self.assertEqual(ctx.exception.code, code)
        return ctx.exception

    def test_not_enough_data_fails_clearly_and_writes_nothing(self):
        self.assert_error([training_row()], InvalidArgumentError, "insufficient_training_data")
        no_positives = [dict(r, label=0, metadata=dict(r["metadata"], incidents=[])) for r in synthetic_rows()]
        self.assert_error(no_positives, InvalidArgumentError, "insufficient_training_data")
        self.assertFalse((self.dir / ARTIFACT_FILE).exists())

    def test_malformed_rows_name_their_position(self):
        rows = synthetic_rows()
        rows[1] = "not a row"
        error = self.assert_error(rows, InvalidRecordError, "invalid_training_row")
        self.assertIn("rows[1]", error.detail)

    def test_non_finite_feature_is_refused(self):
        rows = synthetic_rows()
        rows[1] = dict(rows[1], features=dict(rows[1]["features"], latest_temp_c=float("nan")))
        self.assert_error(rows, InvalidRecordError, "invalid_training_row")

    def test_rows_from_old_feature_definitions_are_refused(self):
        rows = synthetic_rows()
        rows[1] = dict(rows[1], metadata=dict(rows[1]["metadata"], feature_schema="features-v0"))
        self.assert_error(rows, InvalidRecordError, "feature_schema_mismatch")

    def test_bad_arguments(self):
        self.assert_error({"rows": 1}, InvalidArgumentError, "not_iterable")
        self.assert_error(synthetic_rows(), InvalidArgumentError, "not_a_path", artifact_dir=42)


@unittest.skipUnless((DATA_DIR / "MANIFEST.json").exists(), "run tools/generate_dataset.py first")
class SampleDataTests(TrainCase):
    def test_train_on_the_sample(self):
        manifest = read_manifest(DATA_DIR / "MANIFEST.json")
        rows = build_training_rows(read_jsonl(DATA_DIR / "events.jsonl"), read_jsonl(DATA_DIR / "labels.jsonl"),
                                   read_jsonl(DATA_DIR / "decisions.jsonl"), manifest.observation_end)
        report = train(rows, self.dir)
        split = report["split"]
        self.assertEqual(split["eligible_rows"] + split["censored_rows_ignored"], len(rows))
        self.assertEqual(split["train_rows"] + split["test_rows"] + split["purged_rows"]
                         + split["grouped_out_rows"], split["eligible_rows"])
        for metrics in report["test_metrics"].values():
            self.assertTrue(0.0 <= metrics["average_precision"] <= 1.0)
        # No accuracy target: only check the model is evaluated, not how good it is.
        self.assertEqual(report["model_version"], load_artifact(self.dir)[1])


if __name__ == "__main__":
    unittest.main()
