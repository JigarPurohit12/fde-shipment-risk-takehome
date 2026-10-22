"""Logistic regression Pipeline and its pickle + metadata artifact."""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from shipment_risk._canonical import canonical_json
from shipment_risk._model import ARTIFACT_FILE, PICKLE_FILE, WEIGHT_DECIMALS, fit_logistic, load_artifact, save_artifact
from shipment_risk.errors import ArtifactError

SRC = Path(__file__).resolve().parents[1] / "src"
TEMPS = [3.5, 4.0, 4.2, 4.6, 5.0, 6.5, 7.5, 8.5, 4.1, 3.9]
LABELS = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
CARRIERS = ["coast", "north"] * 5


def features(latest, carrier="coast", **overrides):
    values = {"carrier": carrier, "hours_since_latest_reading": 0.5, "latest_temp_c": latest,
              "max_temp_c": latest, "mean_temp_c": latest - 0.5, "reading_count": 5,
              "trend_c_per_hour": (latest - 4.0) / 4.0}
    values.update(overrides)
    return values


def small_model():
    return fit_logistic([features(t, c) for t, c in zip(TEMPS, CARRIERS)], LABELS)


class FitTests(unittest.TestCase):
    def test_warmer_means_riskier_and_probabilities_are_valid(self):
        model = small_model()
        probs = [model.predict(features(t)) for t in (3.0, 5.0, 7.0, 9.0)]
        self.assertEqual(probs, sorted(probs))
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_missing_value_counts_as_the_training_mean(self):
        model = small_model()
        imputer = model.pipeline.named_steps["encode"].named_transformers_["numeric"].named_steps["impute"]
        mean_latest = imputer.statistics_[1]                       # latest_temp_c is column 1
        self.assertEqual(model.predict(features(5.0, latest_temp_c=None)),
                         model.predict(features(5.0, latest_temp_c=mean_latest)))

    def test_constant_feature_does_not_divide_by_zero(self):
        scaler = small_model().pipeline.named_steps["encode"].named_transformers_["numeric"].named_steps["scale"]
        self.assertEqual(scaler.scale_[4], 1.0)                    # reading_count is always 5

    def test_unseen_or_missing_carrier_gets_no_carrier_adjustment(self):
        model = small_model()
        self.assertEqual(model.carriers, ("coast", "north"))
        self.assertEqual(model.predict(features(5.0, carrier="arctic-express")),
                         model.predict(features(5.0, carrier=None)))

    def test_extreme_inputs_never_overflow(self):
        model = small_model()
        for latest in (-1e300, -273.0, 1e6, 1e300):
            p = model.predict(features(latest))
            self.assertTrue(math.isfinite(p) and 0.0 <= p <= 1.0, latest)

    def test_weights_are_rounded_for_cross_platform_reproducibility(self):
        classifier = small_model().pipeline.named_steps["classifier"]
        for w in list(classifier.coef_[0]) + list(classifier.intercept_):
            self.assertEqual(w, round(float(w), WEIGHT_DECIMALS))

    def test_fit_minimizes_logloss_plus_half_squared_weights(self):
        # At the optimum of  sum(log-loss) + 1/2 * |w|^2  (intercept not
        # penalized) the gradient is zero: pins down which objective is solved.
        model = small_model()
        rows = [features(t, c) for t, c in zip(TEMPS, CARRIERS)]
        from shipment_risk._model import _matrix
        encoded = model.pipeline.named_steps["encode"].transform(_matrix(rows))
        weights = model.pipeline.named_steps["classifier"].coef_[0]
        residuals = [model.predict(f) - y for f, y in zip(rows, LABELS)]
        self.assertAlmostEqual(sum(residuals), 0.0, places=3)
        for j, w in enumerate(weights):
            self.assertAlmostEqual(sum(r * x[j] for r, x in zip(residuals, encoded)) + w, 0.0, places=3)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.model = small_model()
        self.version = save_artifact(self.model, self.dir, {"note": "test"})
        self.meta, self.pickle = self.dir / ARTIFACT_FILE, self.dir / PICKLE_FILE

    def rewrite_metadata(self, **changes):
        metadata = json.loads(self.meta.read_text())
        metadata.update(changes)
        self.meta.write_bytes(canonical_json(metadata))

    def assert_code(self, code):
        with self.assertRaises(ArtifactError) as ctx:
            load_artifact(self.dir)
        self.assertEqual(ctx.exception.code, code)

    def test_round_trip_gives_identical_predictions_and_version(self):
        loaded, version = load_artifact(self.dir)
        self.assertEqual(version, self.version)
        for t in (3.0, 6.0, 9.0):
            self.assertEqual(loaded.predict(features(t)), self.model.predict(features(t)))

    def test_version_comes_from_parameters_not_pickle_bytes(self):
        loaded, _ = load_artifact(self.dir)                # a load/save round trip changes the bytes
        other = self.dir / "again"
        other.mkdir()
        self.assertEqual(save_artifact(loaded, other, {"note": "test"}), self.version)

    def test_metadata_is_readable_json_with_the_weights(self):
        metadata = json.loads(self.meta.read_text())
        self.assertEqual(metadata["model_version"], self.version)
        self.assertEqual(len(metadata["parameters"]["coefficients"]), 6 + 2)    # 6 numeric + 2 carriers

    def test_missing_files(self):
        self.pickle.unlink()
        self.assert_code("artifact_missing")
        self.meta.unlink()
        self.assert_code("artifact_missing")

    def test_garbage_metadata(self):
        self.meta.write_text("not json")
        self.assert_code("artifact_corrupt")

    def test_tampered_pickle_is_refused_before_unpickling(self):
        self.pickle.write_bytes(self.pickle.read_bytes() + b"x")
        self.assert_code("artifact_corrupt")

    def test_old_feature_schema_is_refused(self):
        self.rewrite_metadata(feature_schema="features-v0")
        self.assert_code("feature_schema_mismatch")

    def test_other_scikit_learn_version_is_refused(self):
        self.rewrite_metadata(sklearn_version="0.0.1")
        self.assert_code("artifact_format")

    def test_fresh_process_loads_the_same_model(self):
        expected = repr(self.model.predict(features(6.0)))
        code = (
            "import sys; from pathlib import Path\n"
            "from shipment_risk._model import load_artifact\n"
            "model, version = load_artifact(Path(sys.argv[1]))\n"
            "f = {'carrier': 'coast', 'hours_since_latest_reading': 0.5, 'latest_temp_c': 6.0,\n"
            "     'max_temp_c': 6.0, 'mean_temp_c': 5.5, 'reading_count': 5, 'trend_c_per_hour': 0.5}\n"
            "print(version, repr(model.predict(f)))\n"
        )
        env = dict(os.environ, PYTHONPATH=str(SRC))
        out = subprocess.run([sys.executable, "-c", code, str(self.dir)], env=env,
                             capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, [self.version, expected])


if __name__ == "__main__":
    unittest.main()
