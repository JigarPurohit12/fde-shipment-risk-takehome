"""Logistic regression as a scikit-learn Pipeline, saved with pickle.

Pipeline (fitted on TRAINING rows only):
    numeric features : SimpleImputer(mean)  -> StandardScaler
    carrier          : OneHotEncoder(handle_unknown="ignore")
    classifier       : LogisticRegression(C=1.0)
  * a missing number becomes the training mean, so it contributes 0 after scaling
  * a feature that never varies gets scale 1 (no division by zero)
  * a carrier not seen in training (or None) gets all-zero columns: no adjustment
  * objective: C * sum(log-loss) + 1/2 * |weights|^2 (intercept not penalized),
    a mild L2 penalty that keeps weights finite on nearly separable data

Artifact = two files in artifact_dir:
    model.pkl   the pickled Pipeline
    model.json  metadata: format, feature schema, scikit-learn version,
                SHA-256 of model.pkl, training info, model_version

Pickle risks and how they are handled:
  * Unpickling can execute code. load_artifact checks the SHA-256 of model.pkl
    against model.json BEFORE unpickling, so a swapped or corrupted file is
    refused. That is an integrity check, not authentication: artifacts must
    come from our own training job and a write-protected location.
  * A pickle only loads reliably with the scikit-learn version that wrote it.
    The version is recorded and checked; upgrading the library means retraining.
  * Pickle BYTES are not stable (a load/save round trip changes them), so
    model_version is a hash of the fitted PARAMETERS, rounded, not of the file.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ._canonical import canonical_json, sha256_digest
from ._features import FEATURE_SCHEMA_VERSION
from ._io import parse_json_object, write_atomic
from .errors import ArtifactError, InvalidRecordError

NUMERIC_FEATURES = (
    "hours_since_latest_reading",
    "latest_temp_c",
    "max_temp_c",
    "mean_temp_c",
    "reading_count",
    "trend_c_per_hour",
)
MODEL_FORMAT = "shipment-risk-sklearn-pipeline"
MODEL_FORMAT_VERSION = 2
ARTIFACT_FILE = "model.json"      # metadata, read and checked first
PICKLE_FILE = "model.pkl"         # the Pipeline itself
L2_PENALTY = 1.0                  # sklearn C = 1 / L2_PENALTY
SOLVER_TOLERANCE = 1e-10
MAX_SOLVER_ITERATIONS = 10_000
WEIGHT_DECIMALS = 6               # coefficients, so scores do not depend on solver/BLAS noise


@dataclass(frozen=True)
class LogisticModel:
    pipeline: Pipeline
    carriers: tuple               # carriers seen in training, sorted (for reports)

    def predict(self, features) -> float:
        """Probability of an excursion in the next 6 hours, always in [0, 1]."""
        return float(self.pipeline.predict_proba(_matrix([features]))[0, 1])


def _matrix(feature_sets) -> np.ndarray:
    """Feature dicts -> one row per dict: 6 numbers (NaN if missing), then the carrier."""
    rows = [[np.nan if f.get(name) is None else float(f[name]) for name in NUMERIC_FEATURES]
            + [f.get("carrier")] for f in feature_sets]
    return np.array(rows, dtype=object)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def fit_logistic(feature_sets: list[dict], labels: list[int]) -> LogisticModel:
    numeric = Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())])
    encode = ColumnTransformer([
        ("numeric", numeric, list(range(len(NUMERIC_FEATURES)))),
        ("carrier", OneHotEncoder(handle_unknown="ignore"), [len(NUMERIC_FEATURES)]),
    ])
    classifier = LogisticRegression(C=1.0 / L2_PENALTY, tol=SOLVER_TOLERANCE, max_iter=MAX_SOLVER_ITERATIONS)
    pipeline = Pipeline([("encode", encode), ("classifier", classifier)])
    pipeline.fit(_matrix(feature_sets), labels)
    if classifier.n_iter_[0] >= MAX_SOLVER_ITERATIONS:
        raise ArithmeticError("logistic regression did not converge")
    # Round, so every machine scores with exactly the numbers model_version describes.
    classifier.coef_ = np.round(classifier.coef_, WEIGHT_DECIMALS) + 0.0
    classifier.intercept_ = np.round(classifier.intercept_, WEIGHT_DECIMALS) + 0.0
    carriers = tuple(str(c) for c in encode.named_transformers_["carrier"].categories_[0])
    return LogisticModel(pipeline, carriers)


def _parameters(model: LogisticModel) -> dict:
    """The fitted numbers that define the model (rounded): what model_version hashes."""
    numeric = model.pipeline.named_steps["encode"].named_transformers_["numeric"]
    classifier = model.pipeline.named_steps["classifier"]
    r = lambda values: [round(float(v), 10) + 0.0 for v in values]
    return {
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "numeric_features": list(NUMERIC_FEATURES),
        "fill_values": r(numeric.named_steps["impute"].statistics_),
        "scaler_means": r(numeric.named_steps["scale"].mean_),
        "scaler_scales": r(numeric.named_steps["scale"].scale_),
        "carriers": list(model.carriers),
        "coefficients": r(classifier.coef_[0]),
        "intercept": r(classifier.intercept_)[0],
    }


# --------------------------------------------------------------------------
# Artifact
# --------------------------------------------------------------------------

def save_artifact(model: LogisticModel, artifact_dir: Path, training: dict) -> str:
    """Write model.pkl, then model.json (atomically); return model_version."""
    artifact_dir = Path(artifact_dir)
    payload = pickle.dumps(model.pipeline, protocol=pickle.HIGHEST_PROTOCOL)
    version = "logreg-" + sha256_digest({**_parameters(model), "training": training})[len("sha256:"):][:16]
    metadata = {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "sklearn_version": sklearn.__version__,
        "pickle_sha256": hashlib.sha256(payload).hexdigest(),
        "parameters": _parameters(model),          # human-readable copy of the weights
        "training": training,
        "model_version": version,
    }
    write_atomic(artifact_dir / PICKLE_FILE, payload)
    write_atomic(artifact_dir / ARTIFACT_FILE, canonical_json(metadata) + b"\n")
    return version


def load_artifact(artifact_dir: Path) -> tuple[LogisticModel, str]:
    """Check model.json, verify model.pkl's hash, THEN unpickle. Raises ArtifactError."""
    meta_path, pickle_path = Path(artifact_dir) / ARTIFACT_FILE, Path(artifact_dir) / PICKLE_FILE
    if not meta_path.is_file():
        raise ArtifactError("artifact_missing", f"{meta_path} does not exist")
    try:
        metadata = parse_json_object(meta_path.read_text(encoding="utf-8"), meta_path.name)
    except (InvalidRecordError, UnicodeDecodeError) as exc:
        raise ArtifactError("artifact_corrupt", f"{meta_path.name} is not valid JSON: {exc}") from None
    if metadata.get("format") != MODEL_FORMAT or metadata.get("format_version") != MODEL_FORMAT_VERSION:
        raise ArtifactError("artifact_format", f"{meta_path.name}: unsupported model format")
    if metadata.get("feature_schema") != FEATURE_SCHEMA_VERSION:
        raise ArtifactError("feature_schema_mismatch",
                            f"artifact built for {metadata.get('feature_schema')!r}, code computes {FEATURE_SCHEMA_VERSION!r}")
    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ArtifactError("artifact_format",
                            f"pickled with scikit-learn {metadata.get('sklearn_version')}, installed {sklearn.__version__}")
    if not pickle_path.is_file():
        raise ArtifactError("artifact_missing", f"{pickle_path} does not exist")
    payload = pickle_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata.get("pickle_sha256"):   # checked BEFORE unpickling
        raise ArtifactError("artifact_corrupt", f"{pickle_path.name} does not match its recorded SHA-256")
    try:
        pipeline = pickle.loads(payload)
    except Exception as exc:                                                 # broken pickle of any kind
        raise ArtifactError("artifact_corrupt", f"{pickle_path.name} cannot be unpickled: {exc}") from None
    if not isinstance(pipeline, Pipeline):
        raise ArtifactError("artifact_format", f"{pickle_path.name} does not contain a scikit-learn Pipeline")
    carriers = tuple(str(c) for c in pipeline.named_steps["encode"].named_transformers_["carrier"].categories_[0])
    return LogisticModel(pipeline, carriers), metadata["model_version"]
