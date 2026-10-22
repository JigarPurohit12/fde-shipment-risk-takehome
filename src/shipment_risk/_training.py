"""train(): purged temporal split -> logistic regression -> honest evaluation.

The split (README "Report a temporal split ... Account for overlap between
six-hour label windows across split boundaries"):
  1. Keep eligible rows (step 4), sorted by time. Censored rows are counted.
  2. split_time = as_of of the row at the 70 % point.
     test  = rows with as_of >= split_time
  3. train = rows that were MATURE at split_time (as_of + 6h + 30h <= split_time),
     re-labelled with only the labels available by split_time.
     Rows in the 36 h gap before split_time are purged: their label windows or
     reporting delays reach past split_time, into the test period.
  4. No shipment is on both sides (shipment-grouped).
This replaces customer note 2 (random row split, same shipment in both sets).

Evaluation on the test rows, against two baselines:
  prevalence_baseline   always the training positive rate ("no skill")
  threshold_rule_8c     latest reading >= 8 C, the usual cold-chain alarm,
                        turned into the training positive rate of each side
Metrics: average precision and Brier score, plus two slices (carrier; fresh
vs stale latest reading). With only ~20 test positives the numbers are noisy;
DECISIONS.md says so instead of computing confidence intervals.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from ._canonical import sha256_digest
from ._features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, feature_digest
from ._labels import is_mature
from ._metrics import average_precision, brier_score
from ._model import LogisticModel, fit_logistic, save_artifact
from ._time import format_timestamp, parse_timestamp
from .errors import InvalidArgumentError, InvalidRecordError, ShipmentRiskError

TEST_FRACTION = 0.3
RULE_THRESHOLD_C = 8.0          # upper bound of the usual 2-8 C cold-chain range
FRESH_HOURS = 1.0               # slice: latest reading at most 1 h old
METRIC_DECIMALS = 6             # reported numbers; also hides last-bit float noise


@dataclass(frozen=True)
class Row:
    entity_id: str
    as_of: datetime
    features: dict
    label: int
    label_times: tuple          # label_available_at of each incident in the window


@dataclass(frozen=True)
class Split:
    split_time: datetime
    train: list
    test: list
    purged: int
    grouped_out: int
    relabelled: int


def train_model(rows, artifact_dir) -> dict:
    directory = _artifact_directory(artifact_dir)
    eligible, ignored = _eligible_rows(rows)
    split = _temporal_split(eligible)
    _require_enough(split)

    model = fit_logistic([r.features for r in split.train], [r.label for r in split.train])
    report = {
        "split": _split_report(split, eligible, ignored),
        **_evaluate(model, split),        # test_metrics, slices
        "model": {"type": "logistic_regression", "features": list(FEATURE_NAMES),
                  "carriers_seen": list(model.carriers)},
    }

    training_info = {
        "split_time": format_timestamp(split.split_time),
        "train_rows": len(split.train),
        "train_positives": sum(r.label for r in split.train),
        "rows_digest": sha256_digest([
            [r.entity_id, format_timestamp(r.as_of), r.label, feature_digest(r.features)]
            for r in split.train
        ]),
    }
    report["model_version"] = save_artifact(model, directory, training_info)
    return report


# --------------------------------------------------------------------------
# Input rows
# --------------------------------------------------------------------------

def _artifact_directory(value) -> Path:
    if not isinstance(value, (str, Path)):
        raise InvalidArgumentError("not_a_path", f"artifact_dir must be a path, got {type(value).__name__}")
    directory = Path(value)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _eligible_rows(rows) -> tuple[list[Row], int]:
    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Iterable):
        raise InvalidArgumentError("not_iterable", "rows must be a list of training rows")
    eligible, ignored = [], 0
    for index, row in enumerate(rows):
        if isinstance(row, Mapping) and isinstance(row.get("metadata"), Mapping) \
                and row["metadata"].get("eligible") is not True:
            ignored += 1                                   # censored rows never train or test
            continue
        eligible.append(_checked_row(row, f"rows[{index}]"))
    eligible.sort(key=lambda r: (r.as_of, r.entity_id))
    return eligible, ignored


def _checked_row(row, where: str) -> Row:
    """Rows come from build_training_rows; check only what training relies on."""
    try:
        metadata, features, label = row["metadata"], row["features"], row["label"]
        if metadata["feature_schema"] != FEATURE_SCHEMA_VERSION:
            raise InvalidRecordError("feature_schema_mismatch",
                                     f"built with {metadata['feature_schema']!r}, code uses {FEATURE_SCHEMA_VERSION!r}")
        if set(features) != set(FEATURE_NAMES) or type(label) is not int or label not in (0, 1):
            raise ValueError("wrong feature names, or label is not 0/1")
        if any(isinstance(v, float) and not math.isfinite(v) for v in features.values()):
            raise ValueError("a feature is NaN or infinite")
        label_times = tuple(parse_timestamp(i["label_available_at"], "label_available_at")
                            for i in metadata["incidents"])
        return Row(row["entity_id"], parse_timestamp(row["as_of"], "as_of"), dict(features), label, label_times)
    except ShipmentRiskError as exc:                       # keep its code, add the position
        raise type(exc)(exc.code, f"{where}: {exc.detail}") from None
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidRecordError("invalid_training_row", f"{where} is not a valid training row: {exc}") from None


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

def _temporal_split(rows: list[Row]) -> Split:
    if len(rows) < 2:
        raise InvalidArgumentError("insufficient_training_data", f"only {len(rows)} eligible rows")
    split_time = rows[int(len(rows) * (1 - TEST_FRACTION))].as_of
    test = [r for r in rows if r.as_of >= split_time]
    test_shipments = {r.entity_id for r in test}
    train, purged, grouped_out, relabelled = [], 0, 0, 0
    for r in rows:
        if r.as_of >= split_time:
            continue
        if not is_mature(r.as_of, split_time):
            purged += 1
            continue
        if r.entity_id in test_shipments:
            grouped_out += 1
            continue
        # What did we know about this row's outcome AT split_time?
        label = 1 if any(t <= split_time for t in r.label_times) else 0
        relabelled += label != r.label
        train.append(replace(r, label=label))
    return Split(split_time, train, test, purged, grouped_out, relabelled)


def _require_enough(split: Split) -> None:
    positives = sum(r.label for r in split.train)
    if not split.test or positives == 0 or positives == len(split.train):
        raise InvalidArgumentError(
            "insufficient_training_data",
            f"need positives and negatives to train and rows to test; got {len(split.train)} train rows "
            f"({positives} positive) and {len(split.test)} test rows",
        )


def _split_report(split: Split, eligible: list[Row], ignored: int) -> dict:
    return {
        "method": "temporal; train rows mature at split_time (as_of + 36h <= split_time); "
                  "36 h gap purged; no shipment on both sides",
        "split_time": format_timestamp(split.split_time),
        "test_fraction": TEST_FRACTION,
        "eligible_rows": len(eligible),
        "censored_rows_ignored": ignored,
        "train_rows": len(split.train),
        "train_positives": sum(r.label for r in split.train),
        "test_rows": len(split.test),
        "test_positives": sum(r.label for r in split.test),
        "purged_rows": split.purged,
        "grouped_out_rows": split.grouped_out,
        "train_labels_unknown_at_split": split.relabelled,
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _evaluate(model: LogisticModel, split: Split) -> dict:
    base_rate = sum(r.label for r in split.train) / len(split.train)
    hot_rate, cold_rate = _rule_rates(split.train, base_rate)

    # One record per test row: label plus the probability from each scorer.
    scored = [
        {
            "label": r.label,
            "carrier": r.features["carrier"] or "unknown",
            "fresh": _is_fresh(r),
            "logistic_regression": model.predict(r.features),
            "prevalence_baseline": base_rate,
            "threshold_rule_8c": hot_rate if _is_hot(r) else cold_rate,
        }
        for r in split.test
    ]
    scorers = ("logistic_regression", "prevalence_baseline", "threshold_rule_8c")
    test_metrics = {name: _metrics(scored, name) for name in scorers}

    slices = {"carrier": {}, "freshness": {}}
    for carrier in sorted({s["carrier"] for s in scored}):
        slices["carrier"][carrier] = _metrics([s for s in scored if s["carrier"] == carrier],
                                              "logistic_regression")
    for key, fresh in (("fresh_le_1h", True), ("stale_gt_1h", False)):
        slices["freshness"][key] = _metrics([s for s in scored if s["fresh"] == fresh],
                                            "logistic_regression")
    return {"test_metrics": test_metrics, "slices": slices}


def _rule_rates(train: list[Row], base_rate: float) -> tuple[float, float]:
    hot = [r.label for r in train if _is_hot(r)]
    cold = [r.label for r in train if not _is_hot(r)]
    hot_rate = sum(hot) / len(hot) if hot else base_rate
    cold_rate = sum(cold) / len(cold) if cold else base_rate
    return hot_rate, cold_rate


def _is_fresh(row: Row) -> bool:
    age = row.features["hours_since_latest_reading"]
    return age is not None and age <= FRESH_HOURS


def _is_hot(row: Row) -> bool:
    latest = row.features["latest_temp_c"]
    return latest is not None and latest >= RULE_THRESHOLD_C


def _ap(rows, name):
    return average_precision([s["label"] for s in rows], [s[name] for s in rows])


def _brier(rows, name):
    return brier_score([s["label"] for s in rows], [s[name] for s in rows])


def _metrics(rows, name) -> dict:
    return {
        "rows": len(rows),
        "positives": sum(s["label"] for s in rows),
        "average_precision": _round(_ap(rows, name)),   # None when a slice has no positives
        "brier": _round(_brier(rows, name)),
    }


def _round(value):
    return None if value is None else round(value, METRIC_DECIMALS) + 0.0
