#!/usr/bin/env python3
"""Reproduce the shuffled-label check behind the AP 1.0 result (RESULTS.md).

    python tools/shuffled_label_check.py

Train on randomly reassigned TRAINING labels (same number of incidents, random
rows), evaluate on the TRUE test labels. Repeated over several seeds, because
with about 25 training incidents a single shuffle is noisy.

What it shows: with no real signal in the labels, the score falls to chance,
so the evaluation itself is sound. (It does not by itself prove the features
are leak-free; receipt-time filtering and the step 3 parity check do that.)
Nothing is written to disk; data/ is only read.
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works without `pip install -e .`

from shipment_risk._io import read_jsonl, read_manifest  # noqa: E402
from shipment_risk._metrics import average_precision  # noqa: E402
from shipment_risk._model import fit_logistic  # noqa: E402
from shipment_risk._training import _eligible_rows, _temporal_split  # noqa: E402
from shipment_risk.solution import build_training_rows  # noqa: E402

SEEDS = range(20)


def main() -> None:
    # The same rows and the same purged temporal split that train() uses.
    data = ROOT / "data"
    manifest = read_manifest(data / "MANIFEST.json")
    rows = build_training_rows(
        read_jsonl(data / "events.jsonl"),
        read_jsonl(data / "labels.jsonl"),
        read_jsonl(data / "decisions.jsonl"),
        manifest.observation_end,
    )
    eligible, _ = _eligible_rows(rows)
    split = _temporal_split(eligible)

    features = [r.features for r in split.train]
    labels = [r.label for r in split.train]
    test_labels = [r.label for r in split.test]                  # never shuffled

    def test_ap(train_labels: list[int]) -> float:
        model = fit_logistic(features, train_labels)
        return average_precision(test_labels, [model.predict(r.features) for r in split.test])

    results = []
    for seed in SEEDS:
        shuffled = labels[:]
        random.Random(seed).shuffle(shuffled)                    # same incidents, random rows
        results.append(test_ap(shuffled))

    print("Shuffled training labels (sample data)")
    print(f"  real labels:        AP {test_ap(labels):.3f}")
    print(f"  shuffled, {len(results)} seeds: median AP {statistics.median(results):.3f} "
          f"(min {min(results):.3f}, max {max(results):.3f})")
    print(f"  chance level:       AP {sum(test_labels) / len(test_labels):.3f} (test positive rate)")


if __name__ == "__main__":
    main()
