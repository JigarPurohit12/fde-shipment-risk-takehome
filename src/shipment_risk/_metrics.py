"""Evaluation metrics (scikit-learn implementations).

average_precision  ranking quality; the right metric when positives are rare.
                   A random scorer gets about the positive rate.
brier_score        probability quality: mean of (p - y)^2, lower is better.
                   Always predicting the base rate b scores b * (1 - b).

Thin wrappers so the rest of the code is unchanged: they return plain floats,
and None where a metric is undefined (no positives / no rows) instead of
letting scikit-learn warn or raise.
"""
from __future__ import annotations

from sklearn.metrics import average_precision_score, brier_score_loss


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    """Area under the precision-recall step curve. None if there are no positives.

    Tied scores are taken together, so the order of tied rows never matters.
    """
    if sum(labels) == 0:
        return None
    # Floating-point summation can give 1.0000000000000002; AP is at most 1.
    return min(float(average_precision_score(labels, scores)), 1.0)


def brier_score(labels: list[int], probabilities: list[float]) -> float | None:
    if not labels:
        return None
    return float(brier_score_loss(labels, probabilities, pos_label=1))
