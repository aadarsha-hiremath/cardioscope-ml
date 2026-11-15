"""Binary endpoint metrics and duplicate-group uncertainty."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .splits import SEED

Labels = NDArray[np.int64]
Probabilities = NDArray[np.float64]


def check_predictions(y: Labels, p: Probabilities) -> None:
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isin(y, (0, 1)).all():
        raise ValueError("Expected aligned, nonempty binary labels and probabilities.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid positive-class probability.")


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics(y: Labels, p: Probabilities, threshold: float) -> dict[str, float | int | None]:
    check_predictions(y, p)
    positive = p >= threshold
    tp, fp = int(positive[y == 1].sum()), int(positive[y == 0].sum())
    fn, tn = int((y == 1).sum()) - tp, int((y == 0).sum()) - fp
    sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None,
        "precision": ratio(tp, tp + fp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if tp + fn else None,
        "brier": float(brier_score_loss(y, p, pos_label=1)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def choose_threshold(y: Labels, p: Probabilities) -> float:
    check_predictions(y, p)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both training classes.")
    # Integer numerator is proportional to balanced accuracy on these fixed OOF labels.
    pos, neg = p[y == 1], p[y == 0]

    def rank(cut: int) -> tuple[int, int, int]:
        score = int((pos >= cut / 100).sum()) * len(neg) + int((neg < cut / 100).sum()) * len(pos)
        return score, -abs(cut - 50), -cut

    return max(range(20, 81, 5), key=rank) / 100


def balanced_interval(
    y: Labels, p: Probabilities, threshold: float, groups: Labels
) -> dict[str, object]:
    check_predictions(y, p)
    if groups.shape != y.shape:
        raise ValueError("Bootstrap groups do not align with evaluation rows.")
    unique, index = np.unique(groups, return_inverse=True)
    pred = p >= threshold
    counts = np.column_stack(
        [
            np.bincount(index, weights=mask, minlength=len(unique))
            for mask in ((y == 1) & pred, y == 1, (y == 0) & ~pred, y == 0)
        ]
    )
    draws = np.random.default_rng(SEED).integers(len(unique), size=(2000, len(unique)))
    tp, pos, tn, neg = counts[draws].sum(axis=1).T
    valid = (pos > 0) & (neg > 0)
    values = (tp[valid] / pos[valid] + tn[valid] / neg[valid]) / 2
    return {
        "groups": len(unique),
        "draws": 2000,
        "valid": int(valid.sum()),
        "bounds": np.quantile(values, [0.025, 0.975]).tolist() if len(values) else None,
    }
