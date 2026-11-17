"""Calibration, diagnostic summaries and paired within-hospital uncertainty."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from .scoring import Labels, Probabilities, check_predictions
from .splits import SEED


def log_odds(p: Probabilities) -> Probabilities:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def inverse_logit(x: Probabilities) -> Probabilities:
    return np.exp(-np.logaddexp(0.0, -x))


@dataclass(frozen=True)
class Sigmoid:
    intercept: float
    slope: float

    def apply(self, p: Probabilities) -> Probabilities:
        return inverse_logit(self.intercept + self.slope * log_odds(p))


def logistic_line(y: Labels, p: Probabilities, c: float) -> Sigmoid | None:
    x = log_odds(p)
    if len(np.unique(y)) != 2 or np.ptp(x) == 0:
        return None
    if np.isinf(c) and (x[y == 1].min() >= x[y == 0].max() or x[y == 0].min() >= x[y == 1].max()):
        return None  # Complete or quasi-complete separation has no finite MLE.
    with warnings.catch_warnings(record=True) as caught:
        model = LogisticRegression(C=c, solver="lbfgs", tol=1e-8, max_iter=2000, random_state=SEED)
        model.fit(x.reshape(-1, 1), y)
    if any(w.category.__name__ == "ConvergenceWarning" for w in caught):
        return None
    return Sigmoid(float(model.intercept_[0]), float(model.coef_[0, 0]))


def fit_sigmoid(y: Labels, oof: Probabilities) -> Sigmoid | None:
    check_predictions(y, oof)
    if min(int(y.sum()), len(y) - int(y.sum())) < 20:
        return None
    fit = logistic_line(y, oof, 1.0)
    return fit if fit and fit.slope > 0 else None


def calibration_diagnostics(y: Labels, p: Probabilities) -> dict[str, Any]:
    check_predictions(y, p)
    x = log_odds(p)
    intercept = None
    if len(np.unique(y)) == 2:
        low, high = -60.0, 60.0
        for _ in range(70):
            mid = (low + high) / 2
            if inverse_logit(x + mid).mean() < y.mean():
                low = mid
            else:
                high = mid
        intercept = (low + high) / 2
    line = logistic_line(y, p, np.inf)
    bins = []
    membership = np.minimum((p * 5).astype(int), 4)
    for i in range(5):
        mask = membership == i
        bins.append(
            {
                "from": i / 5,
                "to": (i + 1) / 5,
                "n": int(mask.sum()),
                "positive": int(y[mask].sum()),
                "mean_p": float(p[mask].mean()) if mask.any() else None,
                "observed": float(y[mask].mean()) if mask.any() else None,
            }
        )
    return {
        "mean_p": float(p.mean()),
        "citl": intercept,
        "slope": line.slope if line else None,
        "sparse_class": min(int(y.sum()), len(y) - int(y.sum())) < 20,
        "bins": bins,
    }


def group_weights(groups: Labels, repetitions: int = 2000) -> Labels:
    """One row of multiplicities per draw. All rows in one group share a multiplicity."""
    unique, inverse = np.unique(groups, return_inverse=True)
    n = len(unique)
    draws = np.random.default_rng(SEED).integers(n, size=(repetitions, n))
    offsets = np.arange(repetitions)[:, None] * n
    counts = np.bincount((draws + offsets).ravel(), minlength=repetitions * n)
    return np.asarray(counts.reshape(repetitions, n)[:, inverse], dtype=np.int64)


def boot_scores(
    y: Labels, p: Probabilities, threshold: float, weights: Labels
) -> dict[str, Probabilities]:
    """Weighted rank statistic counts ties as half-concordant without expanding records."""
    check_predictions(y, p)
    pred = p >= threshold
    pos, neg = weights @ y, weights @ (1 - y)
    order = np.argsort(p, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    positive = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    negative = np.add.reduceat(weights[:, order] * (1 - y[order]), starts, axis=1)
    concordance = (positive * (negative.cumsum(axis=1) - negative / 2)).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = (weights @ (y * pred)) / pos
        specificity = (weights @ ((1 - y) * ~pred)) / neg
        auc = concordance / (pos * neg)
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": auc,
    }


def interval(samples: Probabilities) -> dict[str, Any]:
    defined = samples[np.isfinite(samples)]
    return {
        "valid": len(defined),
        "bounds": np.quantile(defined, [0.025, 0.975]).tolist() if len(defined) else None,
    }
