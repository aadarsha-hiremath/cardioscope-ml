"""Candidate construction and selection on development rows only."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .preprocessing import Tier, build_preprocessor
from .protocol import position_digest
from .scoring import Labels, Probabilities, check_predictions, choose_threshold, metrics
from .splits import SEED

GRID = {"logistic": (0.1, 1.0), "forest": (4, 8), "xgboost": (2, 3)}


def parameters(family: str, value: float) -> dict[str, Any]:
    if family not in GRID or value not in GRID[family]:
        raise ValueError("Configuration outside the frozen search grid.")
    if family == "logistic":
        return dict(
            C=value,
            solver="lbfgs",
            l1_ratio=0.0,
            fit_intercept=True,
            max_iter=2000,
            tol=1e-4,
            class_weight=None,
            random_state=SEED,
        )
    if family == "forest":
        return dict(
            n_estimators=160,
            max_depth=int(value),
            min_samples_leaf=4,
            max_features="sqrt",
            criterion="gini",
            bootstrap=True,
            class_weight=None,
            random_state=SEED,
            n_jobs=1,
        )
    return dict(
        n_estimators=120,
        max_depth=int(value),
        learning_rate=0.05,
        min_child_weight=1,
        reg_lambda=1,
        reg_alpha=0,
        subsample=1,
        colsample_bytree=1,
        scale_pos_weight=1,
        booster="gbtree",
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        random_state=SEED,
        n_jobs=1,
    )


def make_model(tier: Tier, family: str, value: float) -> Pipeline:
    config = parameters(family, value)
    if family == "logistic":
        estimator = LogisticRegression(**config)
    elif family == "forest":
        estimator = RandomForestClassifier(**config)
    else:
        estimator = XGBClassifier(**config)
    return Pipeline([("features", build_preprocessor(tier)), ("model", estimator)])


Fold = tuple[Labels, Labels]


def inner_folds(y: pd.Series[int], sites: pd.Series[str], groups: pd.Series[int]) -> Iterator[Fold]:
    if not y.index.equals(sites.index) or not y.index.equals(groups.index):
        raise ValueError("Development labels, hospitals and groups must align.")
    strata = sites.astype(str) + "/" + y.astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    seen = np.zeros(len(y), dtype=int)
    for fit, val in splitter.split(y, strata, groups):
        if set(groups.iloc[fit]) & set(groups.iloc[val]):
            raise ValueError("Duplicate group crosses a CV fold.")
        if y.iloc[fit].nunique() != 2 or y.iloc[val].nunique() != 2:
            raise ValueError("Both classes required in each CV fold.")
        seen[val] += 1
        yield fit, val
    if not np.all(seen == 1):
        raise ValueError("Incomplete out-of-fold coverage.")


def probability_digest(p: Probabilities) -> str:
    return hashlib.sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest()


def probabilities(model: Pipeline, x: pd.DataFrame) -> Probabilities:
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected class ordering.")
    p = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    check_predictions(np.zeros(len(x), dtype=np.int64), p)
    return p


def fold_records(index: pd.Index[Any], folds: tuple[Fold, ...]) -> list[dict[str, object]]:
    return [
        {
            "train_rows": len(fit),
            "validation_rows": len(val),
            "train_digest": position_digest(tuple(index[fit])),
            "validation_digest": position_digest(tuple(index[val])),
        }
        for fit, val in folds
    ]


def select_model(
    x: pd.DataFrame, y: pd.Series[int], folds: tuple[Fold, ...], tier: Tier, family: str
) -> dict[str, Any]:
    if not x.index.equals(y.index):
        raise ValueError("Training feature and label indices differ.")
    started = time.perf_counter()
    labels = np.asarray(y, dtype=np.int64)
    candidates: list[dict[str, Any]] = []
    predictions = []
    for value in GRID[family]:
        oof = np.full(len(y), np.nan)
        scores = []
        for fit, val in folds:
            model = make_model(tier, family, value)
            model.fit(x.iloc[fit], y.iloc[fit])
            oof[val] = probabilities(model, x.iloc[val])
            scores.append(float(average_precision_score(labels[val], oof[val])))
        check_predictions(labels, oof)
        candidates.append(
            {
                "value": value,
                "parameters": parameters(family, value),
                "fold_ap": scores,
                "mean_ap": float(np.mean(scores)),
                "sd_ap": float(np.std(scores, ddof=1)),
            }
        )
        predictions.append(oof)
    best = max(range(len(candidates)), key=lambda i: candidates[i]["mean_ap"])
    chosen = candidates[best]
    p = predictions[best]
    cut = choose_threshold(labels, p)
    return {
        "family": family,
        "tier": tier.value,
        "candidates": candidates,
        "selected": chosen,
        "threshold": cut,
        "oof_probability_sha256": probability_digest(p),
        "oof_metrics": metrics(labels, p, cut),
        "cv_seconds": time.perf_counter() - started,
    }
