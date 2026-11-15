"""Synthetic experiments for the paired pooled comparison."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cardioscope import experiment, training
from cardioscope.data.schema import HOSPITALS, PREDICTORS
from cardioscope.preprocessing import EARLY, Cholesterol, Tier, prepare_data
from cardioscope.scoring import balanced_interval, choose_threshold, metrics
from cardioscope.splits import conventional_split


@pytest.fixture
def source() -> pd.DataFrame:
    n = 100
    base = dict(
        id="0",
        age="50",
        sex="Male",
        cp="asymptomatic",
        dataset="Cleveland",
        trestbps="110",
        chol="210",
        fbs="FALSE",
        restecg="normal",
        thalch="145",
        exang="FALSE",
        oldpeak="-0.5",
        slope="flat",
        ca="",
        thal="normal",
        num="0",
    )
    rows = [
        base
        | {
            "id": str(i + 1),
            "dataset": HOSPITALS[i % 4],
            "age": str(35 + i % 35),
            "trestbps": str(90 + i),
            "chol": "" if i % 5 == 0 else str(120 + i),
            "num": str(3 * (i // 4 % 2)),
        }
        for i in range(n)
    ]
    # A second ID with the exact same recorded predictors.
    rows[-1] = rows[3] | {"id": str(n)}
    return pd.DataFrame(rows, dtype="string")


@pytest.mark.parametrize("family", training.GRID)
def test_estimators_train_on_allowed_features_and_repeat(source: pd.DataFrame, family: str) -> None:
    data = prepare_data(source)
    x = data.features(Tier.EARLY, Cholesterol.RECORDED)
    p = []
    for _ in range(2):
        model = training.make_model(Tier.EARLY, family, training.GRID[family][0])
        model.fit(x.iloc[:80], data.target.iloc[:80])
        assert tuple(model.feature_names_in_) == EARLY
        assert set(PREDICTORS).isdisjoint({"dataset", "id", "num", "heart_disease_recorded"})
        encoder = model.named_steps["features"].named_steps["columns"]
        assert all(n.split("__")[1].split("_")[0] in EARLY for n in encoder.get_feature_names_out())
        p.append(training.probabilities(model, x.iloc[80:]))
    np.testing.assert_array_equal(p[0], p[1])
    assert np.all((0 <= p[0]) & (p[0] <= 1))


def test_cv_fit_indices_statistics_and_threshold_isolation(
    source: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = prepare_data(source)
    split = conventional_split(data)
    train = pd.Index(split.train)
    y = data.target.iloc[train]
    folds = tuple(
        training.inner_folds(y, data.metadata["dataset"].iloc[train], data.groups.iloc[train])
    )
    x = data.features(Tier.EARLY, Cholesterol.RECORDED).iloc[train]
    observed = []
    original = training.make_model

    def traced(*args):
        pipeline = original(*args)
        fit = pipeline.fit

        def trace_fit(part, labels):
            fit(part, labels)
            numeric = (
                pipeline.named_steps["features"]
                .named_steps["columns"]
                .named_transformers_["numeric"]
            )
            j = list(numeric.feature_names_in_).index("chol")
            assert numeric.named_steps["impute"].statistics_[j] == part["chol"].median()
            observed.append(tuple(part.index))
            return pipeline

        pipeline.fit = trace_fit
        return pipeline

    captured = []

    def cutoff(labels, p):
        np.testing.assert_array_equal(labels, y.to_numpy())
        captured.append(training.probability_digest(p))
        return choose_threshold(labels, p)

    monkeypatch.setattr(training, "make_model", traced)
    monkeypatch.setattr(training, "choose_threshold", cutoff)
    result = training.select_model(x, y, folds, Tier.EARLY, "logistic")
    assert observed == [tuple(train[a]) for a, _ in folds] * 2
    all_val = []
    for a, b in folds:
        assert not set(train[a]) & set(split.test)
        assert not set(train[b]) & set(split.test)
        assert not set(data.groups.iloc[train[a]]) & set(data.groups.iloc[train[b]])
        all_val.extend(train[b])
    assert sorted(all_val) == sorted(split.train)
    assert captured == [result["oof_probability_sha256"]]


def test_known_scores_and_class_order() -> None:
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.7, 0.6, 0.9])
    m = metrics(y, p, 0.6)
    assert [m[k] for k in ("tn", "fp", "fn", "tp")] == [2, 1, 0, 2]
    assert m["accuracy"] == 0.8 and m["f1"] == 0.8 and m["sensitivity"] == 1
    for key in ("average_precision", "roc_auc", "balanced_accuracy"):
        assert m[key] == pytest.approx(5 / 6)
    assert m["precision"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(2 / 3)
    assert m["brier"] == pytest.approx(0.142)
    assert m["log_loss"] == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.3, 0.6, 0.9])))
    interval = balanced_interval(y, p, 0.6, np.array([1, 1, 2, 3, 4]))
    assert interval["groups"] == 4 and interval["draws"] == 2000


def test_undefined_metrics_and_cutoff_ties() -> None:
    m = metrics(np.array([0, 0]), np.array([0.1, 0.2]), 0.5)
    assert all(
        m[k] is None
        for k in (
            "roc_auc",
            "balanced_accuracy",
            "average_precision",
            "sensitivity",
            "precision",
            "f1",
        )
    )
    assert m["specificity"] == 1
    assert choose_threshold(np.array([0, 1]), np.array([0.1, 0.9])) == 0.5
    with pytest.raises(ValueError, match="probability"):
        metrics(np.array([0, 1]), np.array([np.nan, 0.6]), 0.5)


def test_aggregate_experiment_record_and_failure(
    source: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = prepare_data(source)
    split = conventional_split(data)
    dev = experiment.development(data, split, [Tier.EARLY])
    lock = {"metadata": experiment.metadata(data, split, "synthetic"), **dev}
    result = experiment.evaluate(data, split, lock)
    assert len(result) == 3 and len(lock["folds"]) == 5
    assert lock["metadata"]["dataset_sha256"] == "synthetic"
    assert {
        "seed",
        "versions",
        "train_positions_sha256",
        "test_positions_sha256",
        "preprocessing",
        "selection",
        "threshold_rule",
    } <= lock["metadata"].keys()
    for r, selected in zip(result, dev["runs"], strict=True):
        assert r["threshold"] == selected["threshold"]
        assert r["parameters"] == selected["selected"]["parameters"]
        assert len(r["probability_sha256"]) == 64 and r["refit_seconds"] > 0
        assert len(selected["selected"]["fold_ap"]) == 5

    def fail(*args):
        raise ValueError("synthetic fit error")

    monkeypatch.setattr(training, "make_model", fail)
    with pytest.raises(RuntimeError, match="early/logistic CV failed: synthetic fit error"):
        experiment.development(data, split, [Tier.EARLY])
    assert (
        experiment.main(
            [
                "--stage",
                "cv",
                "--input",
                str(tmp_path / "absent.csv"),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    failure = json.loads((tmp_path / "failure-cv-both.json").read_bytes())
    assert failure["status"] == "failed" and failure["error_type"] == "FileNotFoundError"
    assert not (tmp_path / "cv-both.json").exists()
