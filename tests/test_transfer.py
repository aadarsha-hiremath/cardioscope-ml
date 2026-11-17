"""Isolate training sites, uncertainty and report contracts with invented data."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from cardioscope import transfer
from cardioscope.data.schema import HOSPITALS, PREDICTORS
from cardioscope.preprocessing import Cholesterol, Tier, prepare_data
from cardioscope.protocol import position_digest
from cardioscope.reliability import (
    boot_scores,
    calibration_diagnostics,
    fit_sigmoid,
    group_weights,
    interval,
)
from cardioscope.scoring import metrics
from cardioscope.splits import hospital_splits


def test_frozen_manifest_reproduces_phase3_configurations(tmp_path: Path) -> None:
    path = Path("docs/phase4-protocol.json")
    manifest = transfer.read_manifest(path)
    pooled = json.loads(Path("reports/phase3/cv-both.json").read_bytes())["selection"]
    assert manifest["models"] == [
        {
            "tier": r["tier"],
            "family": r["family"],
            "value": r["selected"]["value"],
            "parameters": r["selected"]["parameters"],
        }
        for r in pooled["runs"]
    ]
    assert manifest["tiers"] == pooled["metadata"]["feature_columns"]
    assert set(manifest["exclusions"]).isdisjoint(manifest["tiers"]["extended"])
    manifest["seed"] = 0
    altered = tmp_path / "manifest.json"
    altered.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="freeze"):
        transfer.read_manifest(altered)


def test_site_means_respect_metric_specific_support() -> None:
    def site(n, score, sensitivity):
        return {
            "test": n,
            "runs": [
                {
                    "tier": "early",
                    "family": "logistic",
                    "policy": "as_recorded",
                    "mode": "raw",
                    "metrics": {"brier": score, "sensitivity": sensitivity},
                }
            ],
        }

    summary = transfer.aggregate([site(10, 0.1, None), site(90, 0.5, 0.8)])[0]["metrics"]
    assert summary["brier"]["macro"] == pytest.approx(0.3)
    assert summary["brier"]["patient_weighted"] == pytest.approx(0.46)
    assert summary["sensitivity"] == {"sites": 1, "macro": 0.8, "patient_weighted": 0.8}


@pytest.fixture
def raw() -> pd.DataFrame:
    base = dict(
        age="48",
        sex="Male",
        cp="atypical angina",
        trestbps="125",
        chol="200",
        fbs="FALSE",
        restecg="normal",
        thalch="160",
        exang="FALSE",
        oldpeak="-0.2",
        slope="flat",
        ca="",
        thal="normal",
    )
    rows = []
    for j, site in enumerate(HOSPITALS):
        for i in range(24):
            rows.append(
                base
                | {
                    "id": str(len(rows) + 1),
                    "dataset": site,
                    "num": str(i % 2),
                    "age": str(40 + i % 2 * 5 + i % 7),
                    "trestbps": str(90 + len(rows)),
                    "chol": "0" if j == 2 else str(180 + i),
                }
            )
    data = pd.DataFrame(rows, dtype="string")
    # Cross-site matches for both an outer and an inner boundary.
    for dest, src in [(24, 0), (72, 49)]:
        data.loc[dest, list(PREDICTORS)] = data.loc[src, list(PREDICTORS)]
    return data


def test_training_and_calibration_boundaries(
    raw: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = prepare_data(raw)
    split = next(hospital_splits(data))
    assert split.excluded == (24,) and split.test == tuple(range(24))
    tr, te = pd.Index(split.train), pd.Index(split.test)
    fits = []
    original = transfer.make_model

    def tracked(*args):
        pipeline = original(*args)
        fit = pipeline.fit

        def record(x, y):
            assert x.index.equals(y.index)
            assert set(x.index).isdisjoint(te) and 24 not in x.index
            fits.append(tuple(x.index))
            return fit(x, y)

        pipeline.fit = record
        return pipeline

    monkeypatch.setattr(transfer, "make_model", tracked)
    seen = []

    def spy(fn):
        def tracked_call(y, p):
            np.testing.assert_array_equal(y, data.target.loc[tr])
            assert len(p) == len(tr)
            seen.append(transfer.probability_digest(p))
            return fn(y, p)

        return tracked_call

    for name in ("choose_threshold", "fit_sigmoid"):
        monkeypatch.setattr(transfer, name, spy(getattr(transfer, name)))
    x = data.features(Tier.EARLY, Cholesterol.RECORDED)

    def run(t):
        return transfer.transfer_predictions(
            x.loc[tr],
            data.target.loc[tr],
            data.metadata.loc[tr, "dataset"],
            data.groups.loc[tr],
            t,
            Tier.EARLY,
            "logistic",
            1.0,
        )

    result = run(x.loc[te])
    for fold, actual in zip(result.folds, fits[:3], strict=True):
        valid = tr[data.metadata.loc[tr, "dataset"].eq(fold["validation_site"])]
        assert set(actual).isdisjoint(valid)
        assert set(data.groups.loc[list(actual)]).isdisjoint(data.groups.loc[valid])
        assert fold["train_digest"] == position_digest(actual)
        assert fold["validation_digest"] == position_digest(tuple(valid))
    assert sorted(f["purged"] for f in result.folds) == [0, 1, 1]
    assert fits[3] == tuple(tr)
    assert seen == [result.oof_digest] * 2
    changed = x.loc[te].copy()
    changed["chol"] = 600
    alternative = run(changed)
    assert result.threshold == alternative.threshold and result.sigmoid == alternative.sigmoid
    assert result.oof_digest == alternative.oof_digest


def test_weighted_auc_ties_and_duplicate_bootstrap() -> None:
    y = np.array([1, 0, 1, 0, 1])
    p = np.array([0.8, 0.8, 0.1, 0.2, 0.8])
    groups = np.array([1, 2, 3, 4, 1])
    w = group_weights(groups, 100)
    np.testing.assert_array_equal(w, group_weights(groups, 100))
    np.testing.assert_array_equal(w[:, 0], w[:, 4])
    scores = boot_scores(y, p, 0.5, w)
    for i, row in enumerate(w):
        labels = np.repeat(y, row)
        values = np.repeat(p, row)
        expect = metrics(labels, values, 0.5)
        for key, v in scores.items():
            if expect[key] is None:
                assert np.isnan(v[i])
            else:
                assert v[i] == pytest.approx(expect[key])
        if np.unique(labels).size == 2:
            assert scores["roc_auc"][i] == pytest.approx(roc_auc_score(labels, values))
    assert interval(np.array([np.nan])) == {"valid": 0, "bounds": None}


def test_calibration_diagnostics_and_insufficient_support() -> None:
    p = np.repeat([0.25, 0.75], 40)
    y = np.r_[np.zeros(30), np.ones(10), np.zeros(10), np.ones(30)].astype(np.int64)
    diag = calibration_diagnostics(y, p)
    assert diag["citl"] == pytest.approx(0, abs=1e-8)
    assert diag["slope"] == pytest.approx(1, abs=1e-5)
    assert [b["n"] for b in diag["bins"]] == [0, 40, 0, 40, 0]
    assert diag["bins"][0]["observed"] is None
    sigmoid = fit_sigmoid(y, p)
    assert sigmoid and sigmoid.slope > 0
    assert fit_sigmoid(y[:32], p[:32]) is None
    assert fit_sigmoid(y, np.ones(len(y)) * 0.4) is None
    assert calibration_diagnostics(1 - y, 1 - y)["slope"] is None
    assert calibration_diagnostics(np.zeros(5, dtype=int), np.linspace(0, 1, 5))["citl"] is None


def test_study_metadata_replay_and_both_policies(raw: pd.DataFrame) -> None:
    manifest = transfer.read_manifest(Path("docs/phase4-protocol.json"))
    data = prepare_data(raw)
    split = list(hospital_splits(data))[2]
    manifest["sites"][split.hospital] = {
        "test": 24,
        "train": len(split.train),
        "positive": 12,
        "negative": 12,
        "training_exclusions": len(split.excluded),
    }
    manifest["uncertainty"]["repetitions"] = 25
    result = transfer.study_site(data, split, manifest)
    # Same numbers of outcomes; reverse all outer labels. No decision about fitting may change.
    changed = data.target.copy()
    changed.loc[list(split.test)] = 1 - changed.loc[list(split.test)]
    repeat = transfer.study_site(replace(data, target=changed), split, manifest)
    assert result["train_sha256"] == position_digest(split.train)
    assert result["test_sha256"] == position_digest(split.test)
    assert len(result["runs"]) == 24
    for r, other in zip(result["runs"], repeat["runs"], strict=True):
        assert all(
            r[k] == other[k]
            for k in ("threshold", "sigmoid_fit", "oof_sha256", "probability_sha256")
        )
        m = r["metrics"]
        assert m["tn"] + m["fp"] == m["tp"] + m["fn"] == 12
        assert m["ap_over_prevalence"] == pytest.approx(m["average_precision"] / 0.5)
        assert set(r["intervals"]) == {"roc_auc", "balanced_accuracy", "sensitivity", "specificity"}
        assert sum(b["n"] for b in r["diagnostics"]["bins"]) == 24
    assert len(result["cholesterol_sensitivity"]) == 12
    assert set(result["training_sites"]) == set(HOSPITALS) - {split.hospital}
    assert len(transfer.aggregate([result])) == 24
    assert {c["value"] for c in manifest["models"]} == {0.1, 1, 2, 4}
