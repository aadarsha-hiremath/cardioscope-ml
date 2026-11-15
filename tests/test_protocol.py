"""Invented rows exercise fold boundaries. No UCI or Kaggle data are used."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cardioscope.data.schema import COLUMNS, HOSPITALS, PREDICTORS
from cardioscope.preprocessing import EARLY, Cholesterol, Tier, prepare_data
from cardioscope.protocol import preprocess_split
from cardioscope.splits import Split, conventional_split, hospital_splits


@pytest.fixture
def raw() -> pd.DataFrame:
    rows = []
    for site in HOSPITALS:
        for i in range(12):
            k = len(rows)
            rows.append(
                [
                    str(10000 + k),
                    str(30 + i),
                    "Male",
                    site,
                    "asymptomatic",
                    str(95 + k),
                    str(160 + k),
                    "TRUE",
                    "normal",
                    "130",
                    "FALSE",
                    "-0.5",
                    "flat",
                    "0",
                    "normal",
                    str(i % 2 * 4),
                ]
            )
    return pd.DataFrame(rows, columns=pd.Index(COLUMNS), dtype="string")


def test_feature_roles_and_sensitivity(raw: pd.DataFrame) -> None:
    raw.loc[0, ["chol", "trestbps"]] = "0"
    before = raw.copy()
    data = prepare_data(raw)
    pd.testing.assert_frame_equal(data.metadata, raw[["id", "dataset", "num"]])
    assert data.target.tolist()[:2] == [0, 1]
    assert data.metadata["num"].tolist()[:2] == ["0", "4"]
    early = data.features(Tier.EARLY, Cholesterol.RECORDED)
    extended = data.features(Tier.EXTENDED, Cholesterol.ZERO_MISSING)
    assert tuple(early.columns) == EARLY and tuple(extended.columns) == PREDICTORS
    assert not {"id", "dataset", "num", "heart_disease_recorded"}.intersection(extended)
    assert early.loc[0, "chol"] == 0 and pd.isna(extended.loc[0, "chol"])
    assert extended.loc[0, ["trestbps", "oldpeak", "ca"]].tolist() == [0, -0.5, 0]
    pd.testing.assert_frame_equal(raw, before)


def test_fit_is_blind_to_evaluation_features(raw: pd.DataFrame) -> None:
    raw.loc[1, "chol"] = "?"
    raw.loc[2, "cp"] = ""
    data = prepare_data(raw)
    split = tuple(hospital_splits(data))[-1]
    options = dict(tier=Tier.EXTENDED, cholesterol=Cholesterol.RECORDED)
    first = preprocess_split(data, split, **options)
    raw.loc[list(split.test), "chol"] = "1500"
    raw.loc[list(split.test), "cp"] = "previously unseen"
    other = preprocess_split(prepare_data(raw), split, **options)
    pd.testing.assert_frame_equal(first.train, other.train)
    assert first.record == other.record  # Same missing counts, fitted statistics and categories.
    assert not first.test.equals(other.test)
    assert first.train.index.tolist() == list(split.train)
    assert other.test.index.tolist() == list(split.test)
    chol = data.predictors["chol"].iloc[list(split.train)]
    num = first.transformer.named_steps["columns"].named_transformers_["numeric"]
    idx = list(num.feature_names_in_).index("chol")
    assert num.named_steps["impute"].statistics_[idx] == chol.median()
    assert num.named_steps["scale"].mean_[idx] == pytest.approx(chol.fillna(chol.median()).mean())
    assert first.train["numeric__chol"].mean() == pytest.approx(0)
    assert first.train.loc[2, "category__cp___missing__"] == 1
    assert (other.test.filter(like="category__cp_") == 0).all().all()
    assert np.isfinite(other.test.to_numpy()).all()


def test_empty_training_fields_stay_transformable(raw: pd.DataFrame) -> None:
    raw.loc[raw.dataset.ne("Cleveland"), ["thal", "ca"]] = "?"
    data = prepare_data(raw)
    split = next(hospital_splits(data))
    result = preprocess_split(data, split, tier=Tier.EXTENDED, cholesterol=Cholesterol.ZERO_MISSING)
    assert "numeric__ca" in result.train
    assert list(result.train.columns) == list(result.test.columns)
    assert (result.test["category__thal___missing__"] == 0).all()
    assert np.isfinite(result.test.to_numpy()).all()


def test_deterministic_grouped_split_and_repeated_protocol(raw: pd.DataFrame) -> None:
    raw.loc[1, list(PREDICTORS)] = raw.loc[0, list(PREDICTORS)]
    data = prepare_data(raw)
    assert data.groups[0] == data.groups[1]  # Different original targets do not break the group.
    split = conventional_split(data, folds=3)
    assert split == conventional_split(data, folds=3)
    assert set(split.train).isdisjoint(split.test)
    assert (0 in split.test) == (1 in split.test)
    assert set(split.train + split.test) == set(range(len(raw)))
    assert set(data.target.iloc[list(split.test)]) == {0, 1}
    first, second = [
        preprocess_split(data, split, tier=Tier.EARLY, cholesterol=Cholesterol.RECORDED)
        for _ in range(2)
    ]
    assert json.dumps(first.record, allow_nan=False) == json.dumps(second.record, allow_nan=False)
    pd.testing.assert_frame_equal(first.test, second.test)


def test_cross_site_copy_is_removed_from_training_only(raw: pd.DataFrame) -> None:
    raw.loc[12, list(PREDICTORS)] = raw.loc[0, list(PREDICTORS)]
    raw.loc[12, "age"] = "30.0"  # Semantically identical numeric encoding.
    data = prepare_data(raw)
    seen = []
    for split in hospital_splits(data):
        held = split.hospital
        assert set(data.metadata.iloc[list(split.test)].dataset) == {held}
        assert held not in set(data.metadata.iloc[list(split.train)].dataset)
        assert set(split.test) == set(raw.index[raw.dataset.eq(held)])
        expected = {"Cleveland": (12,), "Hungary": (0,)}.get(held, ())
        assert split.excluded == expected
        seen.extend(split.test)
        if expected:
            with pytest.raises(ValueError, match="group crosses"):
                replace(split, train=split.train + expected, excluded=()).validate(data)
    assert sorted(seen) == list(range(len(raw)))


def test_similar_profiles_do_not_imply_linkage(raw: pd.DataFrame) -> None:
    raw.loc[1, list(PREDICTORS)] = raw.loc[0, list(PREDICTORS)]
    raw.loc[1, "restecg"] = "lv hypertrophy"
    data = prepare_data(raw)
    assert data.groups[0] != data.groups[1]


@pytest.mark.parametrize(
    ("name", "token", "message"),
    [
        ("id", "10001", "ID"),
        ("num", "?", "outcome"),
        ("num", "8", "outcome"),
        ("dataset", "other", "hospital"),
        ("chol", "NaN", "non-finite"),
    ],
)
def test_bad_inputs_fail_before_splitting(
    raw: pd.DataFrame, name: str, token: str, message: str
) -> None:
    raw.loc[0, name] = token
    with pytest.raises(ValueError, match=message):
        prepare_data(raw)


def test_malformed_splits_are_rejected_before_fitting(raw: pd.DataFrame) -> None:
    data = prepare_data(raw)
    with pytest.raises(ValueError, match="overlap"):
        Split(tuple(range(47)), (0,)).validate(data)
    with pytest.raises(ValueError, match="complete held-out"):
        replace(conventional_split(data, folds=3), hospital="Cleveland").validate(data)
    with pytest.raises(ValueError, match="Insufficient"):
        conventional_split(data, folds=7)
    with pytest.raises(ValueError, match="Four"):
        list(hospital_splits(prepare_data(raw[raw.dataset.ne("Switzerland")])))
