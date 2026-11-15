"""Stateless field preparation and fresh, fold-local scikit-learn transformers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data.audit import MISSING, binary_target, number
from .data.schema import FIELDS, HOSPITALS, PREDICTORS


class Tier(StrEnum):
    EARLY = "early"
    EXTENDED = "extended"


class Cholesterol(StrEnum):
    RECORDED = "as_recorded"
    ZERO_MISSING = "zero_as_missing"


EARLY = ("age", "sex", "cp", "trestbps", "chol", "fbs")
NUMERIC = ("age", "trestbps", "chol", "thalch", "oldpeak", "ca")
CATEGORICAL = ("sex", "cp", "fbs", "restecg", "exang", "slope", "thal")


def numeric_field(tokens: pd.Series[str], name: str) -> pd.Series[float]:
    values = []
    for token in tokens:
        value, error = number(token, FIELDS[name].kind == "integer")
        if error:
            raise ValueError(f"{name}: {error}")
        values.append(float(value) if value is not None else np.nan)
    return pd.Series(values, index=tokens.index, dtype="float64")


@dataclass(frozen=True)
class ResearchData:
    predictors: pd.DataFrame
    metadata: pd.DataFrame
    target: pd.Series[int]
    groups: pd.Series[int]

    def features(self, tier: Tier, cholesterol: Cholesterol) -> pd.DataFrame:
        fields = EARLY if Tier(tier) == Tier.EARLY else PREDICTORS
        result = self.predictors.loc[:, list(fields)].copy()
        if Cholesterol(cholesterol) == Cholesterol.ZERO_MISSING:
            result["chol"] = result["chol"].mask(result["chol"].eq(0))
        return result


def prepare_data(raw: pd.DataFrame) -> ResearchData:
    """Accept the text frame from load_dataset. Split indices refer to CSV row positions."""
    raw = raw.reset_index(drop=True)
    meta = raw.loc[:, ["id", "dataset", "num"]].copy()
    ids = [number(t, integer=True)[0] for t in meta["id"]]
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError("Missing, invalid or repeated ID requires linkage review.")
    if not meta["dataset"].isin(HOSPITALS).all():
        raise ValueError("Unknown or missing hospital.")
    target = binary_target(meta["num"])
    if target.isna().any():
        raise ValueError("Invalid or missing outcome.")
    x = raw.loc[:, list(PREDICTORS)].copy()
    for name in NUMERIC:
        x[name] = numeric_field(x[name], name)
    for name in CATEGORICAL:
        x[name] = x[name].astype(object).where(~x[name].isin(MISSING), np.nan)
    # Group before tier selection or sensitivity recoding. No outcome or site enters the key.
    groups = x.groupby(list(PREDICTORS), dropna=False, sort=False).ngroup()
    return ResearchData(x, meta, target.astype("int64"), groups)


def build_preprocessor(tier: Tier) -> Pipeline:
    """Return an unfitted estimator, also suitable as a step inside future training-only CV."""
    fields = EARLY if Tier(tier) == Tier.EARLY else PREDICTORS
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    category = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="constant", fill_value="__missing__", keep_empty_features=True
                ),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    columns = ColumnTransformer(
        [
            ("numeric", numeric, [f for f in NUMERIC if f in fields]),
            ("category", category, [f for f in CATEGORICAL if f in fields]),
        ],
        remainder="drop",
    )
    return Pipeline([("columns", columns)]).set_output(transform="pandas")
