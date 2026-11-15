"""Aggregate source records without modifying the loaded frame."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from statistics import mean, median

import pandas as pd

from .loading import Dataset
from .schema import COLUMNS, EXPECTED_ROWS, FIELDS, HOSPITALS, RELEASE_SHA256, TARGETS

MISSING = {"", "?"}
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", flags=re.ASCII)


def percent(count: int, total: int) -> float | None:
    return round(100 * count / total, 3) if total else None


def number(token: str, integer: bool) -> tuple[Decimal | None, str | None]:
    if token in MISSING:
        return None, None
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None, "invalid numeric text"
    if not value.is_finite():
        return None, "non-finite number"
    if not NUMBER.fullmatch(token):
        return None, "invalid numeric text"
    if integer and value != value.to_integral_value():
        return None, "expected an integer"
    if not math.isfinite(float(value)) or (value != 0 and float(value) == 0):
        return None, "outside supported numeric range"
    return value, None


def target_class(token: str) -> int | None:
    value, _ = number(token, integer=True)
    return int(value) if value is not None and 0 <= value <= 4 else None


def binary_target(original: pd.Series[str]) -> pd.Series[int]:
    """Return an aligned nullable 0/1 series. Missing and invalid labels stay unknown."""
    labels = [target_class(token) for token in original]
    return pd.Series(
        [None if label is None else int(label > 0) for label in labels],
        index=original.index,
        dtype="Int64",
        name="heart_disease_recorded",
    )


@dataclass
class Missingness:
    count: int
    percent: float | None


def missingness(tokens: pd.Series[str]) -> Missingness:
    count = int(tokens.isin(MISSING).sum())
    return Missingness(count, percent(count, len(tokens)))


@dataclass
class NumericSummary:
    count: int
    minimum: float
    median: float
    mean: float
    maximum: float


@dataclass
class ColumnAudit:
    storage_dtype: str
    expected_type: str
    role: str
    missing: Missingness
    categories: dict[str, int]
    invalid: dict[str, int]
    review: dict[str, int]
    numeric: NumericSummary | None


def inspect_column(name: str, tokens: pd.Series[str]) -> ColumnAudit:
    field = FIELDS[name]
    invalid: Counter[str] = Counter()
    review: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    numbers: list[Decimal] = []
    for token in tokens:
        if token in MISSING:
            continue
        if field.levels:
            categories[token] += 1
            if token not in field.levels:
                invalid["unexpected category"] += 1
            continue
        value, error = number(token, integer=field.kind == "integer")
        if error:
            invalid[error] += 1
        elif value is not None:
            numbers.append(value)
            if name == "num" and not 0 <= value <= 4:
                invalid["target outside 0–4"] += 1
            if name == "ca" and not 0 <= value <= 3:
                invalid["vessel count outside 0–3"] += 1
            if name in {"age", "trestbps", "chol", "thalch", "id"} and value <= 0:
                review["zero" if value == 0 else "negative"] += 1
            if name == "oldpeak" and value < 0:
                review["negative ST depression"] += 1
    numeric = None
    if numbers:
        numeric = NumericSummary(
            len(numbers),
            float(min(numbers)),
            float(median(numbers)),
            round(float(mean(numbers)), 4),
            float(max(numbers)),
        )
    return ColumnAudit(
        str(tokens.dtype),
        field.kind,
        field.role,
        missingness(tokens),
        dict(sorted(categories.items())),
        dict(invalid),
        dict(review),
        numeric,
    )


@dataclass
class HospitalAudit:
    hospital: str | None
    rows: int
    labeled: int
    positive: int
    prevalence_percent: float | None
    age_median: float | None
    missing: dict[str, Missingness]
    review: dict[str, dict[str, int]]


@dataclass
class Audit:
    sha256: str
    matches_kaggle_v6: bool
    rows: int
    columns: int
    fields: dict[str, ColumnAudit]
    original_target: dict[str, int]
    binary_target: dict[str, int]
    unknown_target: int
    invalid_target_tokens: dict[str, int]
    hospitals: list[HospitalAudit]
    duplicate_excess: dict[str, int]
    errors: list[str]


def audit_dataset(dataset: Dataset) -> Audit:
    frame = dataset.frame
    fields = {name: inspect_column(name, frame[name]) for name in COLUMNS}
    errors = []
    if len(frame) != EXPECTED_ROWS:
        errors.append(f"Expected {EXPECTED_ROWS} combined records; found {len(frame)}.")
    absent = sorted(set(HOSPITALS) - set(fields["dataset"].categories))
    if absent:
        errors.append(f"Absent hospital labels: {', '.join(absent)}.")
    for name, field in fields.items():
        for reason, count in field.invalid.items():
            errors.append(f"{name}: {reason} ({count} records).")
        if name in {"id", "dataset", "num"} and field.missing.count:
            errors.append(f"{name}: {field.missing.count} missing required values.")

    counts = Counter(target_class(token) for token in frame["num"])
    invalid_targets = Counter(
        token for token in frame["num"] if token not in MISSING and target_class(token) is None
    )
    original = {key: counts[int(key)] for key in TARGETS}
    for key, count in original.items():
        if not count:
            errors.append(f"Expected original target class {key} is absent.")
    binary = binary_target(frame["num"])
    hospitals = []
    site = frame["dataset"].mask(frame["dataset"].isin(MISSING))
    for label, group in frame.groupby(site, dropna=False, sort=True):
        outcomes = binary.loc[group.index]
        labeled, positive = int(outcomes.count()), int(outcomes.sum())
        site_fields = {name: inspect_column(name, group[name]) for name in COLUMNS}
        age = site_fields["age"].numeric
        hospitals.append(
            HospitalAudit(
                None if pd.isna(label) else str(label),
                len(group),
                labeled,
                positive,
                percent(positive, labeled),
                age.median if age else None,
                {name: field.missing for name, field in site_fields.items()},
                {name: field.review for name, field in site_fields.items() if field.review},
            )
        )
    ids = [value for token in frame["id"] if (value := number(token, True)[0]) is not None]
    duplicates = {
        "all_fields": int(frame.duplicated().sum()),
        "excluding_id": int(frame.drop(columns="id").duplicated().sum()),
        "repeated_valid_identifier": len(ids) - len(set(ids)),
    }
    if duplicates["repeated_valid_identifier"]:
        errors.append("Valid identifiers repeat. Investigate linkage before partitioning.")
    return Audit(
        dataset.sha256,
        dataset.sha256 == RELEASE_SHA256,
        len(frame),
        len(frame.columns),
        fields,
        original,
        {"0": int(binary.eq(0).sum()), "1": int(binary.eq(1).sum())},
        counts[None],
        dict(sorted(invalid_targets.items())),
        hospitals,
        duplicates,
        errors,
    )
