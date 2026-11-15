"""Reproducible row-position splits with explicit duplicate and hospital checks."""

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data.schema import HOSPITALS
from .preprocessing import ResearchData

SEED = 2026


@dataclass(frozen=True)
class Split:
    train: tuple[int, ...]
    test: tuple[int, ...]
    excluded: tuple[int, ...] = ()
    hospital: str | None = None

    def validate(self, data: ResearchData) -> None:
        train, test, excluded = set(self.train), set(self.test), set(self.excluded)
        n = len(data.target)
        if not train or not test or len(self.train) + len(self.test) + len(self.excluded) != n:
            raise ValueError("Expected nonempty train/test and an exhaustive partition.")
        if train | test | excluded != set(range(n)):
            raise ValueError("Indices overlap, repeat, are missing or out of range.")
        test_groups = set(data.groups.iloc[pd.Index(self.test)])
        if test_groups.intersection(data.groups.iloc[pd.Index(self.train)]):
            raise ValueError("Exact predictor group crosses the training boundary.")
        if self.hospital is None:
            if excluded:
                raise ValueError("Pooled evaluation requires complete group assignment.")
        else:
            site = data.metadata["dataset"]
            if self.hospital not in HOSPITALS or test != set(site.index[site.eq(self.hospital)]):
                raise ValueError("Test must contain the complete held-out hospital.")
            overlap = data.groups.isin(test_groups) & site.ne(self.hospital)
            if excluded != set(site.index[overlap]):
                raise ValueError("Only cross-site copies may be excluded from training.")


def conventional_split(data: ResearchData, *, seed: int = SEED, folds: int = 5) -> Split:
    strata = data.metadata["dataset"].astype(str) + "/" + data.target.astype(str)
    support = data.groups.groupby(strata).nunique()
    if folds < 2 or support.min() < folds:
        raise ValueError("Insufficient independent groups per observed site/outcome stratum.")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    train, test = next(splitter.split(data.predictors, strata, data.groups))
    split = Split(tuple(map(int, train)), tuple(map(int, test)))
    split.validate(data)
    return split


def hospital_splits(data: ResearchData) -> Iterator[Split]:
    site = data.metadata["dataset"]
    if set(site) != set(HOSPITALS):
        raise ValueError("Four source hospitals are required for leave-one-hospital-out.")
    for hospital in HOSPITALS:
        held = site.eq(hospital)
        overlap = data.groups.isin(data.groups[held]) & ~held
        split = Split(
            tuple(site.index[~(held | overlap)]),
            tuple(site.index[held]),
            tuple(site.index[overlap]),
            hospital,
        )
        split.validate(data)
        yield split
