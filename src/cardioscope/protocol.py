"""Run preprocessing on validated splits and emit an aggregate, reproducible fit record."""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline

from .data.loading import load_dataset
from .preprocessing import Cholesterol, ResearchData, Tier, build_preprocessor, prepare_data
from .splits import SEED, Split, conventional_split, hospital_splits


@dataclass
class PreparedSplit:
    transformer: Pipeline
    train: pd.DataFrame
    test: pd.DataFrame
    record: dict[str, object]


def position_digest(positions: tuple[int, ...]) -> str:
    payload = ",".join(map(str, positions)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def preprocess_split(
    data: ResearchData, split: Split, *, tier: Tier, cholesterol: Cholesterol
) -> PreparedSplit:
    split.validate(data)
    x = data.features(tier, cholesterol)
    train, test = x.iloc[list(split.train)], x.iloc[list(split.test)]
    pipeline = build_preprocessor(tier)
    transformed_train = pipeline.fit_transform(train)
    transformed_test = pipeline.transform(test)
    for result in (transformed_train, transformed_test):
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("Non-finite transformed values.")
    columns = pipeline.named_steps["columns"]
    numeric = columns.named_transformers_["numeric"]
    category = columns.named_transformers_["category"]
    encoder = category.named_steps["encode"]
    scaler = numeric.named_steps["scale"]
    record: dict[str, object] = {
        "tier": tier.value,
        "cholesterol": cholesterol.value,
        "source_columns": list(x.columns),
        "output_columns": transformed_train.columns.tolist(),
        "numeric_columns": scaler.feature_names_in_.tolist(),
        "median_imputer_statistics": numeric.named_steps["impute"].statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "categorical_fill": "__missing__",
        "unseen_category": "all-zero block",
        "training_categories": dict(
            zip(
                encoder.feature_names_in_,
                (levels.tolist() for levels in encoder.categories_),
                strict=True,
            )
        ),
    }
    for name, part in (("train", train), ("test", test)):
        record[name] = {
            "rows": len(part),
            "positions_sha256": position_digest(tuple(part.index)),
            "binary_positive": int(data.target.loc[part.index].sum()),
            "missing_after_policy": part.isna().sum().to_dict(),
            "chol_zeros_masked": int(data.predictors.loc[part.index, "chol"].eq(0).sum())
            if cholesterol == Cholesterol.ZERO_MISSING
            else 0,
        }
    return PreparedSplit(pipeline, transformed_train, transformed_test, record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write UTF-8 JSON (default: stdout)")
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.input)
        data = prepare_data(dataset.frame)
        experiments = {}
        for split in (conventional_split(data), *hospital_splits(data)):
            experiments[split.hospital or "pooled"] = {
                "training_exclusions": len(split.excluded),
                "runs": [
                    preprocess_split(data, split, tier=tier, cholesterol=policy).record
                    for tier in Tier
                    for policy in Cholesterol
                ],
            }
        payload = json.dumps(
            {
                "csv_sha256": dataset.sha256,
                "sklearn": sklearn.__version__,
                "pooled_seed": SEED,
                "experiments": experiments,
            },
            allow_nan=False,
            indent=2,
        )
        if args.output is None:
            print(payload)
        else:
            if args.output.exists() and args.output.samefile(args.input):
                raise ValueError("Output would overwrite the source CSV.")
            args.output.write_text(payload + "\n", encoding="utf-8", newline="\n")
    except (ValueError, OSError) as exc:
        print(f"Protocol failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
