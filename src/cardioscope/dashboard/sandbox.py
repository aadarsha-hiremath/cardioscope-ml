"""Explicitly build and verify one local Phase 3 demonstration pipeline."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from cardioscope.dashboard.evidence import ROOT, Evidence, EvidenceError, load_evidence
from cardioscope.data.schema import FIELDS

MODEL_DIR = ROOT / "models"
LIMITS = {"age": (18, 100), "trestbps": (50, 250), "chol": (0, 700)}
PROFILE = {
    "age": 48.0,
    "trestbps": 120.0,
    "chol": 210.0,
    "sex": "Female",
    "cp": "atypical angina",
    "fbs": "FALSE",
}


def validate_profile(values: dict[str, Any]) -> pd.DataFrame:
    if set(values) != set(PROFILE):
        raise ValueError("The early model requires exactly six feature values.")
    for key, (low, high) in LIMITS.items():
        v = values[key]
        if type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    for key in ("sex", "cp", "fbs"):
        if values[key] not in FIELDS[key].levels:
            raise ValueError(f"Unsupported category for {key}.")
    return pd.DataFrame([values]).loc[:, ["age", "sex", "cp", "trestbps", "chol", "fbs"]]


def expected_metadata(e: Evidence) -> dict[str, Any]:
    r = next(r for r in e.pooled["evaluation"] if (r["tier"], r["family"]) == ("early", "logistic"))
    return {
        "dataset_sha256": e.audit["sha256"],
        "selection_sha256": e.pooled["selection_sha256"],
        "family": "logistic",
        "tier": "early",
        "parameters": r["parameters"],
        "threshold": r["threshold"],
        "holdout_probability_sha256": r["probability_sha256"],
        "versions": e.pooled["selection"]["metadata"]["versions"],
    }


@lru_cache(maxsize=1)
def _deserialize_verified(raw: bytes) -> Any:
    """Cache only model bytes, never fictional inputs or prediction outputs."""
    import joblib

    return joblib.load(io.BytesIO(raw))


def load_verified(e: Evidence, directory: Path = MODEL_DIR) -> Any:
    """Accept only the known local artifact; never accept uploaded pickle paths."""
    from importlib.metadata import version

    try:
        saved = json.loads((directory / "sandbox.json").read_bytes())
        for key, value in expected_metadata(e).items():
            if saved[key] != value:
                raise ValueError("Incompatible metadata")
        for name in ("scikit-learn", "numpy", "pandas"):
            if version(name) != saved["versions"][name]:
                raise ValueError("Incompatible runtime")
        raw = (directory / "sandbox.joblib").read_bytes()
        if hashlib.sha256(raw).hexdigest() != saved["model_sha256"]:
            raise ValueError("Model fingerprint mismatch")
        model = _deserialize_verified(raw)
        if list(model.feature_names_in_) != list(validate_profile(PROFILE).columns) or list(
            model.classes_
        ) != [0, 1]:
            raise ValueError("Incompatible fitted pipeline")
        return model
    except (
        OSError,
        KeyError,
        ValueError,
        TypeError,
        AttributeError,
        EOFError,
        ImportError,
        pickle.UnpicklingError,
    ) as exc:
        raise EvidenceError(
            "Verified local demonstration model unavailable. Run the explicit "
            "sandbox build command from the README."
        ) from exc


def build(directory: Path = MODEL_DIR) -> None:
    import joblib
    from threadpoolctl import threadpool_limits

    from cardioscope.data.loading import load_dataset
    from cardioscope.preprocessing import Cholesterol, Tier, prepare_data
    from cardioscope.splits import conventional_split
    from cardioscope.training import make_model, probabilities, probability_digest

    e = load_evidence()
    dataset = load_dataset(ROOT / "data/raw/heart_disease_uci.csv")
    if dataset.sha256 != e.audit["sha256"]:
        raise EvidenceError("Unexpected training CSV.")
    data = prepare_data(dataset.frame)
    split = conventional_split(data)
    x = data.features(Tier.EARLY, Cholesterol.RECORDED)
    meta = expected_metadata(e)
    with threadpool_limits(limits=1):
        model = make_model(Tier.EARLY, "logistic", meta["parameters"]["C"])
        model.fit(x.iloc[list(split.train)], data.target.iloc[pd.Index(split.train)])
        if (
            probability_digest(probabilities(model, x.iloc[list(split.test)]))
            != meta["holdout_probability_sha256"]
        ):
            raise EvidenceError("Refitted model failed to reproduce the Phase 3 prediction digest.")
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "sandbox.joblib")
    meta["model_sha256"] = hashlib.sha256((directory / "sandbox.joblib").read_bytes()).hexdigest()
    (directory / "sandbox.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()
    print("Local sandbox pipeline verified against the Phase 3 prediction digest.")
