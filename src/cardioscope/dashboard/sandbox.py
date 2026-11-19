"""Explicitly build and verify one local Phase 3 demonstration pipeline."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from cardioscope.dashboard.evidence import FAMILIES, ROOT, Evidence, EvidenceError, load_evidence
from cardioscope.data.schema import FIELDS
from cardioscope.preprocessing import EARLY

MODEL_DIR = ROOT / "models"
LIMITS = {"age": (18, 100), "trestbps": (50, 250), "chol": (0, 700)}
INPUT_LABELS = {
    "age": "Age (years)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dL)",
    "sex": "Recorded sex category",
    "cp": "Chest pain category",
    "fbs": "Fasting blood sugar above 120 mg/dL",
}
PROFILE = {
    "age": 48.0,
    "trestbps": 120.0,
    "chol": 210.0,
    "sex": "Female",
    "cp": "atypical angina",
    "fbs": "FALSE",
}


def validate_profile(values: dict[str, Any]) -> pd.DataFrame:
    if set(values) != set(EARLY):
        raise ValueError("Provide the six supported measurement fields.")
    if all(value is None for value in values.values()):
        raise ValueError("Enter at least one example measurement before running a prediction.")
    for key, (low, high) in LIMITS.items():
        v = values[key]
        if v is not None and (type(v) not in (int, float) or not low <= v <= high or v != int(v)):
            raise ValueError(f"{INPUT_LABELS[key]} must be a whole number from {low} to {high}.")
    for key in ("sex", "cp", "fbs"):
        if values[key] is not None and (
            type(values[key]) is not str or values[key] not in FIELDS[key].levels
        ):
            raise ValueError(f"Choose a supported option for {INPUT_LABELS[key]}.")
    # Match the representations used by the frozen numeric and categorical imputers.
    encoded = {}
    for key in EARLY:
        value = values[key]
        encoded[key] = pd.Series(
            [np.nan if value is None else value],
            dtype="float64" if key in LIMITS else "object",
        )
    return pd.DataFrame(encoded)


@dataclass(frozen=True)
class Prediction:
    id: str
    model: str
    feature_tier: str
    version: str
    artifact_digest: str
    score: float
    historical_class: int
    threshold: float


def predict(e: Evidence, model: Any, values: dict[str, Any]) -> Prediction:
    frame = validate_profile(values)
    metadata = expected_metadata(e)
    threshold = float(metadata["threshold"])
    try:
        score = float(model.predict_proba(frame)[0, 1])
    except (ValueError, TypeError, IndexError) as exc:
        raise ValueError("The verified model could not score these measurements.") from exc
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("The verified model did not return a valid score.")
    return Prediction(
        id=uuid4().hex,
        model=FAMILIES[metadata["family"]],
        feature_tier=metadata["tier"],
        version=metadata["versions"]["scikit-learn"],
        artifact_digest=metadata["holdout_probability_sha256"],
        score=score,
        historical_class=int(score >= threshold),
        threshold=threshold,
    )


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
    """Cache only model bytes, never input values or prediction outputs."""
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
            "model build command from the README."
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
    print("Local Prediction Lab pipeline verified against the Phase 3 prediction digest.")
