"""Verify frozen report bytes and semantic boundaries before displaying evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cardioscope.data.schema import HOSPITALS, RELEASE_SHA256

ROOT = Path(__file__).resolve().parents[3]
REPORTS = frozenset(
    {
        "reports/dataset-audit.json",
        "reports/phase3/pooled-both.json",
        "reports/phase4/hospital-results.json",
        "docs/phase4-protocol.json",
        "reports/phase3/cv-both.json",
        "reports/preprocessing-protocol.json",
    }
)
FAMILIES = {"logistic": "Logistic Regression", "forest": "Random Forest", "xgboost": "XGBoost"}
METRICS = {
    "roc_auc": "ROC-AUC",
    "average_precision": "Average precision",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "f1": "F1",
    "brier": "Brier score",
    "log_loss": "Log loss",
}
Record = dict[str, Any]


class EvidenceError(ValueError):
    """Research artifact is missing or incompatible with the dashboard."""


@dataclass(frozen=True)
class Evidence:
    audit: Record
    pooled: Record
    transfer: Record
    protocol: Record
    cv: Record
    preprocessing: Record


def metric_check(metrics: Record) -> None:
    for key in METRICS:
        value = metrics[key]
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Invalid metric value.")
        if value < 0 or (key != "log_loss" and value > 1):
            raise ValueError("Metric outside its supported range.")
    for key in ("tn", "fp", "fn", "tp"):
        if type(metrics[key]) is not int or metrics[key] < 0:
            raise ValueError("Invalid confusion matrix.")


def validate(e: Evidence) -> None:
    try:
        if e.transfer["schema_version"] != 1 or e.transfer["protocol"] != e.protocol:
            raise ValueError("Protocol or schema differs.")
        if {
            e.audit["sha256"],
            e.pooled["selection"]["metadata"]["dataset_sha256"],
            e.transfer["protocol"]["dataset_sha256"],
            e.protocol["dataset_sha256"],
            e.preprocessing["csv_sha256"],
        } != {RELEASE_SHA256}:
            raise ValueError("Dataset fingerprints differ.")
        if e.audit["rows"] != sum(s["test"] for s in e.transfer["sites"]):
            raise ValueError("Record counts differ.")
        if set(s["hospital"] for s in e.transfer["sites"]) != set(HOSPITALS):
            raise ValueError("Hospital coverage differs.")
        if e.pooled["selection"] != e.cv["selection"]:
            raise ValueError("Pooled evaluation and CV selections differ.")
        expected = {
            (t, m, p, c)
            for t in ("early", "extended")
            for m in FAMILIES
            for p in ("as_recorded", "zero_as_missing")
            for c in ("raw", "sigmoid")
        }
        for site in e.transfer["sites"]:
            keys = [(r["tier"], r["family"], r["policy"], r["mode"]) for r in site["runs"]]
            if len(keys) != len(set(keys)) or not set(keys) <= expected:
                raise ValueError("Duplicate or unexpected experiments.")
            unavailable = site["sigmoid_unavailable"]
            missing = expected - set(keys)
            if missing != {(r["tier"], r["family"], r["policy"], "sigmoid") for r in unavailable}:
                raise ValueError("Missing result availability explanation.")
            for r in site["runs"]:
                metric_check(r["metrics"])
                m = r["metrics"]
                if m["tp"] + m["fn"] != site["positive"] or m["tn"] + m["fp"] != site["negative"]:
                    raise ValueError("Outcome counts differ.")
                for name in ("balanced_accuracy", "sensitivity", "specificity", "roc_auc"):
                    bounds = r["intervals"][name]["bounds"]
                    if bounds is not None and not (
                        len(bounds) == 2 and 0 <= bounds[0] <= bounds[1] <= 1
                    ):
                        raise ValueError("Invalid interval.")
                if sum(b["n"] for b in r["diagnostics"]["bins"]) != site["test"]:
                    raise ValueError("Calibration bins differ from site population.")
        for r in e.pooled["evaluation"]:
            metric_check(r["holdout_metrics"])
        pairs = {(t, m) for t in ("early", "extended") for m in FAMILIES}
        for rows in (e.pooled["evaluation"], e.cv["selection"]["runs"]):
            if len(rows) != len(pairs) or {(r["tier"], r["family"]) for r in rows} != pairs:
                raise ValueError("Model comparison coverage differs.")
        for r in e.cv["selection"]["runs"]:
            metric_check(r["oof_metrics"])
            if not 0 <= r["selected"]["mean_ap"] <= 1:
                raise ValueError("Invalid CV average precision.")
        for experiment in e.preprocessing["experiments"].values():
            for r in experiment["runs"]:
                if r["source_columns"] != e.protocol["tiers"][r["tier"]]:
                    raise ValueError("Phase 2 feature contract differs.")
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError("Report schema or research consistency check failed.") from exc


def load_evidence(root: Path = ROOT) -> Evidence:
    try:
        expected = json.loads((root / "docs/dashboard-artifacts.json").read_bytes())
        if not isinstance(expected, dict) or set(expected) != REPORTS:
            raise EvidenceError("Report manifest schema is invalid.")
        loaded = {}
        for name, checksum in expected.items():
            raw = (root / name).read_bytes().replace(b"\r\n", b"\n")
            if hashlib.sha256(raw).hexdigest() != checksum:
                raise EvidenceError(f"Artifact fingerprint mismatch: {name}")
            loaded[name] = json.loads(raw)
        evidence = Evidence(
            loaded["reports/dataset-audit.json"],
            loaded["reports/phase3/pooled-both.json"],
            loaded["reports/phase4/hospital-results.json"],
            loaded["docs/phase4-protocol.json"],
            loaded["reports/phase3/cv-both.json"],
            loaded["reports/preprocessing-protocol.json"],
        )
        validate(evidence)
        return evidence
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise EvidenceError(
            "Required research reports are missing or unreadable. Restore the "
            "tracked reports and docs/dashboard-artifacts.json."
        ) from exc


def site_run(
    site: Record, family: str, tier: str, policy: str = "as_recorded", mode: str = "raw"
) -> Record | None:
    return next(
        (
            r
            for r in site["runs"]
            if (r["family"], r["tier"], r["policy"], r["mode"]) == (family, tier, policy, mode)
        ),
        None,
    )
