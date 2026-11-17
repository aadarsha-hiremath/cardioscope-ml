"""Frozen outer-site experiment with training-site cross-fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from .data.loading import load_dataset
from .experiment import atomic_text, digest, rounded
from .preprocessing import Cholesterol, ResearchData, Tier, prepare_data
from .protocol import position_digest
from .reliability import (
    Sigmoid,
    boot_scores,
    calibration_diagnostics,
    fit_sigmoid,
    group_weights,
    interval,
)
from .scoring import Probabilities, choose_threshold, metrics
from .splits import Split, hospital_splits
from .training import make_model, parameters, probabilities, probability_digest

FREEZE = "7103b157534040b6e29a55d8fa00660eff6c8813"
MANIFEST_HASH = "d306a129d657f38d22ffa573bddc0042144c255ccf46b142c571985474b69892"
Row = dict[str, Any]


def read_manifest(path: Path) -> Row:
    # Normalize checkout line endings when identifying the frozen UTF-8 document.
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != MANIFEST_HASH:
        raise ValueError("Protocol differs from the committed freeze.")
    return dict(json.loads(raw))


@dataclass
class FittedTransfer:
    probability: Probabilities
    threshold: float
    sigmoid: Sigmoid | None
    oof_digest: str
    folds: list[Row]


def transfer_predictions(
    train: pd.DataFrame,
    y: pd.Series[int],
    sites: pd.Series[str],
    groups: pd.Series[int],
    test: pd.DataFrame,
    tier: Tier,
    family: str,
    value: float,
) -> FittedTransfer:
    if not all(train.index.equals(s.index) for s in (y, sites, groups)):
        raise ValueError("Training views must share row indices.")
    if (
        not train.index.is_unique
        or train.index.intersection(test.index).size
        or sites.nunique() != 3
    ):
        raise ValueError("Expected disjoint positions and three training hospitals.")
    oof = pd.Series(np.nan, index=train.index, dtype="float64")
    folds = []
    for site in sorted(sites.unique()):
        valid = sites.eq(site)
        duplicate = groups.isin(groups[valid]) & ~valid
        fit = ~valid & ~duplicate
        if y[fit].nunique() != 2:
            raise ValueError(f"Only one training class in inner holdout {site}.")
        model = make_model(tier, family, value)
        model.fit(train.loc[fit], y[fit])
        oof.loc[valid] = probabilities(model, train.loc[valid])
        folds.append(
            {
                "validation_site": site,
                "training_sites": sorted(sites[fit].unique()),
                "fit": int(fit.sum()),
                "validate": int(valid.sum()),
                "purged": int(duplicate.sum()),
                "train_digest": position_digest(tuple(train.index[fit])),
                "validation_digest": position_digest(tuple(train.index[valid])),
            }
        )
    labels, scores = np.asarray(y, dtype=np.int64), np.asarray(oof, dtype=np.float64)
    cut = choose_threshold(labels, scores)
    sigmoid = fit_sigmoid(labels, scores)
    final = make_model(tier, family, value)
    final.fit(train, y)
    return FittedTransfer(
        probabilities(final, test), cut, sigmoid, probability_digest(scores), folds
    )


def study_site(data: ResearchData, split: Split, manifest: Row) -> Row:
    split.validate(data)
    tr, te = pd.Index(split.train), pd.Index(split.test)
    y = np.asarray(data.target.loc[te], dtype=np.int64)
    n, pos = len(y), int(y.sum())
    observed = {
        "test": n,
        "train": len(tr),
        "positive": pos,
        "negative": n - pos,
        "training_exclusions": len(split.excluded),
    }
    if observed != manifest["sites"][split.hospital]:
        raise ValueError(f"Unexpected split counts for {split.hospital}.")
    weights = group_weights(
        np.asarray(data.groups.loc[te], dtype=np.int64), manifest["uncertainty"]["repetitions"]
    )
    rows, sensitivity, disagreement, unavailable = [], [], [], []
    decisions: dict[tuple[str, str], NDArray[np.bool_]] = {}
    paired: dict[tuple[str, str], Probabilities] = {}
    training_sites = data.metadata.loc[tr, "dataset"]
    inner: list[Row] = []
    for config in manifest["models"]:
        tier, family, value = Tier(config["tier"]), config["family"], config["value"]
        if parameters(family, value) != config["parameters"]:
            raise ValueError("Estimator settings disagree with frozen protocol.")
        baseline: Row = {}
        for policy in Cholesterol:
            x = data.features(tier, policy)
            if list(x.columns) != manifest["tiers"][tier.value]:
                raise ValueError("Feature columns differ from manifest.")
            trained = transfer_predictions(
                x.loc[tr],
                data.target.loc[tr],
                training_sites,
                data.groups.loc[tr],
                x.loc[te],
                tier,
                family,
                value,
            )
            inner = trained.folds
            modes = [("raw", trained.probability, trained.threshold)]
            if trained.sigmoid:
                modes.append(
                    (
                        "sigmoid",
                        trained.sigmoid.apply(trained.probability),
                        float(trained.sigmoid.apply(np.array([trained.threshold]))[0]),
                    )
                )
            else:
                unavailable.append(
                    {
                        "tier": tier.value,
                        "family": family,
                        "policy": policy.value,
                        "reason": "Insufficient support, variation or positive sigmoid slope.",
                    }
                )
            for mode, p, cut in modes:
                scores = metrics(y, p, cut)
                ap = scores["average_precision"]
                scores["ap_minus_prevalence"] = ap - pos / n if ap is not None else None
                scores["ap_over_prevalence"] = ap / (pos / n) if ap is not None and pos else None
                samples = boot_scores(y, p, cut, weights)
                row = {
                    "tier": tier.value,
                    "family": family,
                    "policy": policy.value,
                    "mode": mode,
                    "threshold": cut,
                    "metrics": scores,
                    "intervals": {k: interval(v) for k, v in samples.items()},
                    "diagnostics": calibration_diagnostics(y, p),
                    "oof_sha256": trained.oof_digest,
                    "probability_sha256": probability_digest(p),
                    "sigmoid_fit": asdict(trained.sigmoid)
                    if mode == "sigmoid" and trained.sigmoid
                    else None,
                    "confident_fp": int(((y == 0) & (p >= 0.9) & (p >= cut)).sum()),
                    "confident_fn": int(((y == 1) & (p <= 0.1) & (p < cut)).sum()),
                }
                rows.append(row)
                if policy == Cholesterol.RECORDED:
                    baseline[mode] = (p, cut, scores)
                    if mode == "raw":
                        decisions[tier.value, family] = p >= cut
                        paired[tier.value, family] = samples["balanced_accuracy"]
                elif mode in baseline:
                    old, old_cut, old_scores = baseline[mode]
                    deltas = {
                        k: scores[k] - old_scores[k]
                        if scores[k] is not None and old_scores[k] is not None
                        else None
                        for k in ("balanced_accuracy", "roc_auc", "brier", "log_loss")
                    }
                    shift = float(np.abs(p - old).mean())
                    sensitivity.append(
                        {
                            "tier": tier.value,
                            "family": family,
                            "mode": mode,
                            "mean_abs_probability_change": shift,
                            "changed_decisions": int(((old >= old_cut) != (p >= cut)).sum()),
                            "deltas": deltas,
                            "flag": shift >= 0.05 or abs(deltas["balanced_accuracy"] or 0) >= 0.05,
                        }
                    )
    for tier in Tier:
        for a, b in combinations(("logistic", "forest", "xgboost"), 2):
            disagreement.append(
                {
                    "tier": tier.value,
                    "families": [a, b],
                    "different_decisions": int(
                        (decisions[tier.value, a] != decisions[tier.value, b]).sum()
                    ),
                }
            )
    base = paired["early", "logistic"]
    contrasts = {
        key: interval(paired[k] - base)
        for key, k in [
            ("extended_lr_minus_early_lr", ("extended", "logistic")),
            ("early_xgb_minus_early_lr", ("early", "xgboost")),
        ]
    }
    predictors = data.predictors
    return {
        "hospital": split.hospital,
        **observed,
        "prevalence": pos / n,
        "training_sites": sorted(training_sites.unique()),
        "train_sha256": position_digest(split.train),
        "test_sha256": position_digest(split.test),
        "inner_folds": inner,
        "test_groups": data.groups.loc[te].nunique(),
        "test_chol_zeros": int(predictors.loc[te, "chol"].eq(0).sum()),
        "missing_pct": {
            side: (predictors.loc[idx].isna().mean() * 100).to_dict()
            for side, idx in [("train", tr), ("test", te)]
        },
        "runs": rows,
        "sigmoid_unavailable": unavailable,
        "paired_ba_intervals": contrasts,
        "cholesterol_sensitivity": sensitivity,
        "disagreement": disagreement,
    }


def aggregate(sites: list[Row]) -> list[Row]:
    buckets: dict[tuple[str, ...], list[tuple[int, Row]]] = defaultdict(list)
    for site in sites:
        for r in site["runs"]:
            buckets[tuple(r[k] for k in ("tier", "family", "policy", "mode"))].append(
                (site["test"], r["metrics"])
            )
    summary = []
    for key, items in buckets.items():
        stats = {}
        for metric in items[0][1]:
            if metric in {"tn", "fp", "fn", "tp"}:
                continue
            available = [(n, m[metric]) for n, m in items if m[metric] is not None]
            stats[metric] = {
                "sites": len(available),
                "macro": float(np.mean([v for _, v in available])) if available else None,
                "patient_weighted": sum(n * v for n, v in available) / sum(n for n, _ in available)
                if available
                else None,
            }
        summary.append(
            dict(zip(("tier", "family", "policy", "mode"), key, strict=True)) | {"metrics": stats}
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    from .transfer_report import write_reports

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("docs/phase4-protocol.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase4"))
    args = parser.parse_args(argv)
    start = time.perf_counter()
    try:
        manifest = read_manifest(args.manifest)
        versions = {
            k: platform.python_version() if k == "python" else version(k)
            for k in manifest["software"]
        }
        if versions != manifest["software"]:
            raise ValueError(
                "Install the environment in requirements-dev.lock to reproduce the freeze."
            )
        dataset = load_dataset(args.input)
        if dataset.sha256 != manifest["dataset_sha256"]:
            raise ValueError("Unexpected input checksum.")
        data = prepare_data(dataset.frame)
        with threadpool_limits(limits=1):
            sites = [study_site(data, s, manifest) for s in hospital_splits(data)]
        record = rounded(
            {
                "schema_version": 1,
                "protocol_commit": FREEZE,
                "protocol_sha256": MANIFEST_HASH,
                "protocol": manifest,
                "sites": sites,
                "aggregate": aggregate(sites),
            }
        )
        record["scientific_sha256"] = digest(record)
        record["experiment_seconds"] = round(time.perf_counter() - start, 6)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output_dir) as temp:
            staging = Path(temp)
            write_reports(record, staging)
            atomic_text(
                staging / "hospital-results.json",
                json.dumps(record, indent=2, allow_nan=False) + "\n",
            )
            files = sorted(staging.iterdir(), key=lambda p: p.name == "hospital-results.json")
            for file in files:
                destination = args.output_dir / file.name
                if destination.exists() and any(
                    destination.samefile(source) for source in (args.input, args.manifest)
                ):
                    raise ValueError("Report destination overlaps an input.")
            for file in files:
                file.replace(args.output_dir / file.name)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Transfer experiment failed: {exc}", file=sys.stderr)
        return 1
    print(f"Completed {len(sites)} sites. Scientific SHA-256 {record['scientific_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
