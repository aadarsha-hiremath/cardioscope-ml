"""Two-stage pooled experiment. CV writes a selection record required by evaluation."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .comparison import render
from .data.loading import load_dataset
from .data.schema import PREDICTORS, RELEASE_SHA256
from .preprocessing import EARLY, Cholesterol, ResearchData, Tier, prepare_data
from .protocol import position_digest
from .scoring import balanced_interval, metrics
from .splits import SEED, Split, conventional_split
from .training import (
    GRID,
    fold_records,
    inner_folds,
    make_model,
    parameters,
    probabilities,
    probability_digest,
    select_model,
)

PHASE2_POSITIONS = (
    "039d450b7f2a4bd4818e2cf4782c00d6cbf1382b139971e40d01c722fe7d9e44",
    "cf80c09963c65a177017bf2d6fec7eeb0ce30192b8eb6a285ec806d32086b2be",
)


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [rounded(x) for x in value]
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    return value


def digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def peak_working_set_mib() -> float | None:
    """Windows process high-water mark, including native estimator memory."""
    if sys.platform != "win32":
        return None

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("size", ctypes.c_uint32), ("page_faults", ctypes.c_uint32)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak_working_set",
                "working_set",
                "peak_paged_pool",
                "paged_pool",
                "peak_nonpaged_pool",
                "nonpaged_pool",
                "pagefile",
                "peak_pagefile",
            )
        ]

    counters = MemoryCounters()
    counters.size = ctypes.sizeof(counters)
    success = ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(-1), ctypes.byref(counters), counters.size
    )
    return float(counters.peak_working_set) / (1024**2) if success else None


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, delete=False, mode="w", encoding="utf-8", newline="\n"
    ) as handle:
        temp = Path(handle.name)
        try:
            handle.write(content)
        except BaseException:
            handle.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def metadata(data: ResearchData, split: Split, sha: str) -> dict[str, Any]:
    split.validate(data)
    train, test = pd.Index(split.train), pd.Index(split.test)
    return {
        "dataset_sha256": sha,
        "seed": SEED,
        "cpu_threads": 1,
        "versions": {
            "python": platform.python_version(),
            **{
                p: version(p)
                for p in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "threadpoolctl")
            },
        },
        "positive_class": "num in 1..4",
        "cholesterol": "as_recorded",
        "split_rule": "first fold of five shuffled site/outcome-stratified group folds",
        "train_positions_sha256": position_digest(split.train),
        "test_positions_sha256": position_digest(split.test),
        "train_rows": len(train),
        "train_positive": int(data.target.iloc[train].sum()),
        "test_rows": len(test),
        "test_positive": int(data.target.iloc[test].sum()),
        "feature_columns": {"early": list(EARLY), "extended": list(PREDICTORS)},
        "preprocessing": "Phase 2 median/standard scale; constant missing label/one-hot "
        "unknown ignored; fresh pipeline per fit",
        "selection": "highest mean fold AP, first candidate on tie",
        "threshold_rule": (
            "Max OOF balanced accuracy, 0.20 to 0.80 by 0.05, nearest 0.50 then lower"
        ),
    }


def development(data: ResearchData, split: Split, tiers: list[Tier]) -> dict[str, Any]:
    split.validate(data)
    train = pd.Index(split.train)
    y = data.target.iloc[train]
    folds = tuple(inner_folds(y, data.metadata["dataset"].iloc[train], data.groups.iloc[train]))
    runs = []
    recommended = {}
    for tier in tiers:
        x = data.features(tier, Cholesterol.RECORDED).iloc[train]
        tier_runs = []
        for family in GRID:
            try:
                tier_runs.append(select_model(x, y, folds, tier, family))
            except Exception as exc:
                raise RuntimeError(f"{tier.value}/{family} CV failed: {exc}") from exc
        recommended[tier.value] = max(tier_runs, key=lambda r: r["selected"]["mean_ap"])["family"]
        runs.extend(tier_runs)
    return {"folds": fold_records(y.index, folds), "cv_recommendations": recommended, "runs": runs}


def evaluate(data: ResearchData, split: Split, lock: dict[str, Any]) -> list[dict[str, Any]]:
    split.validate(data)
    train, test = pd.Index(split.train), pd.Index(split.test)
    y = np.asarray(data.target.iloc[test], dtype=np.int64)
    groups = np.asarray(data.groups.iloc[test], dtype=np.int64)
    results = []
    for run in lock["runs"]:
        tier, family = Tier(run["tier"]), run["family"]
        try:
            value, cut = run["selected"]["value"], run["threshold"]
            if parameters(family, value) != run["selected"]["parameters"]:
                raise ValueError("Model configuration differs from CV lock.")
            x = data.features(tier, Cholesterol.RECORDED)
            model = make_model(tier, family, value)
            start = time.perf_counter()
            model.fit(x.iloc[train], data.target.iloc[train])
            duration = time.perf_counter() - start
            p = probabilities(model, x.iloc[test])
            results.append(
                {
                    "tier": tier.value,
                    "family": family,
                    "parameters": parameters(family, value),
                    "threshold": cut,
                    "holdout_metrics": metrics(y, p, cut),
                    "balanced_accuracy_ci95": balanced_interval(y, p, cut, groups),
                    "probability_sha256": probability_digest(p),
                    "refit_seconds": duration,
                }
            )
        except Exception as exc:
            raise RuntimeError(f"{tier.value}/{family} evaluation failed: {exc}") from exc
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--stage", choices=["cv", "evaluate", "report"], required=True)
    parser.add_argument("--tier", choices=["both", "early", "extended"], default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase3"))
    args = parser.parse_args(argv)
    directory = args.output_dir
    cv_file = directory / f"cv-{args.tier}.json"
    result_file = directory / f"pooled-{args.tier}.json"
    md_file = result_file.with_suffix(".md")
    failure_file = directory / f"failure-{args.stage}-{args.tier}.json"
    started, cpu = time.perf_counter(), time.process_time()
    try:
        if args.input and any(
            p.exists() and args.input.exists() and p.samefile(args.input)
            for p in (cv_file, result_file, md_file, failure_file)
        ):
            raise ValueError("Input conflicts with report destination.")
        if args.stage == "report":
            if (directory / f"failure-evaluate-{args.tier}.json").exists():
                raise ValueError("Latest evaluation failed. Re-run evaluation before rendering.")
            atomic_text(md_file, render(json.loads(result_file.read_bytes())))
        else:
            if args.input is None:
                raise ValueError("--input is required for CV and evaluation.")
            dataset = load_dataset(args.input)
            if dataset.sha256 != RELEASE_SHA256:
                raise ValueError("Expected the verified Phase 2 CSV.")
            data = prepare_data(dataset.frame)
            split = conventional_split(data)
            meta = metadata(data, split, dataset.sha256)
            if (meta["train_positions_sha256"], meta["test_positions_sha256"]) != PHASE2_POSITIONS:
                raise ValueError("Pooled split does not reproduce Phase 2.")
            tiers = list(Tier) if args.tier == "both" else [Tier(args.tier)]
            with threadpool_limits(limits=1):
                if args.stage == "cv":
                    lock = rounded({"metadata": meta, **development(data, split, tiers)})
                    timings = {
                        f"{r['tier']}/{r['family']}": r.pop("cv_seconds") for r in lock["runs"]
                    }
                    record = {
                        "selection_sha256": digest(lock),
                        "selection": lock,
                        "cv_training_seconds": timings,
                    }
                    destination = cv_file
                else:
                    if (directory / f"failure-cv-{args.tier}.json").exists():
                        raise ValueError("Latest CV failed. Re-run CV before evaluation.")
                    cv = json.loads(cv_file.read_bytes())
                    lock = cv["selection"]
                    if cv["selection_sha256"] != digest(lock) or lock["metadata"] != meta:
                        raise ValueError("CV lock or dataset, split or environment changed.")
                    if [(r["tier"], r["family"]) for r in lock["runs"]] != [
                        (t.value, family) for t in tiers for family in GRID
                    ]:
                        raise ValueError("CV lock does not cover the requested experiments.")
                    record = {**cv, "evaluation": evaluate(data, split, lock)}
                    destination = result_file
            record[args.stage + "_resources"] = {
                "wall_seconds": time.perf_counter() - started,
                "process_cpu_seconds": time.process_time() - cpu,
                "peak_working_set_mib": peak_working_set_mib(),
            }
            atomic_text(destination, json.dumps(rounded(record), indent=2, allow_nan=False) + "\n")
            if args.stage == "evaluate":
                atomic_text(md_file, render(rounded(record)))
        failure_file.unlink(missing_ok=True)
    except Exception as exc:
        failure = {
            "status": "failed",
            "stage": args.stage,
            "tier": args.tier,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        print(json.dumps(failure), file=sys.stderr)
        if not (
            args.input
            and args.input.exists()
            and failure_file.exists()
            and args.input.samefile(failure_file)
        ):
            try:
                atomic_text(failure_file, json.dumps(failure, indent=2) + "\n")
            except OSError:
                pass  # stderr already carries the failure if the directory is unwritable.
        return 1
    print(f"{args.stage} complete ({args.tier}) in {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
