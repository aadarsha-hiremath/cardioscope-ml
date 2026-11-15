"""Readable comparison from the recorded selections and held-out metrics."""

from typing import Any

from .report import table


def fmt(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def render(record: dict[str, Any]) -> str:
    lock = record["selection"]
    meta = lock["metadata"]
    evals = record["evaluation"]
    negatives = meta["test_rows"] - meta["test_positive"]
    sections = [
        "# Pooled model comparison",
        f"Outcome 1 means original num 1–4. Development used {meta['train_rows']} records "
        f"({meta['train_positive']} positives). The pooled holdout contained {meta['test_rows']} "
        f"records ({meta['test_positive']} positives and {negatives} "
        "negatives). Every row below uses the as-recorded cholesterol policy.",
        "## Training-only selection",
        "AP is the mean of five validation-fold average precision scores. SD describes variation "
        "between folds. OOF balanced accuracy uses the threshold chosen on those same out-of-fold "
        "probabilities and is optimistic as a performance estimate.",
        table(
            [
                "Tier",
                "Model",
                "Selected C or depth",
                "Mean AP",
                "Fold SD",
                "Threshold",
                "OOF balanced accuracy",
            ],
            (
                (
                    r["tier"],
                    r["family"],
                    r["selected"]["value"],
                    fmt(r["selected"]["mean_ap"]),
                    fmt(r["selected"]["sd_ap"]),
                    fmt(r["threshold"]),
                    fmt(r["oof_metrics"]["balanced_accuracy"]),
                )
                for r in lock["runs"]
            ),
        ),
        "Family selected by mean CV AP for each tier: "
        + ", ".join(f"{tier} = {family}" for tier, family in lock["cv_recommendations"].items())
        + ".",
        "## Pooled holdout at locked thresholds",
        table(
            [
                "Tier",
                "Model",
                "Accuracy",
                "Balanced accuracy",
                "95% interval",
                "Precision",
                "Sensitivity",
                "Specificity",
                "F1",
                "TN",
                "FP",
                "FN",
                "TP",
            ],
            (
                [
                    r["tier"],
                    r["family"],
                    fmt(m["accuracy"]),
                    fmt(m["balanced_accuracy"]),
                    "–".join(fmt(v) for v in r["balanced_accuracy_ci95"]["bounds"])
                    if r["balanced_accuracy_ci95"]["bounds"]
                    else "undefined",
                    *[fmt(m[key]) for key in ("precision", "sensitivity", "specificity", "f1")],
                    *[m[key] for key in ("tn", "fp", "fn", "tp")],
                ]
                for r in evals
                for m in [r["holdout_metrics"]]
            ),
        ),
        "## Pooled holdout probabilities",
        table(
            ["Tier", "Model", "ROC-AUC", "Average precision", "Brier", "Log loss"],
            (
                [
                    r["tier"],
                    r["family"],
                    *[
                        fmt(r["holdout_metrics"][key])
                        for key in ("roc_auc", "average_precision", "brier", "log_loss")
                    ],
                ]
                for r in evals
            ),
        ),
        "The interval resamples whole duplicate groups 2,000 times with the fitted model and "
        "threshold fixed. All six evaluations share the same holdout. Small score differences "
        "cannot establish superiority across hospitals. Diagnostic measurements in the extended "
        "tier and site-dependent measurement patterns limit interpretation. Probabilities refer "
        "to the recorded endpoint in this dataset.",
        "## Runtime",
        table(
            ["Tier", "Model", "10 CV fits (s)", "Final refit (s)"],
            (
                (
                    r["tier"],
                    r["family"],
                    fmt(record["cv_training_seconds"][r["tier"] + "/" + r["family"]]),
                    fmt(r["refit_seconds"]),
                )
                for r in evals
            ),
        ),
        "All fits ran serially on CPU with one compute thread. Runtime varies by machine. The "
        "companion JSON records fold and prediction digests, candidate scores, estimator settings, "
        "package versions, seeds and process timings.",
    ]
    return "\n\n".join(sections) + "\n"
