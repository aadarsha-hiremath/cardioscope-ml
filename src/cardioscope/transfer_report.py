"""Aggregate research tables, interpretation and primary-model figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from .experiment import atomic_text
from .report import table

Row = dict[str, Any]
LABEL = {"logistic": "LR", "forest": "RF", "xgboost": "XGB"}


def f(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def ci(item: Row) -> str:
    return "–".join(f(v) for v in item["bounds"]) if item["bounds"] else "undefined"


def cases(site: Row, policy: str = "as_recorded", mode: str = "raw") -> list[Row]:
    return [r for r in site["runs"] if r["policy"] == policy and r["mode"] == mode]


def primary(site: Row, mode: str = "raw") -> Row:
    return next(
        r for r in cases(site, mode=mode) if r["tier"] == "early" and r["family"] == "logistic"
    )


def comparison(record: Row) -> str:
    sections = [
        "# Transfer between hospitals",
        f"Protocol freeze `{record['protocol_commit']}`. "
        "This retrospective stress test refits six configurations at each hospital boundary. "
        "Previous pooled development used all four sources. Class 1 denotes recorded UCI "
        "outcomes 1–4. The main tables use uncalibrated scores and as-recorded cholesterol. "
        "LR = Logistic Regression, RF = Random Forest, XGB = XGBoost.",
        "![Balanced accuracy by hospital](hospital-performance.png)",
    ]
    for s in record["sites"]:
        sections.extend(
            [
                f"## {s['hospital']}",
                f"Training sites: {', '.join(s['training_sites'])} ({s['train']} records). "
                f"Evaluation: {s['test']} records, {s['positive']} positive and "
                f"{s['negative']} negative. "
                f"The AP prevalence baseline is {f(s['prevalence'])}. Training exclusions: "
                f"{s['training_exclusions']}.",
                table(
                    [
                        "Tier",
                        "Model",
                        "Cut",
                        "BA",
                        "Sensitivity",
                        "Specificity",
                        "Precision",
                        "F1",
                        "TN/FP/FN/TP",
                        "AUC",
                        "AP",
                        "AP / prevalence",
                        "Brier",
                        "Log loss",
                    ],
                    (
                        [
                            r["tier"],
                            LABEL[r["family"]],
                            f(r["threshold"]),
                            *[
                                f(m[k])
                                for k in (
                                    "balanced_accuracy",
                                    "sensitivity",
                                    "specificity",
                                    "precision",
                                    "f1",
                                )
                            ],
                            "/".join(str(m[k]) for k in ("tn", "fp", "fn", "tp")),
                            *[
                                f(m[k])
                                for k in (
                                    "roc_auc",
                                    "average_precision",
                                    "ap_over_prevalence",
                                    "brier",
                                    "log_loss",
                                )
                            ],
                        ]
                        for r in cases(s)
                        for m in [r["metrics"]]
                    ),
                ),
                table(
                    ["Tier", "Model", "BA 95%", "Sensitivity 95%", "Specificity 95%", "AUC 95%"],
                    (
                        [
                            r["tier"],
                            LABEL[r["family"]],
                            *[
                                ci(r["intervals"][k])
                                for k in (
                                    "balanced_accuracy",
                                    "sensitivity",
                                    "specificity",
                                    "roc_auc",
                                )
                            ],
                        ]
                        for r in cases(s)
                    ),
                ),
            ]
        )
    sections.extend(
        [
            "## Across-site summaries",
            "Each pair below gives the equal-hospital mean / "
            "record-count-weighted mean of four site estimates. The latter gives Cleveland "
            "more weight than Switzerland. A weighted mean of site AUCs or APs measures "
            "within-site ranking. Cross-site patient pairs do not enter it.",
            table(
                [
                    "Tier",
                    "Model",
                    "BA",
                    "Sensitivity",
                    "Specificity",
                    "AUC",
                    "AP",
                    "Brier",
                    "Log loss",
                ],
                (
                    [
                        r["tier"],
                        LABEL[r["family"]],
                        *[
                            f(m[k]["macro"]) + " / " + f(m[k]["patient_weighted"])
                            for k in (
                                "balanced_accuracy",
                                "sensitivity",
                                "specificity",
                                "roc_auc",
                                "average_precision",
                                "brier",
                                "log_loss",
                            )
                        ],
                    ]
                    for r in record["aggregate"]
                    for m in [r["metrics"]]
                    if r["policy"] == "as_recorded" and r["mode"] == "raw"
                ),
            ),
            "Intervals use 2,000 paired within-site duplicate-group resamples. They "
            "condition on the "
            "observed site, fitted model and cutoff. Training variability and uncertainty "
            "over future "
            "hospitals are outside their scope. Empty or single-class replicates "
            "contribute only to "
            "metrics they define. JSON records the denominator for each interval. Swiss "
            "specificity "
            "moves by 0.125 per negative classification.",
            "All 96 prespecified combinations, including both cholesterol policies and "
            "both probability "
            "methods when feasible, are described in the [machine-readable "
            "results](hospital-results.json). "
            "See [calibration](calibration-analysis.md), "
            "[sensitivity](cholesterol-sensitivity.md) and "
            "[research findings](reliability-findings.md).",
        ]
    )
    return "\n\n".join(sections) + "\n"


def calibration_report(record: Row) -> str:
    sections = [
        "# Probability calibration under transfer",
        "The sigmoid uses regularized logistic regression on training-site out-of-fold log odds. "
        "The transformed cutoff preserves the raw operating point. Held-out calibration intercepts "
        "and slopes are descriptive diagnostics only. An offset intercept (CITL) above zero "
        "indicates underprediction on average. A free-intercept slope below one suggests excessive "
        "spread. Ideal intercept and slope are zero and one, respectively. "
        "[Calibration definitions](https://doi.org/10.1186/s12916-019-1466-7)",
        "![Early logistic calibration by hospital](calibration-curves.png)",
        "The diagonal denotes agreement. Points show observed fractions against mean probabilities "
        "within five fixed probability bands. Point size reflects bin count; empty bins "
        "are omitted. "
        "Sparse bins, especially at Switzerland with eight negatives, make local estimates "
        "unstable.",
    ]
    for s in record["sites"]:
        sections.extend(
            [
                f"## {s['hospital']}",
                table(
                    [
                        "Tier",
                        "Model",
                        "Brier raw → sigmoid",
                        "Log loss raw → sigmoid",
                        "CITL raw → sigmoid",
                        "Slope raw → sigmoid",
                        "Mean p raw → sigmoid",
                        "Observed",
                    ],
                    (
                        [
                            r["tier"],
                            LABEL[r["family"]],
                            *[
                                f(r[section][key]) + " → " + f(c[section][key])
                                for section, key in [
                                    ("metrics", "brier"),
                                    ("metrics", "log_loss"),
                                    ("diagnostics", "citl"),
                                    ("diagnostics", "slope"),
                                    ("diagnostics", "mean_p"),
                                ]
                            ],
                            f(s["prevalence"]),
                        ]
                        for r in cases(s)
                        for c in cases(s, mode="sigmoid")
                        if (r["tier"], r["family"]) == (c["tier"], c["family"])
                    ),
                ),
                "Early Logistic Regression bin counts and observed fractions under the "
                "primary cholesterol policy.",
                table(
                    ["Band", "Method", "Records", "Positive", "Mean p", "Observed"],
                    (
                        [
                            f"{b['from']:.1f}–{b['to']:.1f}",
                            r["mode"],
                            b["n"],
                            b["positive"],
                            f(b["mean_p"]),
                            f(b["observed"]),
                        ]
                        for r in s["runs"]
                        if r["tier"] == "early"
                        and r["family"] == "logistic"
                        and r["policy"] == "as_recorded"
                        for b in r["diagnostics"]["bins"]
                    ),
                ),
            ]
        )
    sections.append(
        "Sigmoid support requires at least 20 training OOF observations per class, "
        "varying scores and a positive fitted slope. An undefined diagnostic slope "
        "can reflect insufficient class support, constant scores or separation. "
        "The JSON includes every bin, fitted sigmoid and any unavailable method."
    )
    return "\n\n".join(sections) + "\n"


def sensitivity_report(record: Row) -> str:
    sections = [
        "# Sensitivity to recorded cholesterol zeros",
        "Both policies retain the same records and partitions. Masking zeros affects cholesterol "
        "alone. The complete pipeline, training-site cutoff and sigmoid are refitted for each "
        "policy. Changes below compare masked minus recorded for raw probabilities. The frozen "
        "descriptive flag marks mean absolute probability change or absolute balanced accuracy "
        "change of at least 0.05. This is an operational threshold for this experiment.",
        "All 123 Swiss records and 49 of 200 VA records contain a cholesterol zero. Cleveland and "
        "Hungary contain none. A recorded zero therefore conveys site information even though site "
        "is excluded from predictors. Masking can also change predictions at sites without zeros "
        "through changes in the training distribution. No Swiss cholesterol measurement remains "
        "under masking; Swiss imputation comes entirely from the other hospitals.",
    ]
    for s in record["sites"]:
        sections.extend(
            [
                f"## {s['hospital']}",
                table(
                    [
                        "Tier",
                        "Model",
                        "Mean |Δp|",
                        "Changed calls",
                        "ΔBA",
                        "ΔAUC",
                        "ΔBrier",
                        "Δlog loss",
                        "Flag",
                    ],
                    (
                        [
                            r["tier"],
                            LABEL[r["family"]],
                            f(r["mean_abs_probability_change"]),
                            r["changed_decisions"],
                            *[
                                f(r["deltas"][k])
                                for k in ("balanced_accuracy", "roc_auc", "brier", "log_loss")
                            ],
                            "yes" if r["flag"] else "no",
                        ]
                        for r in s["cholesterol_sensitivity"]
                        if r["mode"] == "raw"
                    ),
                ),
            ]
        )
    sections.append(
        "Prediction sensitivity quantifies dependence on an encoding assumption. "
        "The meaning of a zero in the collection process remains unresolved. Both policies "
        "and the corresponding calibrated comparisons are retained in the JSON."
    )
    return "\n\n".join(sections) + "\n"


def findings(record: Row) -> str:
    sites = record["sites"]
    ranked = sorted(sites, key=lambda s: primary(s)["metrics"]["balanced_accuracy"])
    worst, best = ranked[0], ranked[-1]
    gain, loss = [], []
    xgb_above = 0
    for site in sites:
        pairs = site["paired_ba_intervals"]
        lo, hi = pairs["extended_lr_minus_early_lr"]["bounds"]
        if lo > 0:
            gain.append(site["hospital"])
        if hi < 0:
            loss.append(site["hospital"])
        xgb_above += int(pairs["early_xgb_minus_early_lr"]["bounds"][0] > 0)
    swiss = next(s for s in sites if s["hospital"] == "Switzerland")
    swiss_raw = primary(swiss)["diagnostics"]
    masked = [
        d
        for s in sites
        for d in s["cholesterol_sensitivity"]
        if (d["tier"], d["family"], d["mode"]) == ("early", "logistic", "raw")
    ]
    sections = [
        "# Reliability findings",
        f"Early Logistic Regression achieved its lowest balanced accuracy at {worst['hospital']} "
        f"({f(primary(worst)['metrics']['balanced_accuracy'])}) and its highest at "
        f"{best['hospital']} "
        f"({f(primary(best)['metrics']['balanced_accuracy'])}). Pooling these sites obscures the "
        "balance of false positives and false negatives. Difficulty depends on which error "
        "matters.",
        table(
            [
                "Site",
                "Prevalence",
                "BA (95%)",
                "False negative / positive cases",
                "False positive / negative cases",
                "High-confidence FN / FP",
                "AP − prevalence",
                "CITL",
                "Calibration slope",
            ],
            (
                [
                    s["hospital"],
                    f(s["prevalence"]),
                    f(m["balanced_accuracy"])
                    + " ("
                    + ci(r["intervals"]["balanced_accuracy"])
                    + ")",
                    f"{m['fn']} / {s['positive']}",
                    f"{m['fp']} / {s['negative']}",
                    f"{r['confident_fn']} / {r['confident_fp']}",
                    f(m["ap_minus_prevalence"]),
                    f(r["diagnostics"]["citl"]),
                    f(r["diagnostics"]["slope"]),
                ]
                for s in sites
                for r in [primary(s)]
                for m in [r["metrics"]]
            ),
        ),
        "High-confidence errors use p ≤ 0.10 for false negatives and p ≥ 0.90 for false positives. "
        "Hungary contains 106 positives among 293 records (36.2%); Switzerland contains 115 among "
        "123 (93.5%). A high Swiss AP therefore starts from a high prevalence baseline. Eight "
        "negatives provide little precision for specificity, even when sensitivity looks stable.",
        "The training populations led to different operating points. Hungary's high threshold "
        "favored specificity and missed over half its recorded positives. VA Long Beach had the "
        "lowest specificity. Site, threshold and case mix change together in this design, so their "
        "individual contributions cannot be isolated.",
        "## Do extended features or boosting help?",
        "These within-site contrasts use the primary raw, as-recorded policy and each model’s "
        "own locked cutoff. Paired intervals reuse identical patient-group draws. "
        "Intervals excluding "
        "zero give conditional evidence within that site; the comparisons have no "
        "multiplicity correction.",
        table(
            [
                "Site",
                "Extended LR − early LR BA (95%)",
                "Early XGB − early LR BA (95%)",
                "Early LR/RF, LR/XGB, RF/XGB differing calls",
            ],
            (
                [
                    s["hospital"],
                    *[
                        f(
                            r["metrics"]["balanced_accuracy"]
                            - primary(s)["metrics"]["balanced_accuracy"]
                        )
                        + " ("
                        + ci(s["paired_ba_intervals"][key])
                        + ")"
                        for tier, family, key in [
                            ("extended", "logistic", "extended_lr_minus_early_lr"),
                            ("early", "xgboost", "early_xgb_minus_early_lr"),
                        ]
                        for r in cases(s)
                        if r["tier"] == tier and r["family"] == family
                    ],
                    " / ".join(
                        str(d["different_decisions"])
                        for d in s["disagreement"]
                        if d["tier"] == "early"
                    ),
                ]
                for s in sites
            ),
        ),
        f"Paired intervals supported higher extended-tier LR balanced accuracy in "
        f"{', '.join(gain) or 'no site'} and lower balanced accuracy in "
        f"{', '.join(loss) or 'no site'}. Early XGBoost had a positive lower bound against early "
        f"LR in {xgb_above} of {len(sites)} sites. The data offer no uniform boosting advantage.",
        "## Measurement availability and probability reliability",
        table(
            [
                "Site",
                "Missing ca % train → test",
                "Missing thal % train → test",
                "Missing slope % train → test",
                "Cholesterol zeros",
                "ΔBrier sigmoid − raw (early LR)",
                "Mean |Δp| masking zeros (early LR)",
            ],
            (
                [
                    s["hospital"],
                    *[
                        f"{s['missing_pct']['train'][k]:.1f} → {s['missing_pct']['test'][k]:.1f}"
                        for k in ("ca", "thal", "slope")
                    ],
                    s["test_chol_zeros"],
                    f(c["metrics"]["brier"] - primary(s)["metrics"]["brier"]),
                    f(d["mean_abs_probability_change"]),
                ]
                for s in sites
                for c in cases(s, mode="sigmoid")
                if c["tier"] == "early" and c["family"] == "logistic"
                for d in s["cholesterol_sensitivity"]
                if (d["tier"], d["family"], d["mode"]) == ("early", "logistic", "raw")
            ),
        ),
        f"In Switzerland the primary model assigned a mean probability of {f(swiss_raw['mean_p'])} "
        f"against an observed fraction of {f(swiss['prevalence'])}. The sigmoid learned at "
        "the other hospitals left substantial underprediction. At VA Long Beach a slope below "
        "one and high-confidence false positives are consistent with excessive probability spread. "
        "Calibration also sometimes worsened: the extended forest had higher Brier score after "
        "sigmoid calibration in Switzerland and VA Long Beach. See the bin counts and loss "
        "comparisons in the [calibration analysis](calibration-analysis.md).",
        f"Masking cholesterol zeros changed {sum(d['changed_decisions'] for d in masked)} of "
        f"{sum(s['test'] for s in sites)} primary-model classifications after retraining. "
        f"{sum(d['flag'] for d in masked)} of four sites crossed the prespecified sensitivity "
        "flag. The Swiss primary model changed no classifications, despite all its cholesterol "
        "values being zero. Changes at Cleveland and Hungary arose through training and cutoff "
        "changes at other sites. Neither policy establishes the original recording convention.",
        "## Scope of evidence",
        "The project contribution is the frozen transfer design, duplicate-aware site boundaries, "
        "cross-fitted thresholds and calibration, paired sensitivity experiments and reproducible "
        "failure analysis. The original records come from the UCI investigators via the Kaggle "
        "combined release. The three classification algorithms are established methods. "
        "[Data provenance](../../docs/provenance.md) · [Frozen "
        "protocol](../../docs/phase4-protocol.json)",
        "The study estimates transfer among four historical sources using a recorded "
        "binary outcome. "
        "Previously pooled development informed the model and feature choices. Inner training sets "
        "overlap and site practices can recur across records. Bootstrap intervals omit fitting and "
        "between-hospital uncertainty. A genuinely untouched external cohort with known "
        "measurement "
        "timing and independent outcome ascertainment is needed to assess prospective "
        "generalization. "
        "These results cannot establish a clinical screening or treatment use.",
    ]
    return "\n\n".join(sections) + "\n"


def figures(sites: list[Row], directory: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 3.6), layout="constrained")
    for i, s in enumerate(sites):
        r = primary(s)
        lo, hi = r["intervals"]["balanced_accuracy"]["bounds"]
        ax.hlines(i, lo, hi, color="#22636c", linewidth=2)
        ax.plot(r["metrics"]["balanced_accuracy"], i, "o", color="#13404a")
    ax.axvline(0.5, ls="--", color="gray", lw=1)
    ax.set(
        yticks=range(4),
        yticklabels=[s["hospital"] for s in sites],
        xlim=(0, 1),
        xlabel="Balanced accuracy · 95% group-bootstrap interval",
        title="Early Logistic Regression on each held-out hospital",
    )
    ax.invert_yaxis()
    fig.savefig(directory / "hospital-performance.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5), sharex=True, sharey=True, layout="constrained")
    for ax, s in zip(axes.flat, sites, strict=True):
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8)
        for mode, marker, color in [("raw", "o", "#176678"), ("sigmoid", "^", "#b35220")]:
            rows = [
                r for r in cases(s, mode=mode) if r["tier"] == "early" and r["family"] == "logistic"
            ]
            if not rows:
                continue
            bins = [b for b in rows[0]["diagnostics"]["bins"] if b["n"]]
            x = [b["mean_p"] for b in bins]
            y = [b["observed"] for b in bins]
            ax.plot(x, y, color=color, lw=0.9)
            ax.scatter(
                x,
                y,
                s=[10 + 1.2 * b["n"] for b in bins],
                marker=marker,
                color=color,
                alpha=0.8,
                label=mode,
            )
        ax.set(
            title=f"{s['hospital']} · {s['positive']}+ / {s['negative']}−",
            xlim=(-0.04, 1.04),
            ylim=(-0.04, 1.04),
            xticks=np.linspace(0, 1, 5),
        )
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Mean predicted probability in bin")
    fig.supylabel("Observed positive fraction")
    fig.suptitle("Early Logistic Regression · as-recorded cholesterol")
    fig.savefig(directory / "calibration-curves.png", dpi=160)
    plt.close(fig)


def write_reports(record: Row, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, writer in [
        ("hospital-comparison", comparison),
        ("calibration-analysis", calibration_report),
        ("cholesterol-sensitivity", sensitivity_report),
        ("reliability-findings", findings),
    ]:
        atomic_text(directory / (name + ".md"), writer(record))
    figures(record["sites"], directory)
