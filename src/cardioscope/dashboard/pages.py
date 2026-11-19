"""Native Streamlit views backed by verified aggregate research reports."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

import pandas as pd
import streamlit as st

from . import charts
from .evidence import (
    FAMILIES,
    METRICS,
    ROOT,
    Evidence,
    EvidenceError,
    Record,
    load_evidence,
    site_run,
)

DISCLAIMER = (
    "Educational machine learning research. Not a clinically "
    "validated diagnostic or risk assessment system."
)
PERCENT_METRICS = {"balanced_accuracy", "sensitivity", "specificity", "precision", "f1"}


def header(title: str, subtitle: str) -> None:
    st.caption("CARDIOSCOPE ML / HEART DISEASE CLASSIFICATION & MODEL RELIABILITY LAB")
    st.title(title)
    st.markdown(subtitle)


def fmt(value: Any) -> str:
    return "Unavailable" if value is None else f"{value:.3f}"


def fmt_metric(key: str, value: float | None) -> str:
    return f"{value:.1%}" if key in PERCENT_METRICS and value is not None else fmt(value)


def cards(values: dict[str, Any]) -> None:
    for index, (col, (label, value)) in enumerate(
        zip(st.columns(len(values)), values.items(), strict=True)
    ):
        with col.container(border=True, key=f"metric_card_{index}"):
            st.metric(label, value)


def plot(figure: Any, alt: str) -> None:
    st.plotly_chart(
        figure, alt=alt, width="stretch", config={"displaylogo": False, "responsive": True}
    )


def confusion_plot(metrics: Record) -> None:
    plot(
        charts.confusion(metrics),
        f"Confusion matrix. {metrics['tn']} true negatives, {metrics['fp']} false positives, "
        f"{metrics['fn']} false negatives, {metrics['tp']} true positives.",
    )


def selectors(page: str) -> tuple[str, str]:
    left, right = st.columns(2)
    family = left.selectbox(
        "Model family", list(FAMILIES), format_func=FAMILIES.__getitem__, key=f"{page}_model"
    )
    tier = right.selectbox(
        "Feature tier", ["early", "extended"], format_func=str.capitalize, key=f"{page}_tier"
    )
    return family, tier


def metrics_table(metrics: Record) -> None:
    st.dataframe(
        pd.DataFrame(
            [
                {"Metric": title, "Value": fmt_metric(key, metrics.get(key))}
                for key, title in METRICS.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def hospital_select(e: Evidence, key: str) -> Record:
    hospital = st.selectbox(
        "Held-out hospital", [s["hospital"] for s in e.transfer["sites"]], key=key
    )
    return next(s for s in e.transfer["sites"] if s["hospital"] == hospital)


def overview(e: Evidence) -> None:
    header(
        "Understanding Model Reliability Across Hospitals",
        (
            "Comparing heart-disease classification models under distribution "
            "shift, missing clinical measurements, and changing hospital "
            "populations."
        ),
    )
    st.info(DISCLAIMER, icon=":material/info:")
    primary = e.protocol["primary"]
    sites = e.transfer["sites"]
    values = [site_run(s, primary["family"], primary["tier"]) for s in sites]
    scores = [r["metrics"]["balanced_accuracy"] for r in values if r is not None]
    cards(
        {
            "Dataset records": e.audit["rows"],
            "Hospital datasets": len(sites),
            "Model families": len(FAMILIES),
            "Mean balanced accuracy": fmt_metric("balanced_accuracy", sum(scores) / len(scores)),
        }
    )
    st.caption(
        "Observed retrospective result · Primary early Logistic "
        "Regression · Equal weight per hospital · Raw scores · Cholesterol zeros preserved"
    )
    with st.container(border=True):
        st.subheader("One model family. Four different populations.")
        selected = st.selectbox(
            "Inspect hospital", ["All hospitals"] + [s["hospital"] for s in sites]
        )
        plot(
            charts.hospital_chart(
                sites,
                primary["family"],
                primary["tier"],
                None if selected == "All hospitals" else selected,
            ),
            "Primary model balanced accuracy and 95% bootstrap interval at each hospital.",
        )
        st.caption(
            "95% within-site group-bootstrap intervals. The dotted line marks "
            "balanced accuracy of 50%; it is not a clinical decision "
            "threshold."
        )
        if selected != "All hospitals":
            site = next(s for s in sites if s["hospital"] == selected)
            st.write(
                f"{selected}: {site['test']} records, {site['positive']} recorded "
                f"positives and {site['negative']} negatives. Training used the "
                f"other hospitals."
            )
    left, right = st.columns(2)
    with left:
        st.subheader("What the evidence shows")
        st.write(
            f"Primary balanced accuracy ranges from {min(scores):.1%} to "
            f"{max(scores):.1%}. Transfer errors depend on the hospital, not "
            f"just the classifier."
        )
        st.write(
            "Logistic Regression was selected by training-only mean fold "
            "average precision in both tiers. Random Forest and XGBoost "
            "provide controlled comparisons; complexity does not establish "
            "superiority."
        )
        st.page_link(
            "views/reliability.py",
            label="Explore hospital reliability",
            icon=":material/arrow_forward:",
        )
    with right:
        st.subheader("Why generalization matters")
        st.write(
            "A pooled holdout samples the same mixture of hospitals as "
            "development. Hospital-held-out evaluation tests transfer to "
            "different prevalence and measurement patterns. Diagnostic "
            "features do not improve every hospital."
        )
        st.warning(
            "All four hospitals informed earlier pooled development. This is "
            "a retrospective transfer stress test, not untouched external or "
            "prospective validation."
        )
        st.page_link(
            "views/methodology.py", label="Read the research protocol", icon=":material/menu_book:"
        )


def dataset(e: Evidence) -> None:
    header(
        "Dataset Explorer",
        (
            "Inspect the historical cohort and its measurement gaps through "
            "aggregate audit evidence."
        ),
    )
    a = e.audit
    cards(
        {
            "Records": a["rows"],
            "Original target categories": len(a["original_target"]),
            "Recorded positive": a["binary_target"]["1"],
            "Recorded negative": a["binary_target"]["0"],
        }
    )
    st.caption(
        "Binary target: num = 0 → class 0; num = 1–4 → class 1. These are "
        "recorded historical labels."
    )
    kind = st.selectbox(
        "Hospital comparison", ["Record count", "Class counts", "Positive prevalence"]
    )
    hospitals = a["hospitals"]
    names = [h["hospital"] for h in hospitals]
    if kind == "Class counts":
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_bar(
            x=names,
            y=[h["positive"] for h in hospitals],
            name="Recorded 1",
            marker_color=charts.CYAN,
            hovertemplate="%{x}<br>Recorded 1: %{y:d}<extra></extra>",
        )
        fig.add_bar(
            x=names,
            y=[h["rows"] - h["positive"] for h in hospitals],
            name="Recorded 0",
            marker_color=charts.TEAL,
            hovertemplate="%{x}<br>Recorded 0: %{y:d}<extra></extra>",
        )
        fig.update_layout(barmode="stack", yaxis_title="Records")
        plot(charts.style(fig), "Recorded positive and negative counts by hospital.")
    else:
        plot(
            charts.bars(
                names,
                [
                    h["rows"] if kind == "Record count" else h["prevalence_percent"]
                    for h in hospitals
                ],
                "Records" if kind == "Record count" else "Positive prevalence (%)",
                value_format=":d" if kind == "Record count" else ":.3f",
                suffix="" if kind == "Record count" else "%",
            ),
            f"{kind} by hospital.",
        )
    st.subheader("Feature quality & availability")
    scope = st.selectbox("Missingness population", ["All hospitals"] + names)
    missing = (
        {k: v["missing"] for k, v in a["fields"].items()}
        if scope == "All hospitals"
        else next(h["missing"] for h in hospitals if h["hospital"] == scope)
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Feature": key,
                    "Type": field["expected_type"],
                    "Role": field["role"],
                    "Missing records": missing[key]["count"],
                    "Missing (%)": missing[key]["percent"],
                    "Availability": "Early + extended"
                    if key in e.protocol["tiers"]["early"]
                    else "Extended"
                    if key in e.protocol["tiers"]["extended"]
                    else "Metadata only",
                }
                for key, field in a["fields"].items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Unresolved anomalies", expanded=True):
        st.write(
            "Phase 2 matched one profile labeled Hungary in UCI to Cleveland "
            "in the combined release, accounting for the count discrepancy. "
            "The editing history and true enrollment site remain unverified. "
            "We preserve the release's site labels."
        )
        st.write(
            f"Recorded zeros: cholesterol "
            f"{a['fields']['chol']['review']['zero']}; resting blood pressure "
            f"{a['fields']['trestbps']['review']['zero']}. Negative oldpeak: "
            f"{a['fields']['oldpeak']['review']['negative ST depression']}. "
            f"Their clinical meaning remains unresolved."
        )
        zero_sites = [
            f"{h['hospital']}: {zeros}"
            for h in hospitals
            if (zeros := h["review"].get("chol", {}).get("zero", 0))
        ]
        st.caption("Cholesterol zeros by hospital · " + " · ".join(zero_sites))
        st.write(
            "Duplicate-like records are grouped across split boundaries; "
            "similarity alone does not prove patient identity. Highly sparse "
            "ca and thal remain in the extended tier with documented "
            "missingness."
        )
        st.json({"duplicate_excess": a["duplicate_excess"]}, expanded=False)
    st.subheader("Source & handling")
    with st.expander("Verified Phase 2 preprocessing audit"):
        pooled = e.preprocessing["experiments"]["pooled"]["runs"]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Tier": r["tier"],
                        "Cholesterol policy": r["cholesterol"],
                        "Training rows": r["train"]["rows"],
                        "Test rows": r["test"]["rows"],
                        "Encoded columns": len(r["output_columns"]),
                        "Training zeros masked": r["train"]["chol_zeros_masked"],
                    }
                    for r in pooled
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Transformers were fitted on training rows only. Source columns agree "
            "with the frozen feature tiers."
        )
    st.write(
        "Kaggle combined UCI Heart Disease Data, Redwan Sony, version 6. "
        "Original data remain unchanged. Median imputation and scaling "
        "are fitted only on permitted training partitions; categorical "
        "missing values receive an explicit token. Hospital, identifier "
        "and outcome are excluded as predictors."
    )
    st.caption(f"Verified CSV SHA-256: {a['sha256']}")
    st.page_link("views/methodology.py", label="Licensing, attribution and reproduction")


def laboratory(e: Evidence) -> None:
    header(
        "Model Laboratory",
        (
            "Compare fixed model families and information settings under "
            "explicit evaluation protocols."
        ),
    )
    family, tier = selectors("lab")
    protocol = st.selectbox(
        "Evaluation protocol", ["Pooled holdout", "Training-only cross-validation"]
    )
    cv = protocol == "Training-only cross-validation"
    rows = [
        r
        for r in (e.cv["selection"]["runs"] if cv else e.pooled["evaluation"])
        if r["tier"] == tier
    ]
    chosen = next(r for r in rows if r["family"] == family)
    metric_key = "oof_metrics" if cv else "holdout_metrics"
    metric = chosen[metric_key]
    st.subheader("Development-fold average precision" if cv else "Pooled holdout average precision")
    plot(
        charts.bars(
            [FAMILIES[r["family"]] for r in rows],
            [r["selected"]["mean_ap"] if cv else r[metric_key]["average_precision"] for r in rows],
            "Average precision",
            True,
        ),
        f"{protocol} average precision for each model in the {tier} tier.",
    )
    if cv:
        st.warning(
            "Below: concatenated development out-of-fold metrics, with "
            "threshold chosen on those same predictions. These are selection "
            "diagnostics, not unbiased external performance. Mean fold AP "
            "above differs from concatenated OOF AP."
        )
    else:
        st.caption(
            f"One shared pooled holdout: "
            f"{e.pooled['selection']['metadata']['test_rows']} records. "
            f"Threshold locked using development data."
        )
    cards(
        {
            "Balanced accuracy": fmt_metric("balanced_accuracy", metric["balanced_accuracy"]),
            "ROC-AUC": fmt(metric["roc_auc"]),
            "Locked threshold": fmt(chosen["threshold"]),
        }
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Performance measures")
        metrics_table(metric)
    with right:
        st.subheader("Recorded versus predicted class")
        confusion_plot(metric)
        st.caption(
            "False negatives are recorded class 1 predicted as 0; false "
            "positives reverse this. Neither label is a diagnosis."
        )
    with st.expander("Selection rule & fitted configuration", expanded=True):
        st.write(
            "Within each family and tier, training-only cross-validation selected "
            "the candidate with the highest mean fold average precision. Ties went "
            "to the first candidate. "
            f"Recommended family for the {tier} tier: "
            f"{FAMILIES[e.cv['selection']['cv_recommendations'][tier]]}."
        )
        st.write(
            f"**Selected threshold {chosen['threshold']:.3f}.** The threshold maximized "
            "balanced accuracy on development out-of-fold scores. The pooled "
            "holdout used that locked threshold."
        )
        params = chosen["selected"]["parameters"] if cv else chosen["parameters"]
        labels = {
            "C": "Inverse regularization strength",
            "l1_ratio": "L1 regularization ratio",
            "n_estimators": "Number of trees",
            "n_jobs": "CPU jobs",
            "max_iter": "Maximum iterations",
            "tol": "Solver tolerance",
            "reg_lambda": "L2 regularization",
            "reg_alpha": "L1 regularization",
            "eval_metric": "Evaluation metric",
            "random_state": "Random seed",
        }
        st.table(
            pd.DataFrame(
                [
                    {
                        "Setting": labels.get(key, key.replace("_", " ").capitalize()),
                        "Value": str(value) if value is not None else "None",
                    }
                    for key, value in params.items()
                ]
            ).set_index("Setting")
        )
        st.caption(
            "Early features assume routine blood measurements. Extended "
            "features include diagnostic workup and answer a "
            "later-information question. Pooled results do not establish "
            "transfer reliability."
        )
    with st.expander("Advanced configuration"):
        metadata = e.cv["selection"]["metadata"]
        st.json(
            {
                "parameters": params,
                "selection": metadata["selection"],
                "threshold": chosen["threshold"],
                "threshold_rule": metadata["threshold_rule"],
                "preprocessing": metadata["preprocessing"],
            }
        )


def reliability(e: Evidence) -> None:
    header(
        "Hospital Reliability",
        ("A retrospective transfer stress test: fit on three hospitals, evaluate on the fourth."),
    )
    family, tier = selectors("reliability")
    site = hospital_select(e, "reliability_hospital")
    row = site_run(site, family, tier)
    if row is None:
        st.info("This comparison is unavailable in the verified report.")
        return
    plot(
        charts.hospital_chart(e.transfer["sites"], family, tier, site["hospital"]),
        f"Balanced accuracy and 95% bootstrap intervals across hospitals for "
        f"{FAMILIES[family]}, {tier} tier. {site['hospital']} is highlighted.",
    )
    st.caption("Raw scores · Cholesterol as recorded · 95% within-site group-bootstrap intervals")
    cards(
        {
            "Held-out records": site["test"],
            "Recorded positive": site["positive"],
            "Recorded negative": site["negative"],
            "Positive prevalence": f"{site['prevalence']:.1%}",
        }
    )
    st.write(
        f"Training: {', '.join(site['training_sites'])} · {site['train']} "
        f"records. Held out: {site['hospital']}. Classification threshold: "
        f"{row['threshold']:.3f} (learned only from training-site "
        f"out-of-fold scores)."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Selected hospital performance")
        metrics_table(row["metrics"])
    with right:
        st.subheader("Where errors occur")
        confusion_plot(row["metrics"])
    st.subheader("Uncertainty, not just a point estimate")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Metric": METRICS[k],
                    "Estimate": fmt_metric(k, row["metrics"][k]),
                    "95% interval": "Unavailable"
                    if v["bounds"] is None
                    else "–".join(fmt_metric(k, bound) for bound in v["bounds"]),
                    "Valid bootstrap draws": v["valid"],
                }
                for k, v in row["intervals"].items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    if site["negative"] < 20 or site["positive"] < 20:
        st.warning(
            f"Sparse class: {site['negative']} negatives and "
            f"{site['positive']} positives. Specificity may have substantial "
            f"uncertainty. These intervals do not capture future-population "
            f"uncertainty."
        )
    st.write(
        "Prevalence and measurement availability differ across hospitals. "
        "Extended diagnostic features do not consistently improve "
        "transfer. Switzerland has very few negatives. Compare error "
        "counts as well as summary scores."
    )
    st.info(
        "Each site is excluded from its own fit, but all sites informed "
        "earlier pooled development. Retrospective testing cannot "
        "establish future clinical performance."
    )


def calibration(e: Evidence) -> None:
    header(
        "Calibration & Sensitivity",
        (
            "Inspect probability agreement and the consequences of an "
            "unresolved measurement assumption."
        ),
    )
    family, tier = selectors("calibration")
    site = hospital_select(e, "calibration_hospital")
    policy = st.selectbox(
        "Cholesterol interpretation",
        ["as_recorded", "zero_as_missing"],
        format_func=lambda v: "Preserve recorded zeros"
        if v == "as_recorded"
        else "Treat zeros as missing",
    )
    rows = [
        r
        for mode in ("raw", "sigmoid")
        if (r := site_run(site, family, tier, policy, mode)) is not None
    ]
    plot(
        charts.calibration(rows),
        f"Reliability diagram for {site['hospital']}. Observed positive fraction versus "
        "mean model score in each populated bin, with a diagonal for perfect agreement.",
    )
    st.caption(
        "Five fixed equal-width score bins; hover for bin counts. The "
        "diagonal is perfect agreement on this sample, not proof of "
        "clinical validity."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Method": r["mode"],
                    "Brier score": r["metrics"]["brier"],
                    "Log loss": r["metrics"]["log_loss"],
                    "Mean score": r["diagnostics"]["mean_p"],
                    "Observed prevalence": site["prevalence"],
                }
                for r in rows
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    if len(rows) < 2:
        st.warning("Sigmoid comparison unavailable for this configuration.")
        st.json(site["sigmoid_unavailable"])
    st.write(
        "Sigmoid calibration was fitted only to training-site out-of-fold "
        "scores. It can improve or worsen agreement at a new hospital. "
        "Lower Brier score and log loss are better; apparent numerical "
        "precision does not ensure transportability."
    )
    st.subheader("What changes when cholesterol zeros become missing?")
    st.caption(
        "Paired raw-score comparison: alternative minus as-recorded "
        "baseline. Both policies use fresh training-only preprocessing; "
        "neither is selected by these results."
    )
    paired = []
    for s in e.transfer["sites"]:
        for r in s["cholesterol_sensitivity"]:
            if (r["family"], r["tier"], r["mode"]) == (family, tier, "raw"):
                paired.append({"hospital": s["hospital"], **r})
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Hospital": r["hospital"],
                    "Changed classifications": r["changed_decisions"],
                    "Mean absolute score shift": r["mean_abs_probability_change"],
                    "Δ balanced accuracy": r["deltas"]["balanced_accuracy"],
                    "Δ ROC-AUC": r["deltas"]["roc_auc"],
                    "Δ Brier": r["deltas"]["brier"],
                    "Δ log loss": r["deltas"]["log_loss"],
                    "Descriptive shift flag": r["flag"],
                }
                for r in paired
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    if paired:
        plot(
            charts.bars(
                [r["hospital"] for r in paired],
                [r["changed_decisions"] for r in paired],
                "Changed classifications",
            ),
            "Number of classifications changed by treating recorded cholesterol zeros "
            "as missing, by hospital.",
        )
    st.warning(
        "The meaning of recorded cholesterol zeros remains unresolved. "
        "Sensitivity analysis does not prove which interpretation is "
        "clinically correct. No score here is a medically validated "
        "disease risk."
    )


def sandbox(e: Evidence) -> None:
    from cardioscope.data.schema import FIELDS

    from .sandbox import LIMITS, PROFILE, expected_metadata, load_verified, validate_profile

    header(
        "Educational Prediction Sandbox",
        ("A fictional profile passed through the exact fitted early Logistic Regression pipeline."),
    )
    st.info(DISCLAIMER)
    st.caption(
        "Use synthetic values only. Inputs and outputs are held "
        "transiently in this local session; they are never written to the "
        "registry, files, logs or remote services."
    )
    model = None
    try:
        model = load_verified(e)
    except EvidenceError as exc:
        st.warning(str(exc))
        st.code(
            ".\\.venv\\Scripts\\python.exe -m cardioscope.dashboard.sandbox", language="powershell"
        )
    examples: dict[str, Record] = {
        "Fictional A": PROFILE,
        "Fictional B": {
            **PROFILE,
            "age": 64.0,
            "sex": "Male",
            "cp": "asymptomatic",
            "trestbps": 145.0,
            "chol": 280.0,
            "fbs": "TRUE",
        },
        "Fictional C · zero cholesterol": {
            **PROFILE,
            "age": 37.0,
            "chol": 0.0,
            "cp": "non-anginal",
        },
    }
    preset = st.selectbox("Synthetic example", list(examples), key="sandbox_preset")

    def reset_profile() -> None:
        # New widget IDs discard values still buffered in the browser form.
        st.session_state["sandbox_form_revision"] = form_revision + 1

    form_revision = st.session_state.get("sandbox_form_revision", 0)
    st.button("Reset fictional inputs", on_click=reset_profile)
    profile_key = f"fiction_{preset}_{form_revision}"
    left, right = st.columns(2)
    with left:
        st.subheader("Fictional input profile")
        with st.form(f"{profile_key}_form"):
            values: Record = {}
            for key, label in {
                "age": "Age (years)",
                "trestbps": "Resting blood pressure (mm Hg)",
                "chol": "Cholesterol (mg/dL)",
            }.items():
                low, high = LIMITS[key]
                values[key] = st.number_input(
                    label,
                    min_value=float(low),
                    max_value=float(high),
                    value=float(examples[preset][key]),
                    step=1.0,
                    format="%g",
                    key=f"{profile_key}_{key}",
                )
            for key, label in {
                "sex": "Recorded sex category",
                "cp": "Chest pain category",
                "fbs": "Fasting blood sugar >120 mg/dL",
            }.items():
                levels = FIELDS[key].levels
                values[key] = st.selectbox(
                    label,
                    levels,
                    index=levels.index(str(examples[preset][key])),
                    key=f"{profile_key}_{key}",
                )
            with st.container(key="contrast_submit"):
                submitted = st.form_submit_button(
                    "Validate & run fictional profile", disabled=model is None, type="primary"
                )
        st.caption(
            "Demo input bounds are interface constraints, not clinical "
            "reference ranges. This model preserves recorded cholesterol "
            "zeros."
        )
    with right:
        st.subheader("Experimental output")
        if submitted and model is not None:
            try:
                frame = validate_profile(values)
                score = float(model.predict_proba(frame)[0, 1])
                threshold = expected_metadata(e)["threshold"]
                st.metric("Experimental model score", f"{score:.3f}")
                st.write(
                    f"Predicted historical dataset class: "
                    f"**{int(score >= threshold)}** at threshold {threshold:.2f}."
                )
                st.caption(
                    "Experimental model score — not a clinically validated probability of disease."
                )
                st.caption("Output corresponds to the submitted profile. Submit again after edits.")
            except ValueError as exc:
                st.error(str(exc))
        else:
            st.write("Submit a fictional profile to inspect the pipeline output.")
        st.write(
            "Historical target: class 0 corresponds to recorded num = 0; "
            "class 1 combines num = 1–4. A classification describes the "
            "model's mapping of these historical labels."
        )
        st.write(
            "The pipeline imputes numeric values, scales them, encodes "
            "categories and applies its fitted logistic classifier. No "
            "training occurs here. The pooled model is not the separately "
            "fitted hospital-transfer models."
        )
        st.warning(
            "Changing population, missing measurements and historical "
            "collection practices can invalidate interpretation of a score. "
            "This demonstration offers no diagnosis or treatment "
            "recommendation."
        )


def registry(e: Evidence) -> None:
    from .registry import RegistryError, import_evidence, open_registry, verified_listing

    header("Experiment Registry", "A local, reproducible index of aggregate research results.")
    st.caption(
        "Timestamps record import time, not historical experiment "
        "execution. Reimporting identical evidence is idempotent. No "
        "individual records or prediction attempts are stored."
    )
    engine = None
    try:
        engine = open_registry()
        with st.container(key="contrast_import"):
            imported = st.button("Import verified research records", type="primary")
        if imported:
            import_evidence(engine, e)
        rows, missing = verified_listing(engine, e)
        total = len(rows)
        if imported:
            st.success(f"Registry contains {total} aggregate experiments.")
        if not rows:
            st.info("Registry is empty. Import the verified Phase 3 and Phase 4 results to begin.")
            return
        if missing:
            st.warning(f"{missing} verified records are missing. Reimport to restore them.")
        cols = st.columns(4)
        filters: dict[str, str] = {}
        for col, field in zip(cols, ["model", "tier", "evaluation", "hospital"], strict=True):
            choice = col.selectbox(
                field.capitalize(),
                ["All"] + sorted({r[field] for r in rows}),
                key=f"registry_{field}",
            )
            if choice is not None:
                filters[field] = choice
        rows = [
            row
            for row in rows
            if all(value == "All" or row[key] == value for key, value in filters.items())
        ]
        st.caption(f"Showing {len(rows)} of {total} verified experiments")
        if not rows:
            st.info("No experiments match these filters.")
            return
        st.dataframe(
            pd.DataFrame([{k: v for k, v in r.items() if k != "metadata"} for r in rows]),
            hide_index=True,
            width="stretch",
        )
        labels = {
            r["id"]: (
                f"{FAMILIES[r['model']]} · {r['tier']} · {r['hospital']} · "
                f"{r['metadata'].get('policy', 'pooled')} · "
                f"{r['metadata'].get('calibration', 'raw')} · {r['id'][:8]}"
            )
            for r in rows
        }
        chosen = st.selectbox("Experiment detail", list(labels), format_func=labels.__getitem__)
        st.json(next(r["metadata"] for r in rows if r["id"] == chosen), expanded=False)
        selected = st.multiselect(
            "Compare experiments",
            list(labels),
            default=list(labels)[:3],
            format_func=labels.__getitem__,
        )
        metric = st.selectbox("Comparison measure", list(METRICS), format_func=METRICS.__getitem__)
        comparison = [r for r in rows if r["id"] in selected]
        if comparison:
            plot(
                charts.bars(
                    [r["id"][:8] for r in comparison],
                    [r["metadata"]["metrics"].get(metric) for r in comparison],
                    METRICS[metric],
                    metric != "log_loss",
                    value_format=":.4%" if metric in PERCENT_METRICS else ":.6~f",
                    tickformat=".0%" if metric in PERCENT_METRICS else None,
                ),
                f"{METRICS[metric]} for the selected aggregate experiments.",
            )
            st.caption(
                "Chart labels are experiment ID prefixes; full identities appear "
                "above. Different evaluation populations are descriptive "
                "comparisons, not a shared benchmark."
            )
    except RegistryError as exc:
        st.error(str(exc))
        st.info(
            "Research pages remain available. Check CARDIOSCOPE_DATABASE_URL "
            "or use the default local SQLite database."
        )
    finally:
        if engine is not None:
            engine.dispose()


def document_preview(content: str, path: str) -> str:
    """Display checkout-relative Markdown destinations as repository references."""
    pattern = r"\[([^\]\n]+)\]\(((?:\.\.?/)*[\w./-]+\.(?:md|json|png))(?:#[\w-]+)?\)"

    def reference(match: re.Match[str]) -> str:
        target = (ROOT / PurePosixPath(path).parent / match[2]).resolve()
        if not target.is_relative_to(ROOT):
            return match[1]
        return f"{match[1]} (`{target.relative_to(ROOT).as_posix()}`)"

    return re.sub(pattern, reference, content)


def methodology(e: Evidence) -> None:
    header(
        "Methodology & Reproducibility",
        ("Trace each result to its data release, frozen protocol and aggregate artifacts."),
    )
    st.info(DISCLAIMER)
    st.subheader("Research question")
    st.write(
        "How well do heart-disease classification models generalize "
        "across hospital populations, and how reliable are their "
        "predicted probabilities?"
    )
    for title, text in {
        "Outcome & information settings": e.protocol["endpoint"]
        + " Early: "
        + ", ".join(e.protocol["tiers"]["early"])
        + ". Extended: "
        + ", ".join(e.protocol["tiers"]["extended"])
        + ".",
        "Preprocessing & leakage prevention": e.protocol["fit"]
        + " "
        + e.protocol["duplicate_policy"],
        "Training & pooled selection": e.cv["selection"]["metadata"]["selection"]
        + ". "
        + e.cv["selection"]["metadata"]["split_rule"]
        + ". A shared pooled holdout follows training-only selection.",
        "Hospital evaluation & calibration": e.protocol["design"]
        + " "
        + e.protocol["inner"]
        + " "
        + e.protocol["calibration"],
        "Uncertainty & sensitivity": e.protocol["uncertainty"]["method"]
        + " "
        + e.protocol["sensitivity"],
        "Observed results versus assumptions": (
            "Reported metrics describe retrospective samples. "
            "Cholesterol-zero meanings and duplicate identities remain "
            "unresolved. Neither imputation nor calibration establishes "
            "clinical validity."
        ),
        "Future external validation": (
            "A suitable untouched external cohort must have a compatible "
            "outcome, predictors available at the intended decision time, "
            "documented measurement timing and population characteristics, "
            "provenance and appropriate permissions. Pre-register evaluation "
            "procedures before observing results. No external cohort or "
            "prospective results are available here."
        ),
    }.items():
        with st.expander(title):
            st.write(text)
    st.subheader("Source documents")
    st.caption("Repository paths refer to files in the checkout. Downloads preserve the source.")
    docs = {
        "Data provenance & licensing": "docs/provenance.md",
        "Leakage-safe evaluation": "docs/evaluation.md",
        "Model selection": "docs/model-selection.md",
        "Phase 4 findings": "reports/phase4/reliability-findings.md",
        "Dashboard & reproduction": "docs/dashboard.md",
    }
    for label, path in docs.items():
        with st.expander(label):
            document = ROOT / path
            if document.exists():
                content = document.read_text(encoding="utf-8")
                st.markdown(document_preview(content, path))
                st.download_button(
                    "Download source Markdown", content, file_name=document.name, key=path
                )
            else:
                st.info("Document unavailable in this checkout.")
    with st.expander("Verified release identifiers"):
        st.json(
            {
                "dataset_sha256": e.audit["sha256"],
                "phase3_selection_sha256": e.pooled["selection_sha256"],
                "phase4_scientific_sha256": e.transfer["scientific_sha256"],
                "protocol_commit": e.transfer["protocol_commit"],
            }
        )


def run(page: str) -> None:
    try:
        evidence = load_evidence()
    except EvidenceError as exc:
        header(
            "Research evidence unavailable",
            "Restore the tracked, verified artifacts before displaying results.",
        )
        st.error(str(exc))
        return
    views = {
        "overview": overview,
        "dataset": dataset,
        "laboratory": laboratory,
        "reliability": reliability,
        "calibration": calibration,
        "sandbox": sandbox,
        "registry": registry,
        "methodology": methodology,
    }
    views[page](evidence)
    st.divider()
    st.caption("CardioScope ML · Verified retrospective evidence · " + DISCLAIMER)
