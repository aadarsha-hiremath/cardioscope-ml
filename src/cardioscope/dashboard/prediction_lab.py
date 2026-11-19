"""Manual educational inference and explicit saving of output-only reports."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cardioscope.data.schema import FIELDS

from . import sandbox, workspace_ui
from .evidence import Evidence, EvidenceError, Record
from .examples import REPORT_DISCLAIMER
from .workspace_schema import WorkspaceError
from .workspace_ui import widget_key

CONSENT = "prediction_example_only"
RESULT = "prediction_current_result"
SAVED = "prediction_saved"


def _clear_result() -> None:
    st.session_state.pop(widget_key(RESULT), None)
    st.session_state.pop(widget_key(SAVED), None)


def _input_key(field: str) -> str:
    return widget_key(f"prediction_{field}")


def _reset() -> None:
    _clear_result()
    for field in sandbox.INPUT_LABELS:
        st.session_state[_input_key(field)] = None
    st.session_state[widget_key(CONSENT)] = False


def _category(value: str) -> str:
    return {"TRUE": "Yes", "FALSE": "No"}.get(value, value)


def _inputs() -> Record:
    values: Record = {}
    left, right = st.columns(2)
    with left:
        for field, (low, high) in sandbox.LIMITS.items():
            values[field] = st.number_input(
                sandbox.INPUT_LABELS[field],
                min_value=low,
                max_value=high,
                value=None,
                step=1,
                placeholder="Not provided",
                key=_input_key(field),
                on_change=_clear_result,
            )
    with right:
        for field in ("sex", "cp", "fbs"):
            values[field] = st.selectbox(
                sandbox.INPUT_LABELS[field],
                FIELDS[field].levels,
                index=None,
                placeholder="Not provided",
                format_func=_category,
                key=_input_key(field),
                on_change=_clear_result,
            )
    return values


def _result_view(result: sandbox.Prediction, values: Record) -> None:
    st.subheader("Prediction Result")
    st.info(REPORT_DISCLAIMER)
    score, classification = st.columns(2)
    score.metric("Experimental model score", f"{result.score:.3f}")
    classification.metric("Predicted historical class", str(result.historical_class))
    st.caption(
        f"Historical classification threshold {result.threshold:.2f}. "
        "The class uses the score before rounding. "
        "Class 0 represents dataset label 0. Class 1 groups dataset labels 1–4."
    )
    st.write(f"**Model family** {result.model} · **Feature tier** {result.feature_tier.title()}")
    st.caption(f"scikit-learn {result.version} · Verified Phase 3 pooled holdout replay")
    with st.expander("Model provenance"):
        st.caption("Phase 3 prediction digest")
        st.code(result.artifact_digest, language="text")
    st.subheader("Input Summary")
    st.table(
        pd.DataFrame(
            [
                {
                    "Measurement": label,
                    "Entered value": "Not provided"
                    if values[key] is None
                    else _category(str(values[key])),
                }
                for key, label in sandbox.INPUT_LABELS.items()
            ]
        )
    )
    if st.session_state.get(widget_key(SAVED)):
        st.success("Report saved to My Reports.")
        st.page_link("views/my_reports.py", label="Open My Reports")
        return
    st.caption(
        "Saved reports include the result and model provenance. Measurements stay in this view."
    )
    with st.form(f"prediction_save_{result.id}"):
        title = st.text_input(
            "Report title",
            max_chars=64,
            placeholder="Example run",
            help="Use a non-identifying title without patient information.",
        )
        with st.container(key="contrast_save"):
            save = st.form_submit_button("Save Report", type="primary")
        if save:
            try:
                workspace_ui.store().save_report(workspace_ui.token(), result, title)
                st.session_state[widget_key(SAVED)] = True
                st.rerun()
            except WorkspaceError as exc:
                st.error(str(exc))


def render(e: Evidence) -> None:
    if _input_key("age") not in st.session_state:
        _clear_result()
    profile = st.session_state.pop(widget_key("prediction_profile_to_load"), None)
    if profile is not None:
        _reset()
        try:
            values = workspace_ui.store().profile_inputs(workspace_ui.token(), profile)
            for field, value in values.items():
                st.session_state[_input_key(field)] = (
                    int(value) if field in sandbox.LIMITS else value
                )
        except WorkspaceError as exc:
            st.error(str(exc))
    st.warning("Use invented measurements only. Do not enter real patient data.")
    st.info(REPORT_DISCLAIMER)
    st.caption(
        "Leave unknown fields blank. The verified pipeline handles missing values. "
        "Enter at least one measurement. Entered cholesterol zeros are kept as entered."
    )
    model = None
    try:
        model = sandbox.load_verified(e)
    except EvidenceError as exc:
        st.warning(str(exc))
        st.code(
            ".\\.venv\\Scripts\\python.exe -m cardioscope.dashboard.sandbox", language="powershell"
        )
    with st.container(border=True):
        st.subheader("Example Measurements")
        values = _inputs()
        confirmed = st.checkbox(
            "I am using invented measurements only",
            key=widget_key(CONSENT),
            on_change=_clear_result,
        )
        run, reset = st.columns(2)
        with run.container(key="contrast_submit"):
            submitted = st.button(
                "Run Prediction", type="primary", disabled=model is None or not confirmed
            )
        reset.button("New Prediction", on_click=_reset)
    if submitted and model is not None and confirmed:
        _clear_result()
        try:
            st.session_state[widget_key(RESULT)] = sandbox.predict(e, model, values)
        except ValueError as exc:
            st.error(str(exc))
    result = st.session_state.get(widget_key(RESULT))
    if isinstance(result, sandbox.Prediction):
        _result_view(result, values)
