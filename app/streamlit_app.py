"""Local research dashboard; launch from the repository root."""

from pathlib import Path

import streamlit as st

from cardioscope.dashboard.workspace_ui import active_account, refresh_account

st.set_page_config(page_title="CardioScope ML", page_icon=":material/monitoring:", layout="wide")
refresh_account()
st.logo(str(Path(__file__).parent / "assets/wordmark.svg"), size="large")
# Only documented, application-owned container classes are styled.
st.html("""<style>
[class*="st-key-metric_card_"] { background-color: #18243A; }
.st-key-contrast_submit button:not(:disabled),
.st-key-contrast_import button:not(:disabled) { color: #0B1120; }
</style>""")

groups = {
    "RESEARCH": [
        ("overview", "Research Overview", "monitoring"),
        ("dataset", "Dataset Explorer", "dataset"),
        ("laboratory", "Model Laboratory", "science"),
        ("reliability", "Hospital Reliability", "hub"),
        ("calibration", "Calibration & Sensitivity", "tune"),
    ],
    "WORKSPACE": [
        ("sandbox", "Prediction Sandbox", "experiment"),
        ("registry", "Experiment Registry", "database"),
    ],
    "PERSONAL WORKSPACE": [
        ("account", "Account", "account_circle"),
        ("settings", "Settings", "settings"),
        ("my_reports", "My Reports", "folder"),
        ("shared_reports", "Shared Fictional Reports", "share"),
    ],
    "DOCUMENTATION": [("methodology", "Methodology & Reproducibility", "menu_book")],
}
navigation = st.navigation(
    {
        group: [
            st.Page(
                f"views/{name}.py",
                title=title,
                icon=f":material/{icon}:",
                url_path=name.replace("_", "-"),
                default=name == "overview",
            )
            for name, title, icon in pages
        ]
        for group, pages in groups.items()
    }
)
with st.sidebar:
    account = active_account()
    if account is not None:
        st.text(f"Signed in as {account.username}")
    st.caption("PHASE 6.1 · LOCAL RESEARCH")
    st.caption("Aggregate evidence · CPU inference")
navigation.run()
