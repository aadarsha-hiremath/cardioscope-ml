"""Local research dashboard; launch from the repository root."""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="CardioScope ML", page_icon=":material/monitoring:", layout="wide")
st.logo(str(Path(__file__).parent / "assets/wordmark.svg"), size="large")
# Only documented, application-owned container classes are styled.
st.html("""<style>
[class*="st-key-metric_card_"] { background-color: #18243A; }
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
    "DOCUMENTATION": [("methodology", "Methodology & Reproducibility", "menu_book")],
}
navigation = st.navigation(
    {
        group: [
            st.Page(
                f"views/{name}.py",
                title=title,
                icon=f":material/{icon}:",
                url_path=name,
                default=name == "overview",
            )
            for name, title, icon in pages
        ]
        for group, pages in groups.items()
    }
)
with st.sidebar:
    st.caption("PHASE 5 · LOCAL RESEARCH")
    st.caption("Aggregate evidence · CPU inference")
navigation.run()
