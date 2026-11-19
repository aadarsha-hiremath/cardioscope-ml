"""Streamlit account sessions, preferences, and saved report views."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from .auth import LOGOUT, WELCOME, CookiePolicy, browser_token
from .evidence import FAMILIES
from .examples import EXAMPLES, REPORT_DISCLAIMER
from .registry import RegistryError, database_url
from .workspace import Account, PublicReport, SavedReport, Settings, Workspace
from .workspace_schema import WorkspaceError, open_workspace

if TYPE_CHECKING:
    from sqlalchemy.engine import URL

ACCOUNT = "_workspace_account"
FLASH = "_workspace_flash"


@st.cache_resource(show_spinner=False)
def _workspace(_url: URL, database_key: str) -> Workspace:
    return Workspace(open_workspace(_url))


def store() -> Workspace:
    try:
        url = database_url()
    except RegistryError as exc:
        raise WorkspaceError("Personal workspace storage is not configured correctly.") from exc
    key = hashlib.sha256(url.render_as_string(hide_password=False).encode()).hexdigest()
    return _workspace(url, key)


def token() -> str:
    value = browser_token(st.context.cookies, CookiePolicy.configured())
    if value is None:
        raise WorkspaceError("Sign in to access your personal workspace.")
    return value


def refresh_account() -> None:
    """Guard every rerun independently of browser navigation and widget state."""
    try:
        grant = token()
        st.session_state[ACCOUNT] = store().account(grant)
    except WorkspaceError:
        st.session_state.clear()
        st.info("Sign in to open the research workspace.")
        st.html(
            f'<p><a href="{WELCOME}">Open sign-in</a></p>'
            f'<script>window.location.replace("{WELCOME}");</script>',
            unsafe_allow_javascript=True,
        )
        st.stop()


@st.fragment(run_every=10.0)
def session_watch() -> None:
    try:
        grant = token()
        store().account(grant)
    except WorkspaceError:
        st.rerun(scope="app")


def sidebar_logout() -> None:
    st.html(
        f"""<style>
        .cardioscope-logout button {{
          width: 100%; min-height: 38px; background: #18243a; color: #f1f5f9;
          border: 1px solid #52637b; border-radius: 8px; font: inherit; cursor: pointer;
        }}
        .cardioscope-logout button:hover {{ border-color: #38bdf8; }}
        .cardioscope-logout button:focus-visible {{ outline: 2px solid #38bdf8; }}
        </style>
        <form class="cardioscope-logout" action="{LOGOUT}" method="post">
          <button type="submit">Logout</button>
        </form>"""
    )


def active_account() -> Account | None:
    account = st.session_state.get(ACCOUNT)
    return account if isinstance(account, Account) else None


def account_settings() -> Settings:
    account = active_account()
    return account.settings if account else Settings()


def widget_key(base: str) -> str:
    account = active_account()
    return base if account is None else f"personal_{account.id}_{base}"


def _notice() -> None:
    message = st.session_state.pop(FLASH, None)
    if message:
        st.success(message)


def _sign_in_required() -> bool:
    if active_account() is not None:
        return False
    st.info("Sign in to open your personal workspace.")
    return True


def account_page() -> None:
    st.title("Account")
    account = active_account()
    if account is not None:
        st.text(f"Signed in as {account.username}")
        st.caption("Use Logout in the sidebar to end your session on this device.")


def settings_page() -> None:
    from cardioscope.data.schema import HOSPITALS

    st.title("Settings")
    st.caption("Choose how the research dashboard opens and displays comparisons.")
    if _sign_in_required():
        return
    _notice()
    current = account_settings()
    models = list(FAMILIES)
    tiers = ["early", "extended"]
    with st.form("personal_settings"):
        show_grid = st.checkbox("Show chart grid lines", value=current.show_grid)
        advanced = st.checkbox(
            "Open advanced model configuration by default", value=current.advanced_config
        )
        st.subheader("Default research filters")
        model = st.selectbox(
            "Default model family",
            models,
            index=models.index(current.model),
            format_func=FAMILIES.__getitem__,
        )
        tier = st.selectbox(
            "Default feature tier",
            tiers,
            index=tiers.index(current.tier),
            format_func=str.capitalize,
        )
        hospital = st.selectbox(
            "Default held-out hospital", HOSPITALS, index=HOSPITALS.index(current.hospital)
        )
        if st.form_submit_button("Save settings"):
            try:
                store().save_settings(
                    token(),
                    {
                        "show_grid": show_grid,
                        "advanced_config": advanced,
                        "model": model,
                        "tier": tier,
                        "hospital": hospital,
                    },
                )
                st.session_state[FLASH] = "Settings saved."
                st.rerun()
            except WorkspaceError as exc:
                st.error(str(exc))


def _timestamp(value: int) -> str:
    return datetime.fromtimestamp(value, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _public_fields(report: PublicReport) -> dict[str, str | int]:
    return {
        "Model family": report.model,
        "Feature tier": report.feature_tier.capitalize(),
        "scikit-learn": report.version,
        "Phase 3 prediction digest": report.artifact_digest,
        "Experimental model score": f"{report.score:.3f}",
        "Predicted historical class": report.historical_class,
    }


def _report_title(report: SavedReport) -> str:
    if report.title == report.example == report.profile_label and report.example in EXAMPLES:
        return "Earlier example report"
    return report.title


def _profile_view(db: Workspace, grant: str) -> None:
    st.subheader("Saved Profiles")
    rows = db.profiles(grant)
    if not rows:
        st.info("You have no saved profiles.")
    if rows:
        st.caption("Previously saved profiles are available to load into a new prediction.")
        labels = {p.id: p.name for p in rows}
        choice = st.selectbox(
            "Profile name", list(labels), index=None, format_func=labels.__getitem__
        )
        left, right = st.columns(2)
        if left.button("Load into Prediction Lab", disabled=choice is None):
            st.session_state[widget_key("prediction_profile_to_load")] = choice
            st.switch_page("views/sandbox.py")
        if right.button("Delete saved profile", disabled=choice is None):
            assert choice is not None
            db.delete_profile(grant, choice)
            st.session_state[FLASH] = (
                "Saved profile removed. Its existing reports remain in history."
            )
            st.rerun()
    st.page_link("views/sandbox.py", label="Open Prediction Lab")


def _history_view(db: Workspace, grant: str) -> None:
    st.subheader("Report History")
    rows = db.history(grant)
    if not rows:
        st.info("You have no saved reports.")
        return
    st.caption(f"{len(rows)} saved {'report' if len(rows) == 1 else 'reports'}")
    titles = {r.id: f"{_report_title(r)} · {_timestamp(r.created_at)} · {r.id[:8]}" for r in rows}
    chosen = st.selectbox(
        "Open saved report", list(titles), index=None, format_func=titles.__getitem__
    )
    if chosen:
        report = db.report(grant, chosen)
        with st.container(border=True):
            st.text(_report_title(report))
            if report.profile_label and report.profile_label not in EXAMPLES:
                st.text(f"Saved profile: {report.profile_label}")
            st.caption(f"Created {_timestamp(report.created_at)}")
            st.table(pd.DataFrame([_public_fields(report)]).T.rename(columns={0: "Result"}))
            st.caption(f"Historical class threshold {report.threshold:.2f}")
            st.info(REPORT_DISCLAIMER)
            with st.form(f"personal_rename_{chosen}"):
                new_title = st.text_input(
                    "Report title",
                    value=_report_title(report),
                    max_chars=64,
                    help="Use a non-identifying title without patient information.",
                )
                if st.form_submit_button("Rename report"):
                    db.rename_report(grant, chosen, new_title)
                    st.session_state[FLASH] = "Report renamed."
                    st.rerun()
            with st.expander("Sharing and report deletion", expanded=report.shared):
                st.write(
                    "Other signed-in users can view the shared summary. "
                    "Only the result fields shown in the sharing preview will appear."
                )
                st.caption("Sharing preview")
                st.table(
                    pd.DataFrame([_public_fields(report)]).T.rename(columns={0: "Shared value"})
                )
                st.caption(REPORT_DISCLAIMER)
                if report.shared:
                    st.success("This report is shared.")
                    if st.button("Revoke sharing", key=f"personal_revoke_{chosen}"):
                        db.set_sharing(grant, chosen, shared=False)
                        st.session_state[FLASH] = "Sharing revoked."
                        st.rerun()
                else:
                    confirm_share = st.checkbox(
                        "I approve sharing this summary",
                        key=f"personal_share_confirm_{chosen}",
                    )
                    if st.button(
                        "Share Report",
                        disabled=not confirm_share,
                        key=f"personal_share_{chosen}",
                    ):
                        db.set_sharing(grant, chosen, shared=True)
                        st.session_state[FLASH] = "Report shared."
                        st.rerun()
                delete = st.checkbox(
                    "Confirm deletion of this report", key=f"personal_delete_{chosen}"
                )
                if st.button("Delete report", disabled=not delete, key=f"personal_drop_{chosen}"):
                    db.delete_report(grant, chosen)
                    st.session_state[FLASH] = "Report deleted."
                    st.rerun()
    with st.expander("Clear report history"):
        st.warning("This removes every report in your account and revokes their sharing.")
        clear = st.checkbox("Confirm clearing all my reports")
        if st.button("Clear my report history", disabled=not clear):
            db.clear_history(grant)
            st.session_state[FLASH] = "Your report history was cleared."
            st.rerun()


def reports_page() -> None:
    st.title("My Reports")
    st.caption("Manage your saved profiles and private prediction reports.")
    if _sign_in_required():
        return
    _notice()
    try:
        db, grant = store(), token()
        _profile_view(db, grant)
        st.divider()
        _history_view(db, grant)
    except WorkspaceError as exc:
        st.error(str(exc))


def shared_page() -> None:
    st.title("Shared Reports")
    st.info(REPORT_DISCLAIMER)
    st.caption("Only reports explicitly shared by their owners appear here.")
    try:
        reports = store().shared_reports(token())
    except WorkspaceError as exc:
        st.error(str(exc))
        return
    if not reports:
        st.info("No reports have been shared yet.")
        return
    st.dataframe(
        pd.DataFrame([_public_fields(r) for r in reports]), hide_index=True, width="stretch"
    )
