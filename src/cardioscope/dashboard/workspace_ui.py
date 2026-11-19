"""Streamlit account sessions, preferences, and fictional report views."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from .evidence import FAMILIES
from .examples import EXAMPLES, REPORT_DISCLAIMER
from .registry import RegistryError, database_url
from .workspace import Account, PublicReport, Settings, Workspace
from .workspace_schema import WorkspaceError, open_workspace

if TYPE_CHECKING:
    from sqlalchemy.engine import URL

TOKEN = "_workspace_token"
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
    value = st.session_state.get(TOKEN)
    if not isinstance(value, str):
        raise WorkspaceError("Sign in to access your personal workspace.")
    return value


def _clear_account_state() -> None:
    for key in list(st.session_state):
        if isinstance(key, str) and key.startswith(("_workspace_", "personal_", "fiction_")):
            del st.session_state[key]
    for key in ("sandbox_preset", "sandbox_form_revision"):
        st.session_state.pop(key, None)


def refresh_account() -> None:
    """Refresh the identity on every rerun. A widget value never establishes identity."""
    if TOKEN not in st.session_state:
        return
    try:
        st.session_state[ACCOUNT] = store().account(token())
    except WorkspaceError:
        _clear_account_state()
        st.session_state[FLASH] = "Your workspace session ended. Sign in again to continue."


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
    st.page_link("views/account.py", label="Open account page")
    return True


def account_page() -> None:
    st.title("Account")
    st.caption("Local accounts keep settings and fictional workspaces separate.")
    _notice()
    account = active_account()
    if account:
        st.text(f"Signed in as {account.username}")
        st.caption("Refreshing or restarting the app may require you to sign in again.")
        if st.button("Sign out", key="personal_signout"):
            try:
                store().logout(token())
            except WorkspaceError:
                _clear_account_state()
                st.session_state[FLASH] = "Signed out here. Server revocation was unavailable."
            else:
                _clear_account_state()
                st.session_state[FLASH] = "Signed out."
            st.rerun()
        return
    login_tab, register_tab = st.tabs(["Sign in", "Register"])
    with login_tab, st.form("personal_sign_in", clear_on_submit=True):
        username = st.text_input("Username", key="personal_login_name", max_chars=32)
        password = st.text_input(
            "Password", type="password", key="personal_login_pw", max_chars=128
        )
        if st.form_submit_button("Sign in"):
            try:
                grant = store().login(username, password)
                _clear_account_state_after_submit(grant)
            except WorkspaceError as exc:
                st.error(str(exc))
    with register_tab, st.form("personal_register", clear_on_submit=True):
        new_name = st.text_input("New username", max_chars=32)
        new_password = st.text_input("New password", type="password", max_chars=128)
        confirmation = st.text_input("Confirm password", type="password", max_chars=128)
        st.caption("Use 12–128 characters and a password unique to this local workspace.")
        if st.form_submit_button("Create account"):
            if new_password != confirmation:
                st.error("The passwords do not match.")
            else:
                try:
                    store().register(new_name, new_password)
                    st.success("Account created. Sign in to start your empty personal workspace.")
                except WorkspaceError as exc:
                    st.error(str(exc))


def _clear_account_state_after_submit(grant: str) -> None:
    _clear_account_state()
    st.session_state[TOKEN] = grant
    st.session_state[FLASH] = "Signed in."
    st.rerun()


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
        "Built-in example": report.example,
        "Model": report.model,
        "Tier": report.feature_tier.capitalize(),
        "scikit-learn": report.version,
        "Phase 3 prediction digest": report.artifact_digest,
        "Experimental score": f"{report.score:.3f}",
        "Historical class": report.historical_class,
    }


def _profile_view(db: Workspace, grant: str) -> None:
    st.subheader("Saved fictional profiles")
    rows = db.profiles(grant)
    if not rows:
        st.info("You have no saved fictional profiles.")
    with st.form("personal_profile", clear_on_submit=True):
        name = st.text_input("Private profile name", max_chars=64)
        example = st.selectbox("Built-in example to save", list(EXAMPLES), index=None)
        st.caption("Choose a built-in example. Edited inputs cannot be saved.")
        if st.form_submit_button("Save fictional profile"):
            if example is None:
                st.error("Select a built-in fictional example.")
            else:
                db.create_profile(grant, name, example)
                st.session_state[FLASH] = "Fictional profile saved."
                st.rerun()
    if rows:
        labels = {p.id: f"{p.name} · {p.example}" for p in rows}
        choice = st.selectbox(
            "Saved profile to remove", list(labels), index=None, format_func=labels.__getitem__
        )
        if st.button("Delete saved profile", disabled=choice is None):
            assert choice is not None
            db.delete_profile(grant, choice)
            st.session_state[FLASH] = (
                "Saved profile removed. Its existing reports remain in history."
            )
            st.rerun()
    st.page_link("views/sandbox.py", label="Open Prediction Sandbox")


def _history_view(db: Workspace, grant: str) -> None:
    st.subheader("Private report history")
    rows = db.history(grant)
    if not rows:
        st.info("You have no saved educational reports.")
        return
    st.caption(f"{len(rows)} saved reports")
    titles = {r.id: f"{r.title} · {_timestamp(r.created_at)} · {r.id[:8]}" for r in rows}
    chosen = st.selectbox(
        "Open saved report", list(titles), index=None, format_func=titles.__getitem__
    )
    if chosen:
        report = db.report(grant, chosen)
        with st.container(border=True):
            st.text(report.title)
            st.text(f"Saved fictional profile: {report.profile_label}")
            st.caption(f"Created {_timestamp(report.created_at)}")
            st.table(pd.DataFrame([_public_fields(report)]).T.rename(columns={0: "Result"}))
            st.caption(f"Historical class threshold {report.threshold:.2f}")
            st.info(report.disclaimer)
            with st.form(f"personal_rename_{chosen}"):
                new_title = st.text_input("Private report title", value=report.title, max_chars=64)
                if st.form_submit_button("Rename report"):
                    db.rename_report(grant, chosen, new_title)
                    st.session_state[FLASH] = "Report renamed."
                    st.rerun()
            with st.expander("Sharing and report deletion", expanded=report.shared):
                st.write(
                    "Shared reports can be seen by anyone who can open this dashboard. "
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
                        "I want to share this fictional result",
                        key=f"personal_share_confirm_{chosen}",
                    )
                    if st.button(
                        "Share fictional report",
                        disabled=not confirm_share,
                        key=f"personal_share_{chosen}",
                    ):
                        db.set_sharing(grant, chosen, shared=True)
                        st.session_state[FLASH] = "Fictional report shared."
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
    st.caption("Your saved fictional profiles and educational reports.")
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
    st.title("Shared Fictional Reports")
    st.info(REPORT_DISCLAIMER)
    st.caption("Only reports explicitly shared by their owners appear here.")
    try:
        reports = store().shared_reports()
    except WorkspaceError as exc:
        st.error(str(exc))
        return
    if not reports:
        st.info("No fictional reports have been shared yet.")
        return
    st.dataframe(
        pd.DataFrame([_public_fields(r) for r in reports]), hide_index=True, width="stretch"
    )
