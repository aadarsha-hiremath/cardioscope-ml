"""Local authentication and SQLite isolation contracts for personal workspaces."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest
from sqlalchemy import inspect, select
from sqlalchemy.engine import URL
from streamlit.testing.v1 import AppTest

from cardioscope.dashboard import registry, workspace
from cardioscope.dashboard.evidence import ROOT, load_evidence
from cardioscope.dashboard.examples import EXAMPLES, REPORT_DISCLAIMER
from cardioscope.dashboard.workspace import AUTH_ERROR, Settings, Workspace
from cardioscope.dashboard.workspace_schema import WorkspaceError, open_workspace, sessions, users

PASSWORD = "test-only fixture phrase 2026"
A = "Fictional A"
B = "Fictional B"


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "workspace.db"
    monkeypatch.setenv("CARDIOSCOPE_DATABASE_URL", f"sqlite:///{path}")
    engine = open_workspace()
    yield Workspace(engine), path
    engine.dispose()


def sign_in(db: Workspace, username: str) -> str:
    db.register(username, PASSWORD)
    return db.login(username, PASSWORD)


class FictionalModel:
    def predict_proba(self, frame):
        score = 0.05 if frame["age"].iloc[0] == 48 else 0.91
        return np.array([[1 - score, score]])


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.setattr(workspace, "load_verified", lambda _: FictionalModel())
    return load_evidence()


def test_unique_username_and_argon2id_storage(db):
    store, path = db
    token = sign_in(store, "Case_Test")
    assert store.account(token).username == "case_test"
    with pytest.raises(WorkspaceError, match="Unable to register"):
        store.register("case_TEST", PASSWORD)
    sign_in(store, "another_user")
    with store.engine.connect() as conn:
        hashes = conn.execute(select(users.c.password_hash)).scalars().all()
        assert len(hashes) == len(set(hashes)) == 2
        assert all(value.startswith("$argon2id$") for value in hashes)
        assert conn.execute(select(sessions.c.token_hash)).scalars().first() != token
    assert PASSWORD.encode() not in path.read_bytes()
    assert token.encode() not in path.read_bytes()
    with pytest.raises(WorkspaceError, match="12–128"):
        store.register("valid_name", "short")
    with pytest.raises(WorkspaceError, match="starting with a letter"):
        store.register("invalid username", PASSWORD)


def test_login_errors_rate_limit_and_logout(db, monkeypatch):
    store, _ = db
    token = sign_in(store, "known_user")
    monkeypatch.setattr(workspace, "ACCOUNT_LIMIT", 2)
    for account in ("known_user", "unknown_user"):
        with pytest.raises(WorkspaceError) as exc:
            store.login(account, "wrong passphrase")
        assert str(exc.value) == AUTH_ERROR
    with pytest.raises(WorkspaceError) as exc:
        store.login("known_user", PASSWORD)
    assert str(exc.value) == AUTH_ERROR
    store.logout(token)
    with pytest.raises(WorkspaceError, match="Sign in"):
        store.account(token)
    with pytest.raises(WorkspaceError, match="Sign in"):
        store.clear_history("no_valid_session")


def test_session_expiry_enforced_on_reads_and_writes(db):
    store, _ = db
    token = sign_in(store, "expired_user")
    with store.engine.begin() as conn:
        conn.execute(sessions.update().values(expires_at=int(time.time()) - 1))
    with pytest.raises(WorkspaceError, match="Sign in"):
        store.history(token)
    with pytest.raises(WorkspaceError, match="Sign in"):
        store.save_settings(token, asdict(Settings()))


def test_settings_persist_and_remain_account_specific(db):
    store, path = db
    a = sign_in(store, "owner_alpha")
    b = sign_in(store, "owner_beta")
    choices = Settings(show_grid=False, advanced_config=True, model="forest", tier="extended")
    store.save_settings(a, asdict(choices))
    assert store.account(b).settings == Settings()
    with pytest.raises(WorkspaceError, match="supported settings"):
        store.save_settings(a, {**asdict(choices), "tier": "unsupported"})
    with pytest.raises(WorkspaceError, match="supported settings"):
        store.save_settings(a, {**asdict(choices), "show_grid": 0})
    store.logout(a)
    reopened = reopen_workspace(path)
    try:
        assert reopened.account(reopened.login("owner_alpha", PASSWORD)).settings == choices
        assert reopened.account(reopened.login("owner_beta", PASSWORD)).settings == Settings()
    finally:
        reopened.engine.dispose()


def reopen_workspace(path):
    """Open a new pool to simulate a process restart against the same file."""
    return Workspace(open_workspace(URL.create("sqlite", database=str(path))))


def test_profiles_and_report_storage_reject_edited_values(db, evidence):
    store, _ = db
    token = sign_in(store, "owner_alpha")
    assert store.profiles(token) == store.history(token) == []
    profile = store.create_profile(token, "Demo reference", B)
    with pytest.raises(WorkspaceError, match="distinct"):
        store.create_profile(token, "DEMO REFERENCE", A)
    with pytest.raises(WorkspaceError, match="built-in"):
        store.create_profile(token, "Another reference", "user-submitted inputs")
    with pytest.raises(WorkspaceError, match="Edited values stay transient"):
        store.save_report(token, evidence, B, {**EXAMPLES[B], "age": 65.0}, profile.id)
    with pytest.raises(ValueError):
        store.save_report(token, evidence, B, {**EXAMPLES[B], "age": 5.0}, profile.id)
    assert not store.history(token)
    with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("must never fit")):
        saved = store.save_report(token, evidence, B, EXAMPLES[B], profile.id)
    assert saved.profile_label == "Demo reference"
    assert saved.historical_class == 1
    assert saved.owner_id == store.account(token).id
    assert saved.feature_tier == "early"
    assert (
        saved.artifact_digest == workspace.expected_metadata(evidence)["holdout_probability_sha256"]
    )
    assert saved.disclaimer == REPORT_DISCLAIMER
    assert not saved.shared
    store.delete_profile(token, profile.id)
    assert store.profiles(token) == []
    assert store.report(token, saved.id) == saved


def test_owner_actions_and_shared_projection_across_logins(db, evidence):
    store, path = db
    a = sign_in(store, "owner_alpha")
    private_label = "Private label example@sample.invalid"
    profile = store.create_profile(a, private_label, A)
    one = store.save_report(a, evidence, A, EXAMPLES[A], profile.id)
    hidden = store.save_report(a, evidence, B, EXAMPLES[B])
    assert not store.shared_reports()
    store.set_sharing(a, one.id, shared=True)
    store.logout(a)
    b = sign_in(store, "owner_beta")
    assert store.account(b).settings == Settings()
    assert store.history(b) == store.profiles(b) == []
    for denied in (
        lambda: store.report(b, one.id),
        lambda: store.rename_report(b, one.id, "Changed"),
        lambda: store.delete_report(b, one.id),
        lambda: store.set_sharing(b, one.id, shared=False),
        lambda: store.set_sharing(b, hidden.id, shared=True),
        lambda: store.save_report(b, evidence, A, EXAMPLES[A], profile.id),
    ):
        with pytest.raises(WorkspaceError, match="unavailable"):
            denied()
    store.delete_profile(b, profile.id)
    (public,) = store.shared_reports()
    assert public.example == A and public.score == 0.05
    assert set(asdict(public)) == {
        "example",
        "model",
        "feature_tier",
        "version",
        "artifact_digest",
        "score",
        "historical_class",
        "disclaimer",
    }
    projection = repr(public)
    assert not any(
        value in projection
        for value in (
            one.owner_id,
            one.id,
            private_label,
            "owner_alpha",
            "owner_beta",
        )
    )
    beta_report = store.save_report(b, evidence, B, EXAMPLES[B])
    store.logout(b)
    reopened = reopen_workspace(path)
    try:
        a_again = reopened.login("owner_alpha", PASSWORD)
        b_again = reopened.login("owner_beta", PASSWORD)
        assert reopened.profiles(a_again) == [profile]
        assert {r.id for r in reopened.history(a_again)} == {one.id, hidden.id}
        assert reopened.report(b_again, beta_report.id) == beta_report
        reopened.set_sharing(a_again, one.id, shared=False)
        assert not reopened.shared_reports()
        reopened.rename_report(a_again, one.id, "New private title")
        assert reopened.report(a_again, one.id).title == "New private title"
        reopened.set_sharing(a_again, one.id, shared=True)
        reopened.delete_report(a_again, one.id)
        assert not reopened.shared_reports()
        with pytest.raises(WorkspaceError, match="unavailable"):
            reopened.report(a_again, one.id)
        reopened.set_sharing(a_again, hidden.id, shared=True)
        reopened.clear_history(a_again)
        assert not reopened.history(a_again) and not reopened.shared_reports()
        assert reopened.history(b_again) == [beta_report]
        with reopened.engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA foreign_key_check").all() == []
            assert conn.exec_driver_sql("PRAGMA integrity_check").scalar_one() == "ok"
    finally:
        reopened.engine.dispose()


def test_concurrent_migration_and_writes_preserve_registry(tmp_path, monkeypatch, evidence):
    path = tmp_path / "existing-research.db"
    monkeypatch.setenv("CARDIOSCOPE_DATABASE_URL", f"sqlite:///{path}")
    engine = registry.open_registry()
    registry.import_evidence(engine, evidence)
    legacy = registry.listing(engine)
    table_names_before = inspect(engine).get_table_names()

    def job(username, family):
        local = reopen_workspace(path)
        try:
            grant = sign_in(local, username)
            expected = Settings(model=family)
            local.save_settings(grant, asdict(expected))
            for _ in range(2):
                local.save_report(grant, evidence, A, EXAMPLES[A])
            return local.account(grant), local.history(grant)
        finally:
            local.engine.dispose()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(job, "concurrent_a", "forest")
        second = pool.submit(job, "concurrent_b", "xgboost")
        a, a_reports = first.result()
        b, b_reports = second.result()
    assert a.settings.model == "forest" and b.settings.model == "xgboost"
    assert len(a_reports) == len(b_reports) == 2
    assert {r.owner_id for r in a_reports} == {a.id}
    assert {r.owner_id for r in b_reports} == {b.id}
    assert table_names_before == ["experiments"]
    assert registry.listing(engine) == legacy
    assert len(legacy) == 102
    engine.dispose()


def test_streamlit_first_account_workspace_and_preferences(db, monkeypatch):
    store, _ = db
    token = sign_in(store, "editor")
    monkeypatch.setattr("cardioscope.dashboard.workspace_ui.token", lambda: token)
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30)
    app.run()
    app.switch_page("views/my_reports.py").run()
    assert not app.exception
    assert {x.value for x in app.info} >= {
        "You have no saved fictional profiles.",
        "You have no saved educational reports.",
    }
    app.switch_page("views/sandbox.py").run()
    assert not app.exception
    assert app.selectbox[0].value is None
    assert not app.number_input and not app.metric
    store.save_settings(token, asdict(Settings(show_grid=False, model="forest", tier="extended")))
    app.switch_page("views/reliability.py").run()
    assert not app.exception
    assert app.selectbox[0].value == "forest"
    assert app.selectbox[1].value == "extended"
    assert '"showgrid":false' in app.get("plotly_chart")[0].proto.spec.replace(" ", "")
