"""Evidence boundaries, aggregate persistence, and real Streamlit page interactions."""

from __future__ import annotations

import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
from sqlalchemy import create_engine, inspect
from streamlit.testing.v1 import AppTest

from cardioscope.dashboard import charts, registry, sandbox
from cardioscope.dashboard.evidence import ROOT, EvidenceError, load_evidence, site_run, validate

APP = str(ROOT / "app/streamlit_app.py")
PAGES = [
    "overview",
    "dataset",
    "laboratory",
    "reliability",
    "calibration",
    "sandbox",
    "registry",
    "methodology",
]


@pytest.fixture
def evidence():
    return load_evidence()


def test_verified_reports_and_integrity(evidence):
    assert evidence.audit["rows"] == 920
    assert len(registry.records(evidence)) == 102
    assert evidence.cv["selection"]["cv_recommendations"] == {
        "early": "logistic",
        "extended": "logistic",
    }
    for s in evidence.transfer["sites"]:
        for r in s["runs"]:
            m = r["metrics"]
            assert m["balanced_accuracy"] == pytest.approx(
                (m["sensitivity"] + m["specificity"]) / 2, abs=1e-6
            )


def test_missing_modified_and_malformed_reports(tmp_path, evidence):
    with pytest.raises(EvidenceError, match="missing"):
        load_evidence(tmp_path)
    (tmp_path / "docs").mkdir()
    manifest = json.loads((ROOT / "docs/dashboard-artifacts.json").read_text())
    (tmp_path / "docs/dashboard-artifacts.json").write_text(json.dumps(manifest))
    for name in manifest:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    with pytest.raises(EvidenceError, match="fingerprint"):
        load_evidence(tmp_path)
    invalid = copy.deepcopy(evidence)
    invalid.transfer["sites"][0]["runs"][0]["metrics"]["roc_auc"] = float("nan")
    with pytest.raises(EvidenceError):
        validate(invalid)
    invalid = copy.deepcopy(evidence)
    del invalid.pooled["evaluation"][0]["holdout_metrics"]["brier"]
    with pytest.raises(EvidenceError):
        validate(invalid)
    invalid = copy.deepcopy(evidence)
    invalid.protocol["dataset_sha256"] = "different"
    with pytest.raises(EvidenceError):
        validate(invalid)


def test_chart_values_and_unavailable_state(evidence):
    sites = evidence.transfer["sites"]
    figure = charts.hospital_chart(sites, "logistic", "early")
    expected = [site_run(s, "logistic", "early")["metrics"]["balanced_accuracy"] for s in sites]
    assert list(figure.data[0].y) == expected
    row = site_run(sites[0], "logistic", "early")
    assert figure.data[0].error_y.array[0] == pytest.approx(
        row["intervals"]["balanced_accuracy"]["bounds"][1] - expected[0]
    )
    calibration = charts.calibration([row])
    assert list(calibration.data[1].x) == [
        b["mean_p"] for b in row["diagnostics"]["bins"] if b["n"]
    ]
    assert list(calibration.data[1].customdata) == [
        b["n"] for b in row["diagnostics"]["bins"] if b["n"]
    ]
    assert charts.bars(["Undefined"], [None], "ROC-AUC", True).data[0].y[0] is None
    assert site_run(sites[0], "nonexistent", "early") is None


def test_sqlite_import_idempotency_and_no_patient_records(tmp_path, evidence, monkeypatch):
    monkeypatch.setenv("CARDIOSCOPE_DATABASE_URL", f"sqlite:///{tmp_path / 'registry.db'}")
    engine = registry.open_registry()
    assert registry.import_evidence(engine, evidence) == 102
    original = registry.listing(engine)
    assert registry.import_evidence(engine, evidence) == 102
    assert registry.listing(engine) == original
    assert inspect(engine).get_table_names() == ["experiments"]
    assert {c["name"] for c in inspect(engine).get_columns("experiments")} == {
        "id",
        "model",
        "tier",
        "evaluation",
        "hospital",
        "metadata",
        "imported_at",
    }
    for row in original:
        assert (
            not {"inputs", "prediction", "patient", "raw", "probabilities"} & row["metadata"].keys()
        )
        assert row["metadata"]["dataset_sha256"] == evidence.audit["sha256"]
    engine.dispose()


def test_database_configuration_and_safe_errors(monkeypatch, tmp_path, evidence):
    monkeypatch.setenv(
        "CARDIOSCOPE_DATABASE_URL", "postgresql://user:secret@example.invalid/research"
    )
    assert registry.database_url().drivername == "postgresql+psycopg"
    monkeypatch.setenv("CARDIOSCOPE_DATABASE_URL", "mysql://user:secret@example.invalid/research")
    with pytest.raises(registry.RegistryError) as failure:
        registry.database_url()
    assert "secret" not in str(failure.value)
    monkeypatch.setenv("CARDIOSCOPE_DATABASE_URL", f"sqlite:///{tmp_path / 'absent' / 'db'}")
    with pytest.raises(registry.RegistryError):
        registry.open_registry()
    engine = create_engine("sqlite://")
    with pytest.raises(registry.RegistryError):
        registry.listing(engine)
    with pytest.raises(registry.RegistryError):
        registry.import_evidence(engine, evidence)
    engine.dispose()


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", float("nan")),
        ("age", True),
        ("age", 17),
        ("trestbps", 0),
        ("chol", 701),
        ("sex", "unknown"),
        ("fbs", True),
        ("cp", "unknown"),
    ],
)
def test_fictional_input_validation(field, value):
    with pytest.raises(ValueError):
        sandbox.validate_profile({**sandbox.PROFILE, field: value})


def test_early_feature_contract_and_missing_artifact(tmp_path, evidence):
    frame = sandbox.validate_profile({**sandbox.PROFILE, "chol": 0.0})
    assert list(frame.columns) == evidence.protocol["tiers"]["early"]
    with pytest.raises(ValueError):
        sandbox.validate_profile({**sandbox.PROFILE, "id": 42})
    with pytest.raises(EvidenceError):
        sandbox.load_verified(evidence, tmp_path)
    (tmp_path / "sandbox.json").write_text(
        json.dumps({**sandbox.expected_metadata(evidence), "tier": "extended"})
    )
    with patch("joblib.load", side_effect=AssertionError("must not deserialize")):
        with pytest.raises(EvidenceError):
            sandbox.load_verified(evidence, tmp_path)


def test_local_model_build_and_inference_reproducibility(tmp_path, evidence, caplog):
    if not (ROOT / "data/raw/heart_disease_uci.csv").exists():
        pytest.skip("Verified source CSV is local-only; install it to exercise explicit refit")
    sandbox.build(tmp_path)
    with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("UI must never fit")):
        model = sandbox.load_verified(evidence, tmp_path)
        x = sandbox.validate_profile(sandbox.PROFILE)
        first = model.predict_proba(x)
        second = sandbox.load_verified(evidence, tmp_path).predict_proba(x)
        np.testing.assert_array_equal(first, second)
        assert 0 <= first[0, 1] <= 1
    assert not caplog.records
    (tmp_path / "sandbox.joblib").write_bytes(b"changed")
    with patch("joblib.load", side_effect=AssertionError("must verify before loading")):
        with pytest.raises(EvidenceError):
            sandbox.load_verified(evidence, tmp_path)


def app_page(page):
    app = AppTest.from_file(APP, default_timeout=45).run()
    if page != "overview":
        app.switch_page(f"views/{page}.py").run()
    assert not app.exception
    return app


@pytest.mark.parametrize("page", PAGES)
def test_all_navigation_pages_without_training(page, tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIOSCOPE_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    with patch(
        "sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("navigation may not train")
    ):
        app = app_page(page)
    assert app.title
    assert not app.error


def test_laboratory_protocol_and_tier_controls(evidence):
    app = app_page("laboratory")
    app.selectbox(key="lab_model").select("forest")
    app.selectbox(key="lab_tier").select("extended").run()
    row = next(
        r
        for r in evidence.pooled["evaluation"]
        if (r["family"], r["tier"]) == ("forest", "extended")
    )
    assert app.metric[0].value == f"{row['holdout_metrics']['balanced_accuracy']:.3f}"
    app.selectbox[2].select("Training-only cross-validation").run()
    row = next(
        r
        for r in evidence.cv["selection"]["runs"]
        if (r["family"], r["tier"]) == ("forest", "extended")
    )
    assert app.metric[0].value == f"{row['oof_metrics']['balanced_accuracy']:.3f}"
    assert app.warning
    assert not app.exception


def test_hospital_controls_update_detail(evidence):
    app = app_page("reliability")
    app.selectbox(key="reliability_model").select("xgboost")
    app.selectbox(key="reliability_tier").select("extended")
    app.selectbox(key="reliability_hospital").select("Switzerland").run()
    site = next(s for s in evidence.transfer["sites"] if s["hospital"] == "Switzerland")
    assert app.metric[0].value == str(site["test"])
    assert app.metric[2].value == str(site["negative"])
    assert app.warning
    assert not app.exception
    app.selectbox(key="reliability_hospital").select("Hungary").run()
    assert app.metric[0].value != str(site["test"])
    assert not app.warning


def test_error_pages_and_disabled_sandbox(monkeypatch):
    with patch(
        "cardioscope.dashboard.pages.load_evidence", side_effect=EvidenceError("missing reports")
    ):
        app = AppTest.from_file(APP, default_timeout=45).run()
        assert app.error[0].value == "missing reports"
        assert not app.get("plotly_chart")
    with patch(
        "cardioscope.dashboard.sandbox.load_verified",
        side_effect=EvidenceError("model unavailable"),
    ):
        app = app_page("sandbox")
        assert app.button[-1].disabled
    monkeypatch.setenv("CARDIOSCOPE_DATABASE_URL", "unsupported://secret")
    app = app_page("registry")
    assert app.error
    assert "secret" not in app.error[0].value


def test_registry_ui_import_and_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIOSCOPE_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    app = app_page("registry")
    assert "empty" in app.info[0].value
    app.button[0].click().run()
    assert not app.exception
    app.selectbox(key="registry_model").select("logistic").run()
    assert set(app.dataframe[0].value["model"]) == {"logistic"}


def test_calibration_controls_and_paired_evidence(evidence):
    app = app_page("calibration")
    app.selectbox(key="calibration_model").select("forest")
    app.selectbox(key="calibration_tier").select("extended")
    app.selectbox(key="calibration_hospital").select("Switzerland")
    app.selectbox[3].select("zero_as_missing").run()
    site = next(s for s in evidence.transfer["sites"] if s["hospital"] == "Switzerland")
    expected = site_run(site, "forest", "extended", "zero_as_missing")
    assert app.dataframe[0].value.iloc[0]["Brier score"] == expected["metrics"]["brier"]
    assert app.dataframe[0].value.iloc[0]["Observed prevalence"] == site["prevalence"]
    assert list(app.dataframe[1].value["Hospital"]) == [
        s["hospital"] for s in evidence.transfer["sites"]
    ]
    assert not app.exception


def test_dataset_missingness_selector(evidence):
    app = app_page("dataset")
    app.selectbox[1].select("VA Long Beach").run()
    site = next(s for s in evidence.audit["hospitals"] if s["hospital"] == "VA Long Beach")
    table = app.dataframe[0].value.set_index("Feature")
    assert table.loc["ca", "Missing records"] == site["missing"]["ca"]["count"]
    assert not app.exception


def test_sandbox_state_and_no_persistence(tmp_path, monkeypatch, caplog):
    class FakePipeline:
        def predict_proba(self, x):
            assert list(x.columns) == ["age", "sex", "cp", "trestbps", "chol", "fbs"]
            return np.array([[0.7, 0.3]])

    monkeypatch.setenv("CARDIOSCOPE_DATABASE_URL", f"sqlite:///{tmp_path / 'never-created.db'}")
    with (
        patch("cardioscope.dashboard.sandbox.load_verified", return_value=FakePipeline()),
        patch(
            "cardioscope.dashboard.registry.open_registry",
            side_effect=AssertionError("no persistence"),
        ),
    ):
        app = app_page("sandbox")
        app.button[-1].click().run()
        assert app.metric[0].value == "0.300"
        app.selectbox(key="sandbox_preset").select("Fictional B").run()
        assert not app.metric
        assert app.number_input[0].value == 64.0
        app.number_input[0].set_value(70.0)
        app.button[0].click().run()
        assert app.number_input[0].value == 64.0
        assert not app.exception
    assert not (tmp_path / "never-created.db").exists()
    assert not [r for r in caplog.records if r.name.startswith("cardioscope")]
