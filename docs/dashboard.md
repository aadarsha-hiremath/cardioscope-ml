# Phase 5 research dashboard

CardioScope ML presents verified retrospective evidence about hospital transfer, missing measurements and probability reliability. It does not establish clinical validity. All four hospitals informed earlier pooled development; an untouched external cohort is still needed.

## Run locally on Windows

From the repository root, use Python 3.12 and PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade "pip==26.3"
.\.venv\Scripts\python.exe -m pip install -c requirements-dev.lock -e ".[dev,dashboard]"
.\.venv\Scripts\python.exe -m streamlit run app/serve.py
```

Open `http://127.0.0.1:8501`. The server binds to loopback. Analytics collection is disabled. The eight research pages work from tracked aggregate reports; a raw CSV, GPU and PostgreSQL server are not required. With file watching disabled, restart the server after editing application code.

The Welcome page gates dashboard access. Registration signs you in and opens Research Overview immediately. The authenticated sidebar includes Account, Settings, My Reports, Shared Fictional Reports, and Logout. Workspace storage uses the same database as the separate research catalog. [Session and workspace guide](workspaces.md)

## Pages

| Page | Evidence and interactions |
| --- | --- |
| Research Overview | Primary early Logistic Regression, equal-hospital mean balanced accuracy, hospital inspection, uncertainty and interpretation |
| Dataset Explorer | Record/class counts and prevalence, missingness by hospital, types and feature tiers, unresolved data-quality issues |
| Model Laboratory | Three models, two tiers, pooled holdout or training-only CV; AP comparison, nine metrics, thresholds, confusion matrix and configuration |
| Hospital Reliability | Four held-out hospitals, model/tier controls, training arrangement, counts, metrics and group-bootstrap intervals |
| Calibration & Sensitivity | Recorded reliability bins, raw/sigmoid scores, Brier/log loss, mean score versus prevalence, paired cholesterol-policy changes |
| Prediction Sandbox | Explicitly fictional early-tier inputs, verified local pipeline, validation/reset and experimental output |
| Experiment Registry | Idempotent aggregate import, SQLite storage, filters, metadata inspection and selected-record comparisons |
| Methodology & Reproducibility | Expandable protocols, attribution, downloadable source documents, external-validation requirements |

BA is the mean of sensitivity and specificity. AP is average precision, not a thresholded accuracy measure. ROC-AUC describes ranking. Brier score and log loss measure probability error; lower is better. Missing metrics display as unavailable, never as zero. CV charts show mean fold AP; the development detail table shows concatenated out-of-fold metrics, whose threshold was optimized on those same predictions. The interface labels this selection optimism explicitly.

## Architecture and presentation

```text
app/streamlit_app.py               native st.Page/st.navigation entrypoint
app/views/*.py                     research and personal workspace entrypoints
app/assets/wordmark.svg            local vector wordmark
src/cardioscope/dashboard/
  pages.py                        reusable native layouts and views
  charts.py                       aggregate Plotly figures and styling
  evidence.py                     report fingerprints and semantic validation
  sandbox.py                      explicit refit CLI and verified inference loader
  registry.py                     SQLAlchemy Core schema and aggregate import
.streamlit/config.toml             theme, loopback server, telemetry setting
docs/dashboard-artifacts.json      frozen report byte fingerprints (LF-normalized)
tests/test_dashboard.py            service and Streamlit AppTest coverage
```

The approved analytics/laboratory hybrid uses navy backgrounds, slate cards, cyan comparisons, restrained teal accents and amber limitations. Native themes, forms, selectors, focus behavior, containers and responsive columns supply the UI. A tiny CSS rule colors only the documented `st-key-` classes of application-owned metric cards. No generated DOM selectors, injected JavaScript, remote fonts, external images or unofficial components are used. Text uses the bundled sans-serif default; Plotly uses Inter only if the client already has it, then system sans-serif.

Report bytes are checked on each page rerun so missing or changed evidence cannot be hidden by a stale report cache. The manifest pins six authoritative artifacts. Semantic checks cover dataset identity, protocol equality, result coverage, metrics, outcome counts, uncertainty bounds and calibration-bin counts. These are trusted repository files, not an upload format. Updating research evidence requires deliberate review and manifest regeneration; the UI never does that automatically. Existing Phase 1â€“4 code, reports, dataset provenance and frozen protocols are unchanged.

Only the sandbox loads a fitted model. A one-entry cache is keyed by the verified model bytes; compatibility and byte checks still run before every access. It never caches fictional profiles or scores. No training or research experiment runs during navigation.

## Local demonstration model

The model is optional and intentionally absent from Git. First acquire the exact CSV following [provenance](provenance.md), then explicitly run:

```powershell
.\.venv\Scripts\python.exe -m cardioscope.dashboard.sandbox
```

This uses existing Phase 2â€“3 preparation, grouped split and model-construction functions. It fits only the selected early Logistic Regression on the original 735 development records. The 185 held-out probabilities must reproduce Phase 3's recorded digest exactly before `models/sandbox.joblib` and its JSON manifest are written. This is a verification refit, not a new experiment or a change to model selection.

Loading checks dataset and selection fingerprints, family/tier, hyperparameters, threshold, prediction digest, library compatibility, model-byte checksum, input columns and class ordering. Missing, corrupt or incompatible files disable submission. Joblib is a trusted-local serialization boundary: a checksum does **not** make an attacker-controlled pickle safe. Never replace both local files with downloaded models. There is no upload endpoint or user-selectable model path.

The form accepts six early features and intentionally synthetic examples. Bounds are demonstration constraints rather than clinical reference ranges. Cholesterol zero remains as recorded, matching the selected model. Inputs travel only to the loopback server. Ordinary inference and edited inputs remain transient. A signed-in user can explicitly save an unedited built-in example's result in the personal workspace. No feature vector is stored. Output describes a historical dataset class and an experimental score. Changing examples clears the output, and reset restores that example's defaults. [Persistence and sharing rules](workspaces.md)

## Aggregate experiment tracking

Open Experiment Registry and select **Import verified research records**. This initializes the local SQLite database under ignored `data/local/experiments.db`. The import contains six pooled holdout and 96 hospital result records, with deterministic content IDs. Repeating an import does not duplicate identical records or change their original import timestamps. Import is transactional; database errors do not prevent the other pages from operating.

Stored fields are model, tier, evaluation, hospital, aggregate metric summaries, hyperparameters, threshold, policy/calibration, seed, software versions, dataset/protocol fingerprints, artifact reference and import timestamp (UTC). SQLite may return naive timestamps; their meaning is still UTC. No raw input rows, individual probability arrays or prediction attempts are stored. Content IDs distinguish changes in evidence; this registry is an evidence index, not invented historical execution tracking.

Optional PostgreSQL:

```powershell
.\.venv\Scripts\python.exe -m pip install -c requirements-dev.lock -e ".[dashboard,postgres]"
# Obtain this value securely from your environment; do not commit credentials.
$env:CARDIOSCOPE_DATABASE_URL = $env:RESEARCH_DATABASE_URL
.\.venv\Scripts\python.exe -m streamlit run app/serve.py
```

Use a SQLAlchemy `postgresql+psycopg://` URL; `postgresql://` is normalized to that driver. SQLite URLs are also supported. PostgreSQL is optional: driver setup, URL validation and dialect-specific conflict handling are implemented; no live PostgreSQL instance was provisioned or exercised for this phase. Driver/server errors display credential-free messages. Never put credentials in the repository, screenshots or shared logs.

## Verification and limitations

The Phase 4 starting point was clean commit `2aad6f3`, with 39 passing tests. Phase 5 adds 30 tests, including all eight native navigation pages, protocol/tier/hospital selection, chart evidence, report errors, SQLite import/idempotency, safe database failures, PostgreSQL URL validation, input bounds, artifact compatibility, exact refit reproduction, session reset, no navigation-time fitting and no prediction persistence. The full run passed **69 tests**. Ruff lint, Ruff format, strict MyPy and `pip check` passed. The optional refit test skips on checkouts without the local CSV; it ran here.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m pip check
```

The real local Streamlit server and browser were exercised. See [visual verification](dashboard-visual-review.md) for actual screenshots and observations. Results reflect this local environment, not a portable latency guarantee. Report navigation requires no training. Only the optional sandbox refit needs raw data; inference runs on CPU.

Limitations remain scientific and operational: the dataset is small and historical; subgroup support is uneven; Switzerland has few negative outcomes; recorded-zero meanings are unresolved; missingness can encode site collection practices; extended diagnostic fields imply a later information setting. Within-site intervals do not capture future population uncertainty. A local registry is not a multi-user clinical system. The original CSV's redistribution permission is unresolved and it remains ignored. See [attribution and license details](provenance.md).

## Proposed Phase 6 checklist â€” not started

- Reproduce the locked audit, pooled comparison and transfer outputs in a fresh environment, retaining original reports for comparison.
- Repeat dashboard tests, keyboard/narrow-screen review, dependency and secret checks on the intended distribution machine.
- Review every reported claim, unavailable metric, uncertainty interval and healthcare disclaimer against source artifacts.
- Verify redistribution permissions, attribution, artifact trust boundaries and optional PostgreSQL against an explicitly authorized test server.
- Define requirements for an untouched external cohort: compatible outcome, available predictors, measurement timing, population, provenance, permissions and preregistered evaluation.
- Obtain approval before any publication, deployment, additional research or Phase 6 implementation.
