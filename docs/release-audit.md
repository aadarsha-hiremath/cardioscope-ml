# Phase 6 release readiness audit

Audit date: 2026-11-19. Scope: a local educational research portfolio built from Phase 5.1 (`21265a0`). The approved dark dashboard and frozen Phase 1–4 evidence remain in place. Publication and repository visibility are owner decisions.

## Findings

| Severity | Area | Finding and disposition |
| --- | --- | --- |
| High | Dependency security | The previous setuptools range resolved to an affected build backend, and the Python 3.12 environment initially seeded pip 25.0.1. Pinned setuptools 84.2.1 and documented pip 26.3 in setup. The revised lock and both tool versions passed the PyPI advisory rescan. See the [setuptools archive advisory](https://github.com/advisories/GHSA-h35f-9h28-mq5c), [Windows legacy-install advisory](https://github.com/advisories/GHSA-xmj3-9gf7-h52w), and [pip advisory](https://github.com/advisories/GHSA-qwm4-qh6w-59xr). |
| Medium | Developer tooling | pytest 8.4.2 was affected by a Unix temporary-directory advisory. It was a developer dependency, and the Windows audit used an isolated test directory. Upgraded to 9.0.3 for contributors on affected platforms. [Advisory](https://github.com/advisories/GHSA-6w46-j5rx-g56g) |
| Medium | Accessibility | The chart figures had no accessible names. They now have contextual names, with exact counts in confusion-matrix descriptions. The cyan submit and import buttons now use dark text within their existing controls. |
| Medium | Usability | Relative links from embedded Markdown used the web route as their base. Methodology previews now show repository paths, retain external links, and offer the original source Markdown for download. |
| Informational | Scientific scope | The four hospital holdouts are retrospective transfer stress tests. Every hospital contributed records to earlier pooled development. These results do not establish independent external clinical validation. |
| Informational | Operating scope | The supported portfolio workflow is a local checkout and loopback dashboard. PostgreSQL is optional and live PostgreSQL was outside this audit. An installed wheel exposes the Python package and audit CLI. Running the dashboard requires the repository’s app entrypoint and aggregate reports. |
| Informational | Licensing | The owner has not selected a source-code license. The README states this explicitly and separates it from dataset attribution and redistribution terms. Decide before describing the project as open source or granting reuse rights. |

## A. Scientific integrity

- `data/loading.py` hashes the exact CSV byte snapshot before parsing. The Kaggle v6 file contains 920 rows and matches SHA-256 `574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae`. The UCI comparison accounts for the one reassigned Cleveland/Hungary site label. Its true enrollment site remains unknown.
- `data/audit.py` maps original outcomes 1–4 to the recorded positive class and 0 to negative. `preprocessing.py` keeps source ID, hospital and original outcome in metadata, uses six early or 13 extended predictors, and computes exact duplicate groups before feature-tier selection and cholesterol recoding. Grouping protects the assessed split boundaries, but the sources lack a shared patient identifier.
- `splits.py`, `training.py` and `experiment.py` use duplicate-aware stratified partitions. Five development-only folds select configurations by mean fold average precision and set cutoffs on development out-of-fold scores. Every fitting fold has fresh median imputation, scaling, categorical missing-token handling and encoding. The frozen configuration and cutoff precede holdout evaluation. The holdout is from the same released site mixture.
- `transfer.py` verifies the Phase 4 manifest and fits each fixed Phase 3 configuration on three hospitals, excluding same-profile overlaps with the held-out site. Inner leave-one-site-out scores set the cutoff and fit an optional positive-slope sigmoid. Outer labels support count checks, metrics and diagnostics, and they never enter any model, threshold or calibrator fit. Outer calibration intercept and slope are descriptive. The cholesterol analysis compares raw zeros against masking only that field, then applies the same fold-local fitting rules.
- `scoring.py` defines rates from the confusion matrix, uses the recorded positive class for ROC-AUC, AP and Brier score, and represents undefined metrics without substituting zero. `reliability.py` resamples complete duplicate groups within each held-out site and uses common draws for paired contrasts. Intervals condition on the observed sample and fitted model. In Switzerland, only eight negatives support specificity.
- Transfer aggregates are equal-site and record-count-weighted summaries of site metrics. They do not combine model scores across hospitals. AP depends on prevalence, so the results retain the no-skill prevalence baseline and relative lift. A high Swiss AP alone is weak evidence of portability. The dashboard derives its display from six fingerprinted artifacts, keeping pooled holdout, development CV and hospital transfer separately labeled.

The freeze commits precede their corresponding result commits. Code review and artifact comparisons support the published retrospective claims. The repository cannot independently prove choices made outside recorded history. The older data source, referral case mix, feature timing, unresolved zeros, and limited site support remain visible caveats.

## B. Software quality and reproduction

The audit used a separate tracked-file export and new virtual environment on Windows x64 (build 26300), CPython 3.12.14. The baseline suite at `21265a0` passed 79 tests in 38.33 seconds. The corrected environment passed 78 tests without data or a model in 32.01 seconds. Two tests explicitly skipped because they require the local CSV or a fitted artifact. With the authorized data and freshly built model, all 80 tests passed in 44.04 seconds. The patched editable installation, after the baseline packages had been fetched, took 46.03 seconds and matched the Windows lock. Ruff lint and format, strict MyPy and `pip check` passed. A cold package import took 1.64 ms and imported no ML, dashboard or database libraries.

Audit copies of the source inputs and all replay outputs stayed under an ignored audit directory. To repeat the process, export a clean checkout, create a fresh Python 3.12.14 environment, follow the README setup, and run these commands from that copy. Add the authorized source files locally only after checking the data-free application.

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade "pip==26.3"
.\.venv\Scripts\python.exe -m pip install -c requirements-dev.lock -e ".[dev,dashboard]"
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.audit-no-data
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.port 8512
```

```powershell
$csv = "data/raw/heart_disease_uci.csv"
.\.venv\Scripts\python.exe -m cardioscope.cli --input $csv --output-dir replay/phase1
.\.venv\Scripts\python.exe scripts/compare_sources.py --input $csv --uci-zip data/raw/uci-reference.zip
.\.venv\Scripts\python.exe -m cardioscope.protocol --input $csv --output replay/preprocessing-protocol.json
.\.venv\Scripts\python.exe -m cardioscope.experiment --stage cv --tier both --input $csv --output-dir replay/phase3
.\.venv\Scripts\python.exe -m cardioscope.experiment --stage evaluate --tier both --input $csv --output-dir replay/phase3
.\.venv\Scripts\python.exe -m cardioscope.experiment --stage report --tier both --output-dir replay/phase3
.\.venv\Scripts\python.exe -m cardioscope.transfer --input $csv --output-dir replay/phase4
.\.venv\Scripts\python.exe -m cardioscope.dashboard.sandbox
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.audit-with-data
```

| Replay | Seconds | Comparison with the tracked reference |
| --- | ---: | --- |
| Phase 1 audit | 0.94 | Markdown and JSON matched with normalized line endings. Audit JSON SHA-256 `fd00e744bbfaeb74fde4b00a7379b12eb25f9da853f47f0cf203c1dbc4e72222`. |
| Phase 2 source comparison and preprocessing | 4.11 + 3.13 | The source comparison reproduced the single site-label change and duplicate/zero counts. Preprocessing JSON SHA-256 `c80af74ec16185a666f2666de53aebf6cba9a1b9445b5bdb28f910f1530c5dd2`. |
| Phase 3 CV, holdout and Markdown | 10.15 + 3.00 + 2.25 | Selection SHA-256 `457bd2ba5a1615855566d6fbb9531bfddcc8eb77f3eb5e1931aa53ffc4abb49a`. The six holdout metric sets, cutoffs, interval bounds and probability digests matched. Markdown differences were confined to timings. |
| Phase 4 transfer | 30.71 | Scientific SHA-256 `c5ceaaf55c4e88916d07265962e10c5100d96d3fc4a45ff58d928acb8d0e280c`. The 96 runs matched after excluding elapsed time. Four Markdown reports and two PNGs matched byte for byte. |
| Trusted sandbox build | 3.23 | The builder verified the original held-out probability digest before writing the isolated artifact. Fictional A produced 0.050. |

The replay loaded and checked all six dashboard reports in 7.86 ms. With the audit’s libraries already imported, first sandbox artifact load took 19.30 ms and a cached compatibility check 9.01 ms. These are single local measurements. The isolated registry reimport preserved the 102 identities and timestamps, including six pooled and 96 transfer records. Matplotlib fell back to a writable temporary cache under the audit sandbox and still reproduced the figures.

Use `python -m build` and `python -m twine check --strict dist/*` in a separate build-tools environment. The wheel and source archive built with setuptools 84.2.1 and passed the strict metadata checks in 25.57 seconds. An isolated wheel import and CLI help smoke passed, and the 28 wheel and 45 source-archive entries contained no private data, models or database files. The dashboard runs from the source checkout and the Python CLI is available as a package.

## C. Security, privacy and licensing

The local `origin` is the expected GitHub repository and the checkout contains no remote-tracking refs. Local refs and reflogs exposed the same eight pre-audit commits. The audit verified Git author and committer emails against the repository noreply identity. Remote account visibility and previous actions outside this repository cannot be inferred from the local object database.

Gitleaks 8.30.1, downloaded with its official release checksum verified, reported no secrets in the eight starting commits. Supplemental path and content checks found no private data, model, database, archive, key, cache, log, or executable paths in reachable history. Email-like strings in tracked text were decorators and reserved `example.invalid` fixtures. The 16 PNGs total about 1.9 MB and show aggregate research or fictional UI entries. Their metadata contains only software identification or resolution. Raw files, fitted models, SQLite, and secrets remain ignored.

The dashboard uses fixed evidence paths, JSON validation and bound SQLAlchemy statements. SQLite stores aggregate experiment metadata and import timestamps. The optional connection string comes from the environment and user-facing errors omit it. Streamlit binds to loopback with usage telemetry disabled. The sandbox accepts fictional values only. Its inputs and outputs stay in the local Streamlit session and are absent from app logging and the registry.

The sandbox deserializes a locally built joblib file after checking its manifest and byte checksum. That checksum guards against accidental incompatibility. A matching model and manifest from an untrusted party are still unsafe. There is no model upload or UI path selector. The distribution workflow builds from the checksummed CSV locally and keeps both generated files out of Git.

UCI credits Andras Janosi, William Steinbrunn, Matthias Pfisterer and Robert Detrano and lists CC BY 4.0. The Kaggle API still reports version 6 and “Data files © Original Authors.” The UCI terms do not independently establish permissions for this combined distribution. Keep the CSV, downloaded archives and fitted models excluded. The screenshots capture the project UI, aggregate results and synthetic examples. No downloaded third-party screenshot or source dataset is included. The UI uses Streamlit’s Material Symbols icons, for which Google lists Apache 2.0. The owner should confirm rights for the materials they choose to publish. [UCI dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Kaggle metadata](https://www.kaggle.com/api/v1/datasets/view/redwankarimsony/heart-disease-data) · [Material Symbols](https://developers.google.com/fonts/docs/material_symbols)

Major software dependencies include [Streamlit](https://streamlit.io/) and [XGBoost](https://xgboost.readthedocs.io/) (Apache-2.0), [Plotly.py](https://plotly.com/python/) and [SQLAlchemy](https://www.sqlalchemy.org/) (MIT), [pandas](https://pandas.pydata.org/), [SciPy](https://scipy.org/), [scikit-learn](https://scikit-learn.org/) and [joblib](https://joblib.readthedocs.io/) (BSD family), and [Matplotlib](https://matplotlib.org/) (PSF-based). Dependency distributions have their own notices and licenses. The lock records versions and the repository does not vendor dependency packages.

For project code, leaving it unlicensed allows a viewing-oriented portfolio under the host’s terms and supplies no general reuse permission. MIT is a short permissive option. Apache 2.0 is permissive and includes an express patent grant and notice terms. GPLv3 adds reciprocal distribution requirements. The owner decides whether to grant reuse rights. [No-license guidance](https://choosealicense.com/no-license/) · [License options](https://choosealicense.com/licenses/)

The final `pip-audit` 2.10.1 scan queried 63 exact versions from the Windows project lock plus pip and setuptools. It reported no known PyPI advisories in that set as of the audit date. This scope excludes live infrastructure, OS dependencies, bundled browser assets and the optional PostgreSQL environment. The version constraints do not pin wheel hashes.

## D. Browser behavior and accessibility

All eight direct routes opened in separate fresh sessions in Chrome 157.0.8103.79, Edge 157.0.4341.60, Playwright Firefox 159.0 and Playwright WebKit 27.2 on Windows. Every document request returned HTTP 200. Refreshes returned 200 or a valid cached 304 at the Firefox root. Subpage requests initially tried `/<page>/_stcore/health` and `/<page>/_stcore/host-config` (404), then both corresponding root requests returned 200. The selected page rendered on first load and refresh. Chrome back and forward restored the expected pages. This was internal Streamlit endpoint discovery, with no broken user-facing deep link in the tested sessions. WebKit is engine coverage rather than a test of Safari on Apple hardware.

The Chrome workflow exercised dataset charts, held-out hospitals and model/tier controls, cholesterol sensitivity, development CV, registry filters and repeat import, Methodology expanders and Markdown download. It checked back and forward, unavailable inference, fictional scores of 0.050 and 0.910, invalid age feedback, and reset. The dashboard worked from aggregate reports in the data-free viewer. Its registry started empty, and two imports left 102 rows. Browser-observed HTTP and WebSocket destinations during sandbox inference were both loopback. The audit found no individual sandbox persistence or input logging.

All eight pages rendered at 390 × 844 with no document or main-region horizontal overflow. Chrome also passed the sandbox reflow check at 200% CSS zoom. Keyboard traversal reached the skip link, navigation and labeled sandbox controls. The focused synthetic-example control’s visible border changed from slate to cyan. Browser accessibility trees expose named figures and the confusion-matrix counts. Native data tables have labeled columns and contextual headings. Button text contrast improved from 2.14:1 to 8.79:1, above the 4.5:1 WCAG normal-text threshold. This was a targeted review, with no manual screen-reader session or claim of full WCAG conformance. [WCAG minimum contrast guidance](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html)

Chrome warm direct-route readiness was typically around 0.35–0.7 s. Cold startup and first registry access could take several seconds, particularly while other audit work was running. Measurements include browser and Streamlit session startup and are not load-testing results.

## E. Portfolio release decision

No blocker was identified for a static educational GitHub portfolio after the targeted corrections. The README describes the use case, local launch, provenance, findings and limits, and links to the existing authentic screenshots. The public release scope would be source code, aggregate reports, documentation and reviewed screenshots. A hosted clinical or multi-user deployment remains outside the supported scope.

Before a public release, the owner should approve publication, confirm rights for the selected materials, and decide whether to license code reuse or retain a view-only portfolio. Repeat the dependency and history checks against the exact publication commit. Keep the dataset, local model, database and credentials excluded. If operating a hosted or PostgreSQL instance later, audit that exact deployment separately. External clinical validation remains a separate research milestone.
