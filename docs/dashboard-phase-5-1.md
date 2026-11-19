# Phase 5.1 dashboard acceptance

Phase 5.1 starts from the clean Phase 5 commit `779c2ad` and keeps the approved dark layout. The frozen Phase 1–4 reports, source CSV, preprocessing rules, model configuration and fitted sandbox files were preserved.

| Area | Verified outcome |
| --- | --- |
| Dataset Explorer | Cleveland 304, Hungary 293, Switzerland 123, VA Long Beach 200. Total 920, with 509 recorded positives and 411 negatives. Chart modes and every hospital’s missingness table match the Phase 1 audit. Switzerland accounts for 123 recorded cholesterol zeros and VA Long Beach for 49. The Cleveland discrepancy, feature tiers and unresolved anomalies remain visible. |
| Hospital Reliability | Automated view checks compared all 24 hospital, model and tier combinations with the frozen artifacts. They cover balanced accuracy, sensitivity, specificity, ROC-AUC, confusion matrices, thresholds, intervals and populations. Cleveland, Hungary, Switzerland and VA Long Beach show 77.6%, 66.7%, 75.6% and 60.7% for early Logistic Regression. The Switzerland warning identifies its eight negatives. |
| Experiment Registry | SQLite loads six pooled and 96 hospital records. Reimport preserves content, identities and timestamps. Model, tier, protocol and hospital filters use the verified records and show actual counts. Missing records can be restored by reimport. A row that disagrees with the reports produces an error before the table or details are displayed. A damaged SQLite database should be restored or replaced with a fresh local database. |
| Prediction Sandbox | The existing fitted model reproduced the Phase 3 holdout probability digest exactly. Fictional A and B use their documented defaults, produce different outputs, and clear the previous result after a preset change or reset. The tests guard against pipeline fitting during inference, validate inputs and check that inference never creates a registry. |

The overview now displays mean balanced accuracy as 70.1%, and the default pooled laboratory card shows 77.7%. Classification rates and their intervals use percentages. ROC-AUC, average precision, Brier score, log loss and thresholds retain decimal notation. The laboratory displays named settings and explains how the development folds selected the threshold. Full parameters and protocol text remain in the advanced panel.

The numeric form shows whole numbers for whole-number defaults, with fractional input support, units and bounds preserved. Reset recreates the form widgets to discard buffered edits in the browser. Chart hover labels use the report precision where useful and identify the 95% within-site bootstrap bounds. Plotly adjusts margins for axis labels. Registry comparison axes follow the selected metric’s units. The dashboard retains its original navigation, palette, component styles and research disclaimers.

The baseline suite passed 69 tests. The final suite passed **79 tests** with no skips in 33.00 seconds, including the replay of the existing fitted artifact. The earlier fixture that tests the explicit sandbox build still runs in a temporary directory. Neither that fixture nor the replay writes to the existing `models/` directory.

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.venv/phase51-acceptance
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m pip check
```

Ruff lint and format, strict MyPy and dependency validation passed. Git ignores the original CSV, model and manifest, SQLite database, local secrets and caches. The registry schema and records contain aggregate evidence only.

## Browser acceptance

Chromium opened all eight routes at 1600 × 900. Dataset Explorer, Hospital Reliability, the registry and the sandbox were also checked at 1366 × 768 and 390 × 844. The checks found no horizontal page overflow, controls or charts outside the main content, or Streamlit exceptions. Wide data tables retain their own horizontal scroll.

In the browser, the Dataset Explorer switched to class counts and Switzerland missingness. Hospital Reliability switched across the four hospitals, three model families and both tiers. Filtering the 102-record registry to Logistic Regression, Extended and pooled holdout returned one experiment. The automated checks cover every hospital/model/tier combination and corrupt or missing registry records.

The browser returned scores of 0.050 for Fictional A and 0.910 for Fictional B. Switching presets cleared the prior score. An age of 17 displayed the allowed range, and resetting an unsubmitted edit restored Fictional B’s defaults (64, 145 and 280) and cleared the output. HTTP and WebSocket traffic observed during this workflow went only to the loopback server.

| Capture | Viewport |
| --- | --- |
| [Dataset Explorer](../artifacts/phase5-1/dataset-desktop.png) | 1600 × 900 |
| [Hospital Reliability](../artifacts/phase5-1/reliability-desktop.png) | 1600 × 900 |
| [Model Laboratory settings](../artifacts/phase5-1/model-settings-laptop.png) | 1366 × 768 |
| [Filtered registry](../artifacts/phase5-1/registry-laptop.png) | 1366 × 768 |
| [Sandbox result](../artifacts/phase5-1/sandbox-narrow.png) | 390 × 844 |

Direct subpage visits still emit initial Streamlit health and host-configuration probe 404s before falling back to the root endpoints. The pages render after the fallback.

Phase 5.1 is suitable for local educational acceptance. Results remain retrospective, with no independent external cohort. SQLite was tested live and PostgreSQL was outside live acceptance. The browser checks use Chromium. Independent validation, broader accessibility and cross-browser testing remain appropriate Phase 6 items, subject to approval.
