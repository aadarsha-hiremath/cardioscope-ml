# CardioScope ML — Heart Disease Classification & Model Reliability Lab

> Educational machine learning research prototype. Not a medical device, diagnostic system, or clinically validated risk assessment tool.

**Research question.** How well do heart disease classification models generalize across different hospital populations, and how reliable are their predicted probabilities?

**Phase 1 delivers a reproducible dataset audit.** The combined CSV contains 920 records, 16 columns and four hospital sources. The audit preserves the original outcome, derives a binary label in memory, checks data quality and produces aggregate reports on CPU. Model fitting, partitioning and probability calibration are planned work. [Markdown audit](reports/dataset-audit.md) · [JSON summary](reports/dataset-audit.json)

## Why compare hospitals?

Hungary has 106 positive outcomes among 293 records. Switzerland has 115 among 123, leaving only eight negatives. The vessel-count field (`ca`) is missing in 5 of 304 Cleveland records and 198 of 200 VA Long Beach records. A random split can reproduce these collection patterns on both sides of the split. Holding out a hospital asks whether a model transfers to a population with a different outcome frequency and different tests available.

## Reproduce the audit on Windows

From the repository root, create a local Python 3.12 environment. If Python 3.12 is available outside the Windows launcher, use that interpreter’s executable in place of `py -3.12`. An existing project `.venv` can be used directly from the second command. The constraints file records the dependency versions used for this audit on Windows.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements-dev.lock -e ".[dev]"
New-Item -ItemType Directory -Force data/raw | Out-Null
```

Download version 6 of Redwan Sony’s combined UCI Heart Disease Data from Kaggle. Extract `heart_disease_uci.csv` into `data/raw/` without opening and resaving it. UCI’s 303-instance listing describes the Cleveland subset. The checksum below identifies the exact combined CSV used here. [Kaggle dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) · [Provenance and attribution](docs/provenance.md)

```powershell
(Get-FileHash -Algorithm SHA256 data/raw/heart_disease_uci.csv).Hash
.\.venv\Scripts\cardioscope-audit.exe --input "data/raw/heart_disease_uci.csv" --output-dir reports
```

Expected SHA-256 is `574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae`. The CLI accepts a quoted path containing spaces. The JSON field `matches_kaggle_v6` verifies release identity independently of schema validation. Running the command twice on the same bytes reproduces the two report files. Use a separate output directory when investigating another input.

| Exit code | Meaning |
| --- | --- |
| 0 | Reports written and schema, cohort and value checks passed. Review flags can still be present. |
| 1 | Reports written with validation errors, such as unexpected sites or invalid outcomes. |
| 2 | Input, CSV structure, arguments or report I/O prevented a successful run. |

### Run the quality checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check src tests
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m pip check
```

Tests generate invented records in temporary directories. The source CSV is needed only to reproduce the research reports.

## Architecture and field roles

| Location | Responsibility |
| --- | --- |
| `src/cardioscope/data/` | CSV structure and byte checksum, field definitions, validation, derived labels and aggregate statistics |
| `src/cardioscope/cli.py`, `report.py` | Explicit input path, exit status and deterministic Markdown/JSON rendering |
| `data/raw/` | Local, Git-ignored source files |
| `tests/`, `reports/`, `docs/` | Synthetic regression tests, real-data audit evidence and research design |

`id` identifies a record. `dataset` identifies its hospital for grouped evaluation. The original target `num` ranges from 0 to 4. The derived binary outcome is 0 for no disease recorded and 1 for disease recorded (original classes 1–4). Both targets, the identifier and hospital metadata are excluded from the 13 candidate predictors. Predictor eligibility still depends on when a measurement would be available.

## Limitations and planned experiments

These historical records describe selected hospital populations. Their positive fractions describe this sample. Four sites and small within-site class counts limit estimates of transfer and calibration. The combined release has unresolved zero encodings and matching records under different IDs. This audit provides no estimate of predictive performance.

Phase 2 will review predictor timing, duplicate grouping, missing-value rules and split feasibility. Later phases will compare Logistic Regression, Random Forest and XGBoost under pooled and hospital-held-out evaluation, including discrimination and probability calibration. The proposed protocol keeps final evaluation data separate from tuning. [Evaluation plan](docs/evaluation.md)

**Original data citation.** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. [DOI 10.24432/C52P4X](https://doi.org/10.24432/C52P4X)
