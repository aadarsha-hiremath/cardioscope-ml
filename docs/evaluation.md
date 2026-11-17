# Preprocessing and evaluation protocol

The research endpoint is recorded angiographic disease, with original `num` 1–4 mapped to positive and 0 mapped to negative. The loader retains `num`, `id` and hospital metadata alongside the derived outcome. Every transform uses an explicit predictor allowlist.

## Feature availability by experimental tier

The **early** question assumes history, admission observations and routine blood-test results are available. The **extended** question assumes access to the additional diagnostic results listed below. Neither tier specifies a clinically validated screening protocol. The combined extract lacks timestamps for establishing the sequence of tests, referrals and outcome ascertainment. [UCI field dictionary](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names)

| Tier | Fields | Representation and availability assumption |
| --- | --- | --- |
| Both | `age`, `sex`, `cp` | Numeric age, categorical recorded sex and chest-pain type. Symptom classification may depend on clinical assessment. |
| Both | `trestbps`, `chol`, `fbs` | Numeric admission resting pressure and serum cholesterol, categorical fasting blood glucose above the source threshold. The latter two require laboratory results. |
| Extended | `restecg` | Categorical result from a resting ECG. This procedure is excluded from the defined early setting. |
| Extended | `thalch`, `exang`, `oldpeak`, `slope` | Exercise testing produces peak heart rate (numeric), exertional angina (categorical), exercise-relative ST depression (numeric) and peak ST slope (categorical). |
| Extended | `ca` | Numeric count of vessels visualized by fluoroscopy. This diagnostic workup is close to the angiographic endpoint. The temporal order in these records is unverified. |
| Extended | `thal` | Categorical normal, fixed-defect or reversible-defect result. These labels are consistent with thallium perfusion imaging. The precise test protocol and timing for each patient are unavailable. |

The original research group describes fluoroscopy and thallium perfusion imaging in angiography-referred populations. This supports the extended-tier interpretation. It cannot establish individual test timing across these four databases. Referral selection and measurement availability can also affect early-tier results. [Detrano et al., 1986](https://pubmed.ncbi.nlm.nih.gov/3706156/)

## Transformation contract

1. Load one byte snapshot. Keep source row order as the coordinate system for partitions. Require unique numeric IDs, a recognized site and a valid outcome. Parse numeric fields without learning statistics. Normalize blank and question-mark fields to missing. Preserve negative `oldpeak` and the recorded zero blood pressure.
2. Construct duplicate groups from all 13 normalized predictors before selecting a tier or recoding cholesterol. Target and site are excluded from the comparison. Two missing values in the same field compare equal. The early tier therefore does not group people solely because their six early measurements coincide.
3. Run the primary `as_recorded` policy and the prespecified `zero_as_missing` sensitivity on identical partitions. The latter masks only literal numeric zero in `chol`. The policies describe competing measurement assumptions. Report both, regardless of which performs better.
4. For each training partition, instantiate a fresh scikit-learn pipeline. Its `ColumnTransformer` fits a median imputer and standard scaler on numeric training columns. A wholly missing numeric training column retains its position with a zero fallback and no learned variation. Inspect this case before interpreting coefficients or distances.
5. For categorical columns, use a fixed missing label and fit a one-hot vocabulary on the training partition. A previously unseen category, including missingness first seen in evaluation, maps to an all-zero block. The fixed missing label preserves missing-test information when observed during training. This information can proxy the source hospital.
6. Use only `transform` on held-out rows. Return named train/test matrices retaining source row indices. The JSON record stores the release digest, scikit-learn version, split-position digests, masks and missing counts, numeric imputer and scaler statistics, learned categories and output names. Restoring the release and lockfile reproduces the transformations. [scikit-learn mixed-type transformers](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)

## Two evaluation questions

**Conventional, pooled holdout.** The first fold of a shuffled five-fold stratified group split is held out, with seed 2026. The strata combine hospital and binary outcome, and exact predictor groups stay together. Stratification is approximate at group boundaries. On the verified release this yields 735 training records and 185 evaluation records, with 102 positives and 83 negatives in evaluation. The test site counts are 61 Cleveland, 59 Hungary, 25 Switzerland and 40 VA. The fixed rule avoids searching for an attractive holdout.

**Transfer to one complete hospital.** Run four independent experiments. Each held-out hospital supplies every one of its records for evaluation. If any of its exact predictor profiles also occur elsewhere, remove every matching occurrence from that fold’s training pool. Preserve the full test site and record the training exclusion count. The current release requires zero such exclusions. The validation utility checks row coverage, index disjointness, hospital isolation and duplicate-group isolation before fitting.

| Held-out hospital | Train | Evaluation | Positive | Negative |
| --- | ---: | ---: | ---: | ---: |
| Cleveland | 616 | 304 | 139 | 165 |
| Hungary | 627 | 293 | 106 | 187 |
| Switzerland | 797 | 123 | 115 | 8 |
| VA Long Beach | 720 | 200 | 149 | 51 |

Four hospitals provide four observed transfer settings. Switzerland’s eight negative records mean one changed negative classification moves measured specificity by 12.5 percentage points. A pooled estimate can conceal site differences in precision or probability reliability. Missingness and selection differ across sites, and one combined-release site label is disputed. No partition removes those limitations. [Source investigation](../reports/data-quality-investigation.md)

## Model development and transfer evaluation

The pooled Phase 3 comparison is complete. Commit `6a0951e` froze the six pooled configurations and thresholds before evaluation. The Phase 4 brief specified these configurations as fixed candidates for a retrospective transfer stress test. Commit `7103b15` froze its protocol before any site scores were examined. The experiment followed the frozen plan without a protocol deviation. [Pooled comparison](../reports/phase3/pooled-both.md) · [Phase 4 protocol](phase4-protocol.json) · [Hospital comparison](../reports/phase4/hospital-comparison.md)

- Freeze the feature tiers, zero policies, metrics, tuning budgets and use of each holdout before training. Start with Logistic Regression as the baseline, then compare Random Forest and XGBoost on the same eligible rows and groups.
- For pooled model selection, run group-preserving stratified CV entirely within the 735 training records. Start each inner fold from its eligible **untransformed** features and put a fresh `build_preprocessor(tier)` and estimator inside one pipeline. Reusing the outer-preprocessed training matrix inside CV would leak inner validation statistics.
- Each Phase 4 experiment holds one of the three training hospitals out in turn. Its two-site training subset supplies a new preprocessing pipeline and estimator. Purge matching validation groups from that subset. Use the three sets of out-of-fold predictions to fit the sigmoid and select the raw threshold. Map the threshold through the fitted monotone sigmoid for the secondary probability analysis. Refit the base pipeline on all eligible training sites before scoring the outer hospital.
- The two designs reuse the same people for different research questions. Freeze choices for both designs before opening either set of scores. Complete the pooled final evaluation before using its records in hospital experiments. Run all four outer folds under one frozen protocol. Using one site’s results to retune another would invalidate the cross-hospital comparison.
- Report per-site class counts, AUROC and average precision with prevalence context. Evaluate probability reliability using Brier score, log loss and calibration curves with bin counts. Report uncertainty where class support permits. Any scores used to revise the protocol become development evidence for that revision.
