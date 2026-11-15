# Proposed evaluation protocol

**Decision to review before making partitions.** Pooled validation estimates performance for another record drawn from the same mix of sites. Holding out a whole hospital tests transfer to one of four observed populations. Keep these as separate questions and identify each evaluation population in reported results.

## Phase 2 — define the prediction task and assess split feasibility

1. Specify the prediction time and intended population. Review how each candidate field was collected relative to angiographic outcome ascertainment. For example, `ca` requires fluoroscopy and may be unavailable at an earlier prediction time. Exclude post-outcome or target-derived information. Keep a predictor allowlist that excludes `id`, `dataset`, `num` and the binary target.
2. Review the two non-ID matches, compare predictor-only matches across sites and across outcomes, and investigate any available patient linkage. Predefine how to group plausible repeated observations so a group cannot straddle training and evaluation. If cross-site linkage conflicts with site holdout, document and resolve the conflict before splitting.
3. Assess support for each site × outcome cell and each candidate measurement. Switzerland has only eight negative outcomes. Predefine which metrics and calibration summaries can be interpreted at that sample size. Record the uncertain meanings of zeros and sign conventions. Specify a primary policy and sensitivity analyses before inspecting fitted performance.
4. Propose a grouped, target-stratified final holdout within sites for the same-mixture question. Check feasibility after accounting for duplicates and per-site class counts. Freeze its membership, grouping rules, seed and CSV digest only after review. Phase 1 creates no partition files.

## Phase 3 — develop models on the development pool

- Compare Logistic Regression as the baseline with Random Forest and XGBoost. Use the same eligible records and folds across candidates. For conventional validation, use group-respecting stratified folds on development data. Fit imputation, scaling, encoding, feature selection and any resampling entirely within training folds.
- For transfer evaluation, leave one hospital’s development records out at a time. All fitting, tuning and calibration for that outer fold use only the remaining hospitals. Use inner site-aware folds where class support permits. Fix search spaces, thresholds and comparison rules in advance. Review performance for each outer hospital separately.
- Fit probability calibration only on out-of-fold predictions or a distinct calibration subset within the training portion. Keep calibrator selection inside the inner validation procedure. Compare calibrated and uncalibrated predictions on the same outer observations.

## Phase 4 — evaluate reliability and report uncertainty

Freeze model choices, preprocessing, probability calibration and thresholds using development evidence. Fit the selected pipeline on eligible development data and evaluate the untouched final pooled holdout once. Report the hospital-held-out results separately. Choosing a winner after inspecting outer-site scores makes those scores development evidence for that choice. An independent hospital cohort would be needed for a fresh external test.

Report outcome counts and prevalence beside AUROC, average precision and any threshold-based sensitivity or specificity. Assess probability reliability with Brier score, log loss and calibration curves with bin counts. Use group-aware uncertainty intervals where feasible. A hospital with few negatives can yield unstable specificity even when pooled metrics appear precise. Four observed hospitals provide limited evidence about variation across future hospitals.

Missingness can reveal the site indirectly even when the site label is excluded. Median ages range from 49 in Hungary to 60 at VA Long Beach. A change in measurement practice, population mix or baseline prevalence can affect both discrimination and calibration. Compare site-specific metrics, a site-weighted summary with explicit weights, and prespecified sensitivity analyses for ambiguous zeros and sparse fields. The dataset audit does not identify the causes of site differences.
