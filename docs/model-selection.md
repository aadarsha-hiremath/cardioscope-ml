# Pooled comparison protocol

## Decisions fixed before pooled evaluation

Use the Phase 2 pooled split, seed 2026, original site labels and as-recorded cholesterol for all six experiments. Early assessment is primary. The extended diagnostic tier is secondary. Both include laboratory measurements. The endpoint is the recorded binary label. Class 1 combines original outcomes 1–4. Preserve zero blood pressure, signed oldpeak and duplicate groups. Reserve the cholesterol sensitivity experiment for Phase 4.

Use the same five shuffled stratified group folds of the 735 development rows for every candidate. Stratify by hospital and binary outcome. Exact normalized 13-predictor groups remain intact. Fit a new Phase 2 preprocessor and estimator per candidate per fold. The preprocessor uses numeric median imputation and standardization, a categorical missing label and training-only one-hot encoding.

| Model | Fixed settings | Two candidates in tie-break order |
| --- | --- | --- |
| Logistic Regression | L2 via lbfgs, intercept, tolerance 0.0001, maximum 2,000 iterations | C 0.1, 1.0 |
| Random Forest | 160 bootstrap trees, Gini, square-root feature sampling, minimum leaf size 4 | Maximum depth 4, 8 |
| XGBoost | CPU histogram trees, binary logistic objective, 120 rounds, rate 0.05, minimum child weight 1, L2 1, L1 0, full row and column sampling, log-loss monitor | Maximum depth 2, 3 |

Seed all estimators with 2026. Use one compute thread and serial fits. Use natural training prevalence, unit observation weights and unit positive-class weight. No resampling, early stopping or calibration. All other estimator defaults follow the locked package versions.

Choose the candidate with the highest **unweighted mean validation-fold average precision (AP)** within each model and tier. AP emphasizes ranking positives and depends on prevalence. Break exact ties by candidate order. The primary family recommendation uses early-tier mean CV AP, with family order Logistic Regression, Random Forest, XGBoost for an exact tie. Retain and evaluate all six. Report fold APs and their sample standard deviation as descriptive fold variation.

For each selected configuration, use its out-of-fold probabilities to choose a threshold from 0.20 through 0.80 in 0.05 steps. Maximize balanced accuracy, giving sensitivity and specificity equal weight. Break ties by distance to 0.50, then the lower threshold. Use the inclusive rule probability ≥ threshold. Fit each selected pipeline on the 735 development rows. Apply each fixed threshold to the same 185 evaluation rows. Out-of-fold metrics used for configuration and threshold selection are development evidence and can be optimistic.

Compute a percentile 95% interval for holdout balanced accuracy by resampling whole duplicate groups with replacement 2,000 times, seed 2026. Use common draws across the six models and discard a draw if it lacks either class. This quantifies held-out sampling variation conditional on the fitted model, threshold and observed site mixture. It excludes fitting uncertainty and transfer to new hospitals. AP is non-interpolated average precision. JSON metrics retain six fractional digits, Markdown displays three, and all selections use full precision.

## Phase 4 boundary

Archive pooled selections and cutoffs as the Phase 3 reference. Do not apply these fitted pipelines to a hospital whose records helped fit them. Freeze the three families, grids, tie rules, feature definitions and primary as-recorded policy above before starting any site evaluation. Under the Phase 2 complete-hospital protocol, repeat selection using only the remaining hospitals, with inner hospital holdouts and training-side duplicate purges. Select thresholds from training-only out-of-fold probabilities under the same rule. Predeclare calibration and cholesterol sensitivity procedures before opening outer-site results. Report all four sites under one protocol. Pooled and hospital experiments reuse records in different roles and do not constitute independent external validations.
