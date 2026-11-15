# Pooled model comparison

Outcome 1 means original num 1–4. Development used 735 records (407 positives). The pooled holdout contained 185 records (102 positives and 83 negatives). Every row below uses the as-recorded cholesterol policy.

## Training-only selection

AP is the mean of five validation-fold average precision scores. SD describes variation between folds. OOF balanced accuracy uses the threshold chosen on those same out-of-fold probabilities and is optimistic as a performance estimate.

| Tier | Model | Selected C or depth | Mean AP | Fold SD | Threshold | OOF balanced accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| early | logistic | 1.0 | 0.859 | 0.022 | 0.450 | 0.777 |
| early | forest | 4 | 0.859 | 0.030 | 0.500 | 0.794 |
| early | xgboost | 2 | 0.853 | 0.033 | 0.550 | 0.797 |
| extended | logistic | 0.1 | 0.903 | 0.026 | 0.450 | 0.828 |
| extended | forest | 4 | 0.898 | 0.027 | 0.600 | 0.816 |
| extended | xgboost | 2 | 0.903 | 0.013 | 0.500 | 0.832 |

Family selected by mean CV AP for each tier: early = logistic, extended = logistic.

## Pooled holdout at locked thresholds

| Tier | Model | Accuracy | Balanced accuracy | 95% interval | Precision | Sensitivity | Specificity | F1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early | logistic | 0.789 | 0.777 | 0.717–0.836 | 0.765 | 0.892 | 0.663 | 0.824 | 55 | 28 | 11 | 91 |
| early | forest | 0.768 | 0.760 | 0.695–0.824 | 0.766 | 0.833 | 0.687 | 0.798 | 57 | 26 | 17 | 85 |
| early | xgboost | 0.784 | 0.777 | 0.714–0.836 | 0.782 | 0.843 | 0.711 | 0.811 | 59 | 24 | 16 | 86 |
| extended | logistic | 0.832 | 0.822 | 0.768–0.876 | 0.803 | 0.922 | 0.723 | 0.858 | 60 | 23 | 8 | 94 |
| extended | forest | 0.816 | 0.820 | 0.765–0.873 | 0.870 | 0.784 | 0.855 | 0.825 | 71 | 12 | 22 | 80 |
| extended | xgboost | 0.816 | 0.810 | 0.753–0.867 | 0.809 | 0.873 | 0.747 | 0.840 | 62 | 21 | 13 | 89 |

## Pooled holdout probabilities

| Tier | Model | ROC-AUC | Average precision | Brier | Log loss |
| --- | --- | --- | --- | --- | --- |
| early | logistic | 0.869 | 0.893 | 0.148 | 0.452 |
| early | forest | 0.875 | 0.895 | 0.150 | 0.464 |
| early | xgboost | 0.861 | 0.876 | 0.151 | 0.461 |
| extended | logistic | 0.910 | 0.921 | 0.121 | 0.388 |
| extended | forest | 0.910 | 0.911 | 0.131 | 0.424 |
| extended | xgboost | 0.906 | 0.907 | 0.124 | 0.395 |

The interval resamples whole duplicate groups 2,000 times with the fitted model and threshold fixed. All six evaluations share the same holdout. Small score differences cannot establish superiority across hospitals. Diagnostic measurements in the extended tier and site-dependent measurement patterns limit interpretation. Probabilities refer to the recorded endpoint in this dataset.

## Runtime

| Tier | Model | 10 CV fits (s) | Final refit (s) |
| --- | --- | --- | --- |
| early | logistic | 0.145 | 0.026 |
| early | forest | 1.202 | 0.106 |
| early | xgboost | 0.343 | 0.080 |
| extended | logistic | 0.151 | 0.011 |
| extended | forest | 1.237 | 0.111 |
| extended | xgboost | 0.294 | 0.024 |

All fits ran serially on CPU with one compute thread. Runtime varies by machine. The companion JSON records fold and prediction digests, candidate scores, estimator settings, package versions, seeds and process timings.
