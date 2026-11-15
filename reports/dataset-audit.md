# CardioScope ML dataset audit

920 records × 16 columns. Matches the recorded Kaggle version 6 checksum.

Input SHA-256 `574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae`.

## Validation

All structural and domain checks pass.

## Recorded outcomes

| Original num | Records |
| --- | --- |
| 0 | 411 |
| 1 | 265 |
| 2 | 109 |
| 3 | 107 |
| 4 | 28 |

| Derived binary target | Records |
| --- | --- |
| 0 | 411 |
| 1 | 509 |

Missing or invalid targets: 0. Positive means num is 1–4. Prevalence uses valid targets within each hospital.

## Hospital populations

| Hospital | Records | Valid targets | Positive | Positive (%) | Median age |
| --- | --- | --- | --- | --- | --- |
| Cleveland | 304 | 304 | 139 | 45.724 | 55.5 |
| Hungary | 293 | 293 | 106 | 36.177 | 49.0 |
| Switzerland | 123 | 123 | 115 | 93.496 | 56.0 |
| VA Long Beach | 200 | 200 | 149 | 74.5 | 60.0 |

## Field inventory and missingness

The loader retains CSV text. Expected types describe the values checked during the audit. An empty cell or a literal question mark counts as missing.

| Field | Storage | Expected | Role | Missing | Missing (%) |
| --- | --- | --- | --- | --- | --- |
| id | string | integer | identifier | 0 | 0.0 |
| age | string | integer | candidate predictor | 0 | 0.0 |
| sex | string | category | candidate predictor | 0 | 0.0 |
| dataset | string | category | hospital metadata | 0 | 0.0 |
| cp | string | category | candidate predictor | 0 | 0.0 |
| trestbps | string | integer | candidate predictor | 59 | 6.413 |
| chol | string | integer | candidate predictor | 30 | 3.261 |
| fbs | string | boolean | candidate predictor | 90 | 9.783 |
| restecg | string | category | candidate predictor | 2 | 0.217 |
| thalch | string | integer | candidate predictor | 55 | 5.978 |
| exang | string | boolean | candidate predictor | 55 | 5.978 |
| oldpeak | string | real | candidate predictor | 62 | 6.739 |
| slope | string | category | candidate predictor | 309 | 33.587 |
| ca | string | integer | candidate predictor | 611 | 66.413 |
| thal | string | category | candidate predictor | 486 | 52.826 |
| num | string | integer | original target | 0 | 0.0 |

## Missingness within each hospital

| Field | Cleveland | Hungary | Switzerland | VA Long Beach |
| --- | --- | --- | --- | --- |
| id | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| age | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| sex | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| dataset | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| cp | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| trestbps | 0 (0.0%) | 1 (0.341%) | 2 (1.626%) | 56 (28.0%) |
| chol | 0 (0.0%) | 23 (7.85%) | 0 (0.0%) | 7 (3.5%) |
| fbs | 0 (0.0%) | 8 (2.73%) | 75 (60.976%) | 7 (3.5%) |
| restecg | 0 (0.0%) | 1 (0.341%) | 1 (0.813%) | 0 (0.0%) |
| thalch | 0 (0.0%) | 1 (0.341%) | 1 (0.813%) | 53 (26.5%) |
| exang | 0 (0.0%) | 1 (0.341%) | 1 (0.813%) | 53 (26.5%) |
| oldpeak | 0 (0.0%) | 0 (0.0%) | 6 (4.878%) | 56 (28.0%) |
| slope | 1 (0.329%) | 189 (64.505%) | 17 (13.821%) | 102 (51.0%) |
| ca | 5 (1.645%) | 290 (98.976%) | 118 (95.935%) | 198 (99.0%) |
| thal | 3 (0.987%) | 265 (90.444%) | 52 (42.276%) | 166 (83.0%) |
| num | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

## Numeric summaries

Counts cover finite, representable values of the expected type, including flagged values. The mean and median include recorded zeros.

| Field | Count | Min | Median | Mean | Max |
| --- | --- | --- | --- | --- | --- |
| id | 920 | 1.0 | 460.5 | 460.5 | 920.0 |
| age | 920 | 28.0 | 54.0 | 53.5109 | 77.0 |
| trestbps | 861 | 0.0 | 130.0 | 132.1324 | 200.0 |
| chol | 890 | 0.0 | 223.0 | 199.1303 | 603.0 |
| thalch | 865 | 60.0 | 140.0 | 137.5457 | 202.0 |
| oldpeak | 858 | -2.6 | 0.5 | 0.8788 | 6.2 |
| ca | 309 | 0.0 | 0.0 | 0.6764 | 3.0 |
| num | 920 | 0.0 | 1.0 | 0.9957 | 4.0 |

## Categorical values

| Field | Observed label | Records | Expected label |
| --- | --- | --- | --- |
| sex | Female | 194 | True |
| sex | Male | 726 | True |
| dataset | Cleveland | 304 | True |
| dataset | Hungary | 293 | True |
| dataset | Switzerland | 123 | True |
| dataset | VA Long Beach | 200 | True |
| cp | asymptomatic | 496 | True |
| cp | atypical angina | 174 | True |
| cp | non-anginal | 204 | True |
| cp | typical angina | 46 | True |
| fbs | FALSE | 692 | True |
| fbs | TRUE | 138 | True |
| restecg | lv hypertrophy | 188 | True |
| restecg | normal | 551 | True |
| restecg | st-t abnormality | 179 | True |
| exang | FALSE | 528 | True |
| exang | TRUE | 337 | True |
| slope | downsloping | 63 | True |
| slope | flat | 345 | True |
| slope | upsloping | 203 | True |
| thal | fixed defect | 46 | True |
| thal | normal | 196 | True |
| thal | reversable defect | 192 | True |

## Duplicates and identifiers

Counts measure extra occurrences after the first. Field comparisons use exact cell text. The identifier check compares numeric values and excludes invalid or missing IDs.

| Comparison | Excess records |
| --- | --- |
| all_fields | 0 |
| excluding_id | 2 |
| repeated_valid_identifier | 0 |

## Values for source review

| Field | Reason | Records | Cleveland | Hungary | Switzerland | VA Long Beach |
| --- | --- | --- | --- | --- | --- | --- |
| trestbps | zero | 1 | 0 | 0 | 0 | 1 |
| chol | zero | 172 | 0 | 0 | 123 | 49 |
| oldpeak | negative ST depression | 12 | 0 | 0 | 11 | 1 |

Review nonpositive age, pressure, cholesterol, heart rate or ID and negative oldpeak against collection documentation. A flag alone does not establish a missing-value code.

## Implications for evaluation

Differences in site prevalence can change precision and the interpretation of a pooled calibration curve. Age mix and missing tests can also identify a collection site indirectly. Compare within-site results and report denominators for each held-out site. Matching non-ID fields flag records for linkage review before splitting.
