# Data-quality investigation

## Evidence and comparison method

The combined input is the Phase 1 Kaggle v6 CSV, SHA-256 `574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae`. The independent reference is UCI archive SHA-256 `b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc`. The comparison reads the four `processed.*.data` files. It maps the combined CSV’s category labels to UCI codes, compares decimal values numerically, and equates empty cells with UCI’s processed question-mark marker. It counts occurrences of each complete 13-predictor profile **plus the original target**, so repeated occurrences remain visible. The source files contain no shared release-level patient key. [UCI archive](https://archive.ics.uci.edu/static/public/45/heart+disease.zip) · [UCI field dictionary](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names) · [Combined distribution](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

| Site | UCI records | Kaggle records | Exact within-site matches | UCI only | Kaggle only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cleveland | 303 | 304 | 303 | 0 | 1 |
| Hungary | 294 | 293 | 293 | 1 | 0 |
| Switzerland | 123 | 123 | 123 | 0 | 0 |
| VA Long Beach | 200 | 200 | 200 | 0 | 0 |

**The full multisets agree when site is omitted.** The one extra Cleveland profile exactly accounts for the one missing Hungary profile. The comparison establishes the release-level label discrepancy. Neither file identifies who changed the label, why it changed, or the true enrollment site. Keep the supplied label and disclose the one-record uncertainty in both affected hospital evaluations.

## Findings and handling decisions

| Observation | Confirmed evidence | Uncertainty, policy and validity risk |
| --- | --- | --- |
| Two excess non-ID matches | One pair in Hungary and one at VA Long Beach. Both pairs already occur in the corresponding UCI processed files. Each pair shares its original outcome. Comparing all 13 predictors alone finds the same two pairs and zero cross-site pairs. | Shared identity is unverified. Retain all records and assign identical predictor profiles to one split group. Repetition can still give a profile extra weight within a partition. |
| Cholesterol recorded as zero | All 123 Swiss records and 49 VA records have zero in both releases. Checking field 12 in the long-form UCI files confirms 123 Swiss zeros and 49 VA zeros. VA also contains seven separate `-9` entries, consistent with the original dictionary’s missing marker. | The zeros predate Kaggle. Their clinical or administrative meaning remains undocumented. Preserve them in the primary run. Repeat with **only cholesterol zeros** masked before fold-local imputation. In the latter run a held-out Switzerland has no observed cholesterol at all. Imputation supplies a training-population value with no Swiss measurement evidence. |
| Zero resting blood pressure | One VA record has zero in both processed releases. The field denotes resting pressure on admission. | No verified replacement or field-specific sentinel rule. Preserve the flag and value in both branches. A model could exploit a recording artifact. |
| Negative `oldpeak` | UCI and Kaggle contain 11 in Switzerland and one at VA. The observed minimum is −2.6. UCI describes exercise-relative ST depression. | The collection/sign convention for those entries remains unclear. Preserve their sign. Clipping would discard source information. |
| Sparse measurements | UCI already has 611 missing `ca`, 486 missing `thal` and 309 missing `slope`. The reassigned profile accounts for the small site-count differences between sources. In Kaggle, `ca` is missing in 5/304 Cleveland, 290/293 Hungary, 118/123 Switzerland and 198/200 VA records. | The files do not reliably distinguish an unordered test, an unavailable result and a recording omission. Retain the cohort. Exclude these fields from the early tier. In the extended tier, fold-local imputation cannot recover the measurement mechanism. |

The near-match screen compared every distinct pair across all 13 predictors. It flagged exactly one differing field with at least nine jointly observed predictors, counting a mismatched missing marker as a difference. One VA pair met the rule, differing on resting ECG. Its two records are on the same side of the fixed pooled split. No cross-site pair met the rule. Similarity provides insufficient evidence to assign a shared patient identity. This screen cannot detect every repeat, particularly profiles with several altered measurements or extensive missingness.

## Reproduction

Download the linked UCI ZIP into the ignored raw directory. The script reads the archive in place and prints aggregate comparisons. Preserve the named Kaggle CSV as the experiment input.

```powershell
Invoke-WebRequest "https://archive.ics.uci.edu/static/public/45/heart+disease.zip" -OutFile data/raw/uci-reference.zip
(Get-FileHash data/raw/uci-reference.zip).Hash
.\.venv\Scripts\python.exe scripts/compare_sources.py --input data/raw/heart_disease_uci.csv --uci-zip data/raw/uci-reference.zip
```

The source comparison explains encoding and release differences. It does not validate clinical histories or establish measurement times. [Phase 1 audit](dataset-audit.md) · [Evaluation protocol](../docs/evaluation.md)
