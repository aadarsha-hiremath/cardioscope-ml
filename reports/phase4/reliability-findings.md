# Reliability findings

Early Logistic Regression achieved its lowest balanced accuracy at VA Long Beach (0.607) and its highest at Cleveland (0.776). Pooling these sites obscures the balance of false positives and false negatives. Difficulty depends on which error matters.

| Site | Prevalence | BA (95%) | False negative / positive cases | False positive / negative cases | High-confidence FN / FP | AP − prevalence | CITL | Calibration slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cleveland | 0.457 | 0.776 (0.728–0.822) | 27 / 139 | 42 / 165 | 1 / 0 | 0.321 | -0.281 | 1.080 |
| Hungary | 0.362 | 0.667 (0.613–0.719) | 61 / 106 | 17 / 187 | 3 / 0 | 0.346 | -0.286 | 0.966 |
| Switzerland | 0.935 | 0.756 (0.568–0.945) | 13 / 115 | 3 / 8 | 0 / 1 | 0.022 | 2.120 | 0.836 |
| VA Long Beach | 0.745 | 0.607 (0.542–0.677) | 15 / 149 | 35 / 51 | 0 / 5 | 0.120 | 0.270 | 0.596 |

High-confidence errors use p ≤ 0.10 for false negatives and p ≥ 0.90 for false positives. Hungary contains 106 positives among 293 records (36.2%); Switzerland contains 115 among 123 (93.5%). A high Swiss AP therefore starts from a high prevalence baseline. Eight negatives provide little precision for specificity, even when sensitivity looks stable.

The training populations led to different operating points. Hungary's high threshold favored specificity and missed over half its recorded positives. VA Long Beach had the lowest specificity. Site, threshold and case mix change together in this design, so their individual contributions cannot be isolated.

## Do extended features or boosting help?

These within-site contrasts use the primary raw, as-recorded policy and each model’s own locked cutoff. Paired intervals reuse identical patient-group draws. Intervals excluding zero give conditional evidence within that site; the comparisons have no multiplicity correction.

| Site | Extended LR − early LR BA (95%) | Early XGB − early LR BA (95%) | Early LR/RF, LR/XGB, RF/XGB differing calls |
| --- | --- | --- | --- |
| Cleveland | 0.013 (-0.031–0.060) | -0.019 (-0.059–0.022) | 46 / 46 / 30 |
| Hungary | 0.092 (0.039–0.145) | -0.022 (-0.069–0.028) | 21 / 32 / 17 |
| Switzerland | -0.057 (-0.090–-0.022) | -0.013 (-0.035–0.009) | 8 / 7 / 11 |
| VA Long Beach | 0.070 (-0.000–0.142) | 0.035 (-0.030–0.101) | 17 / 27 / 12 |

Paired intervals supported higher extended-tier LR balanced accuracy in Hungary and lower balanced accuracy in Switzerland. Early XGBoost had a positive lower bound against early LR in 0 of 4 sites. The data offer no uniform boosting advantage.

## Measurement availability and probability reliability

| Site | Missing ca % train → test | Missing thal % train → test | Missing slope % train → test | Cholesterol zeros | ΔBrier sigmoid − raw (early LR) | Mean \|Δp\| masking zeros (early LR) |
| --- | --- | --- | --- | --- | --- | --- |
| Cleveland | 98.4 → 1.6 | 78.4 → 1.0 | 50.0 → 0.3 | 0 | -0.003 | 0.051 |
| Hungary | 51.2 → 99.0 | 35.2 → 90.4 | 19.1 → 64.5 | 0 | -0.002 | 0.055 |
| Switzerland | 61.9 → 95.9 | 54.5 → 42.3 | 36.6 → 13.8 | 123 | -0.001 | 0.010 |
| VA Long Beach | 57.4 → 99.0 | 44.4 → 83.0 | 28.8 → 51.0 | 49 | -0.010 | 0.054 |

In Switzerland the primary model assigned a mean probability of 0.710 against an observed fraction of 0.935. The sigmoid learned at the other hospitals left substantial underprediction. At VA Long Beach a slope below one and high-confidence false positives are consistent with excessive probability spread. Calibration also sometimes worsened: the extended forest had higher Brier score after sigmoid calibration in Switzerland and VA Long Beach. See the bin counts and loss comparisons in the [calibration analysis](calibration-analysis.md).

Masking cholesterol zeros changed 71 of 920 primary-model classifications after retraining. 3 of four sites crossed the prespecified sensitivity flag. The Swiss primary model changed no classifications, despite all its cholesterol values being zero. Changes at Cleveland and Hungary arose through training and cutoff changes at other sites. Neither policy establishes the original recording convention.

## Scope of evidence

The project contribution is the frozen transfer design, duplicate-aware site boundaries, cross-fitted thresholds and calibration, paired sensitivity experiments and reproducible failure analysis. The original records come from the UCI investigators via the Kaggle combined release. The three classification algorithms are established methods. [Data provenance](../../docs/provenance.md) · [Frozen protocol](../../docs/phase4-protocol.json)

The study estimates transfer among four historical sources using a recorded binary outcome. Previously pooled development informed the model and feature choices. Inner training sets overlap and site practices can recur across records. Bootstrap intervals omit fitting and between-hospital uncertainty. A genuinely untouched external cohort with known measurement timing and independent outcome ascertainment is needed to assess prospective generalization. These results cannot establish a clinical screening or treatment use.
