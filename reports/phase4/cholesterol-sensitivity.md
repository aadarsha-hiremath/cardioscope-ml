# Sensitivity to recorded cholesterol zeros

Both policies retain the same records and partitions. Masking zeros affects cholesterol alone. The complete pipeline, training-site cutoff and sigmoid are refitted for each policy. Changes below compare masked minus recorded for raw probabilities. The frozen descriptive flag marks mean absolute probability change or absolute balanced accuracy change of at least 0.05. This is an operational threshold for this experiment.

All 123 Swiss records and 49 of 200 VA records contain a cholesterol zero. Cleveland and Hungary contain none. A recorded zero therefore conveys site information even though site is excluded from predictors. Masking can also change predictions at sites without zeros through changes in the training distribution. No Swiss cholesterol measurement remains under masking; Swiss imputation comes entirely from the other hospitals.

## Cleveland

| Tier | Model | Mean \|Δp\| | Changed calls | ΔBA | ΔAUC | ΔBrier | Δlog loss | Flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early | LR | 0.051 | 23 | -0.010 | 0.010 | 0.001 | 0.001 | yes |
| early | RF | 0.040 | 12 | 0.002 | -0.000 | 0.006 | 0.012 | no |
| early | XGB | 0.038 | 33 | -0.007 | -0.009 | 0.006 | 0.013 | no |
| extended | LR | 0.049 | 12 | 0.008 | 0.009 | 0.013 | 0.029 | no |
| extended | RF | 0.032 | 19 | 0.015 | 0.003 | 0.009 | 0.018 | no |
| extended | XGB | 0.071 | 15 | -0.026 | -0.003 | 0.028 | 0.065 | yes |

## Hungary

| Tier | Model | Mean \|Δp\| | Changed calls | ΔBA | ΔAUC | ΔBrier | Δlog loss | Flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early | LR | 0.055 | 23 | 0.017 | 0.016 | -0.002 | -0.005 | yes |
| early | RF | 0.040 | 32 | 0.061 | -0.002 | 0.005 | 0.010 | yes |
| early | XGB | 0.046 | 20 | -0.014 | -0.008 | 0.006 | 0.015 | no |
| extended | LR | 0.068 | 10 | 0.017 | 0.012 | 0.009 | 0.020 | yes |
| extended | RF | 0.026 | 6 | -0.021 | -0.004 | 0.008 | 0.017 | no |
| extended | XGB | 0.060 | 14 | 0.015 | -0.004 | 0.017 | 0.039 | yes |

## Switzerland

| Tier | Model | Mean \|Δp\| | Changed calls | ΔBA | ΔAUC | ΔBrier | Δlog loss | Flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early | LR | 0.010 | 0 | 0.000 | 0.007 | -0.005 | -0.013 | no |
| early | RF | 0.022 | 12 | 0.082 | 0.011 | 0.007 | 0.013 | yes |
| early | XGB | 0.024 | 5 | 0.013 | 0.018 | 0.008 | 0.014 | no |
| extended | LR | 0.021 | 7 | -0.036 | 0.001 | -0.015 | -0.037 | no |
| extended | RF | 0.011 | 24 | -0.037 | -0.001 | 0.007 | 0.015 | no |
| extended | XGB | 0.011 | 7 | -0.030 | 0.003 | -0.007 | -0.016 | no |

## VA Long Beach

| Tier | Model | Mean \|Δp\| | Changed calls | ΔBA | ΔAUC | ΔBrier | Δlog loss | Flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early | LR | 0.054 | 25 | 0.002 | 0.013 | -0.001 | -0.002 | yes |
| early | RF | 0.052 | 24 | 0.038 | 0.022 | 0.001 | -0.004 | yes |
| early | XGB | 0.060 | 15 | -0.017 | 0.038 | -0.006 | -0.036 | yes |
| extended | LR | 0.069 | 53 | -0.020 | 0.017 | -0.004 | -0.010 | yes |
| extended | RF | 0.037 | 21 | -0.011 | 0.011 | 0.001 | 0.001 | no |
| extended | XGB | 0.100 | 35 | 0.054 | 0.047 | -0.018 | -0.057 | yes |

Prediction sensitivity quantifies dependence on an encoding assumption. The meaning of a zero in the collection process remains unresolved. Both policies and the corresponding calibrated comparisons are retained in the JSON.
