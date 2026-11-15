# Dataset provenance

| Item | Recorded value |
| --- | --- |
| Distribution | Redwan Sony, *UCI Heart Disease Data*, Kaggle version 6 |
| Acquisition | 2026-11-15, public HTTPS download from Kaggle’s dataset API, followed by byte-preserving extraction of the named ZIP member |
| CSV | `heart_disease_uci.csv`, 79,346 bytes, 920 data records, 16 columns |
| CSV SHA-256 | `574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae` |

The Kaggle metadata identified version 6 at acquisition. The download endpoint serves the current release by default, so compare the extracted file’s digest before reproducing results. The audit computes the checksum from the same byte snapshot it parses. [Kaggle dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) · [Download endpoint](https://www.kaggle.com/api/v1/datasets/download/redwankarimsony/heart-disease-data) · [Kaggle metadata](https://www.kaggle.com/api/v1/datasets/view/redwankarimsony/heart-disease-data)

## Attribution and redistribution

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. The investigators credited by the distributor are Andras Janosi (Hungarian Institute of Cardiology, Budapest), William Steinbrunn (University Hospital, Zurich), Matthias Pfisterer (University Hospital, Basel) and Robert Detrano (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation). Redwan Sony distributed the combined CSV. [Original dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) · [Dataset DOI](https://doi.org/10.24432/C52P4X)

UCI lists CC BY 4.0 for its dataset. Reuse under that license requires appropriate credit, a link to the license and an indication of changes. Kaggle labels the combined data files “Data files © Original Authors.” Redistribution terms specific to the combined CSV remain unconfirmed. The download and extracted CSV stay local under the Git-ignored raw directory. The reports contain aggregate statistics. [UCI license statement](https://archive.ics.uci.edu/dataset/45/heart+disease) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Kaggle metadata](https://www.kaggle.com/api/v1/datasets/view/redwankarimsony/heart-disease-data)

## Source conventions and open questions

- UCI describes four source databases and lists 303 instances for the Cleveland subset. Kaggle v6 assigns 304 of its 920 records to Cleveland. The reason for this difference has not been established. All site analyses use the labels in the checksummed CSV.
- The CSV retains release spellings such as `thalch` and `reversable defect`. Blank fields and literal question marks count as missing. Other tokens undergo type and category checks. A malformed numeric token, including a literal NaN or infinity, is counted separately from an absent value.
- UCI defines `chol` as serum cholesterol and `trestbps` as resting blood pressure. The audit found 172 zero cholesterol values, including all 123 Switzerland records and 49 at VA Long Beach. One VA Long Beach record has zero blood pressure. The cited descriptions do not establish a meaning for these zeros. Switzerland has zero formally missing cholesterol values, so a blank-cell count alone overstates measurement availability.
- Twelve records contain negative `oldpeak` values, 11 in Switzerland and one at VA Long Beach. The source describes ST depression relative to rest. Interpretation of the sign needs collection documentation. Zero is a defined outcome class for `num` and a permissible vessel count for `ca`. No raw value has been recoded.
- Every ID is populated and unique. Excluding ID yields two excess records with identical remaining fields. Matching fields alone cannot establish shared patient identity.
