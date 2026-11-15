"""Field definitions for the combined Kaggle CSV (including its original spellings)."""

from dataclasses import dataclass

COLUMNS = (
    "id",
    "age",
    "sex",
    "dataset",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
)
HOSPITALS = ("Cleveland", "Hungary", "Switzerland", "VA Long Beach")
TARGETS = ("0", "1", "2", "3", "4")
EXPECTED_ROWS = 920
RELEASE_SHA256 = "574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae"
PREDICTORS = tuple(c for c in COLUMNS if c not in {"id", "dataset", "num"})


@dataclass(frozen=True)
class Field:
    kind: str
    role: str = "candidate predictor"
    levels: tuple[str, ...] = ()


FIELDS = {
    "id": Field("integer", "identifier"),
    "age": Field("integer"),
    "sex": Field("category", levels=("Female", "Male")),
    "dataset": Field("category", "hospital metadata", HOSPITALS),
    "cp": Field(
        "category", levels=("typical angina", "atypical angina", "non-anginal", "asymptomatic")
    ),
    "trestbps": Field("integer"),
    "chol": Field("integer"),
    "fbs": Field("boolean", levels=("TRUE", "FALSE")),
    "restecg": Field("category", levels=("normal", "st-t abnormality", "lv hypertrophy")),
    "thalch": Field("integer"),
    "exang": Field("boolean", levels=("TRUE", "FALSE")),
    "oldpeak": Field("real"),
    "slope": Field("category", levels=("upsloping", "flat", "downsloping")),
    "ca": Field("integer"),
    "thal": Field("category", levels=("normal", "fixed defect", "reversable defect")),
    "num": Field("integer", "original target"),
}
