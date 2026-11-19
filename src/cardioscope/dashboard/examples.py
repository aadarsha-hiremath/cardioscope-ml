"""Synthetic regression fixtures and references from earlier saved profiles."""

from .evidence import Record
from .sandbox import PROFILE

EXAMPLES: dict[str, Record] = {
    "Fictional A": PROFILE,
    "Fictional B": {
        **PROFILE,
        "age": 64.0,
        "sex": "Male",
        "cp": "asymptomatic",
        "trestbps": 145.0,
        "chol": 280.0,
        "fbs": "TRUE",
    },
    "Fictional C · zero cholesterol": {
        **PROFILE,
        "age": 37.0,
        "chol": 0.0,
        "cp": "non-anginal",
    },
}

REPORT_DISCLAIMER = (
    "For educational research only. The experimental score is not a clinically "
    "validated probability of disease."
)
