"""The only input vectors permitted in saved fictional workspaces."""

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
    "Built-in fictional example. Experimental model score — not a clinically "
    "validated probability of disease."
)
