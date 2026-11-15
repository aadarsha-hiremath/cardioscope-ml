"""Render the same aggregate audit to Markdown and standards-compliant JSON."""

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from .data.audit import Audit
from .data.schema import COLUMNS, FIELDS


def table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    def line(cells: Iterable[object]) -> str:
        values = [
            str(cell if cell is not None else "—").replace("|", r"\|").replace("\n", " ")
            for cell in cells
        ]
        return "| " + " | ".join(values) + " |"

    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def markdown(audit: Audit) -> str:
    release = (
        "Matches the recorded Kaggle version 6 checksum."
        if audit.matches_kaggle_v6
        else "Input differs from the recorded Kaggle version 6 checksum."
    )
    sections = [
        "# CardioScope ML dataset audit",
        f"{audit.rows} records × {audit.columns} columns. {release}",
        f"Input SHA-256 `{audit.sha256}`.",
        "## Validation",
        "\n".join(f"- {error}" for error in audit.errors)
        or "All structural and domain checks pass.",
        "## Recorded outcomes",
        table(["Original num", "Records"], audit.original_target.items()),
        *(
            [table(["Invalid num token", "Records"], audit.invalid_target_tokens.items())]
            if audit.invalid_target_tokens
            else []
        ),
        table(["Derived binary target", "Records"], audit.binary_target.items()),
        f"Missing or invalid targets: {audit.unknown_target}. "
        "Positive means num is 1–4. Prevalence uses valid targets within each hospital.",
        "## Hospital populations",
        table(
            ["Hospital", "Records", "Valid targets", "Positive", "Positive (%)", "Median age"],
            (
                (s.hospital, s.rows, s.labeled, s.positive, s.prevalence_percent, s.age_median)
                for s in audit.hospitals
            ),
        ),
        "## Field inventory and missingness",
        "The loader retains CSV text. Expected types describe the values checked during the audit. "
        "An empty cell or a literal question mark counts as missing.",
        table(
            ["Field", "Storage", "Expected", "Role", "Missing", "Missing (%)"],
            (
                (name, f.storage_dtype, f.expected_type, f.role, f.missing.count, f.missing.percent)
                for name, f in audit.fields.items()
            ),
        ),
        "## Missingness within each hospital",
        table(
            ["Field", *[s.hospital or "(missing site)" for s in audit.hospitals]],
            (
                [
                    name,
                    *[
                        f"{s.missing[name].count} ({s.missing[name].percent}%)"
                        for s in audit.hospitals
                    ],
                ]
                for name in COLUMNS
            ),
        ),
        "## Numeric summaries",
        "Counts cover finite, representable values of the expected type, including flagged values. "
        "The mean and median include recorded zeros.",
        table(
            ["Field", "Count", "Min", "Median", "Mean", "Max"],
            (
                (name, n.count, n.minimum, n.median, n.mean, n.maximum)
                for name, f in audit.fields.items()
                if (n := f.numeric) is not None
            ),
        ),
        "## Categorical values",
        table(
            ["Field", "Observed label", "Records", "Expected label"],
            (
                (name, label, count, label in FIELDS[name].levels)
                for name, f in audit.fields.items()
                for label, count in f.categories.items()
            ),
        ),
        "## Duplicates and identifiers",
        "Counts measure extra occurrences after the first. Field comparisons use exact cell text. "
        "The identifier check compares numeric values and excludes invalid or missing IDs.",
        table(["Comparison", "Excess records"], audit.duplicate_excess.items()),
        "## Values for source review",
        table(
            [
                "Field",
                "Reason",
                "Records",
                *[s.hospital or "(missing site)" for s in audit.hospitals],
            ],
            (
                (
                    name,
                    reason,
                    count,
                    *[s.review.get(name, {}).get(reason, 0) for s in audit.hospitals],
                )
                for name, f in audit.fields.items()
                for reason, count in f.review.items()
            ),
        ),
        "Review nonpositive age, pressure, cholesterol, heart rate or ID and negative oldpeak "
        "against collection documentation. A flag alone does not establish a missing-value code.",
        "## Implications for evaluation",
        "Differences in site prevalence can change precision and the interpretation of a pooled "
        "calibration curve. Age mix and missing tests can also identify a collection site "
        "indirectly. Compare within-site results and report denominators for each held-out site. "
        "Matching non-ID fields flag records for linkage review before splitting.",
    ]
    return "\n\n".join(sections) + "\n"


def write_reports(audit: Audit, directory: Path, *, input_path: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        directory / "dataset-audit.json": json.dumps(asdict(audit), indent=2, allow_nan=False)
        + "\n",
        directory / "dataset-audit.md": markdown(audit),
    }
    for path in outputs:
        if path.exists() and path.samefile(input_path):
            raise ValueError("Report destination would overwrite the input.")
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8", newline="\n")
