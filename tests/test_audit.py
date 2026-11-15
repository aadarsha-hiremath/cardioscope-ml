"""All rows in these tests are invented; tests do not read the research CSV."""

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from cardioscope.cli import main
from cardioscope.data.audit import audit_dataset, binary_target
from cardioscope.data.loading import DatasetError, load_dataset
from cardioscope.data.schema import COLUMNS, PREDICTORS


def synthetic_row(**overrides: str) -> dict[str, str]:
    row = dict(
        zip(
            COLUMNS,
            [
                "9001",
                "49",
                "Female",
                "Hungary",
                "atypical angina",
                "119",
                "217",
                "FALSE",
                "normal",
                "153",
                "FALSE",
                "0.7",
                "flat",
                "1",
                "normal",
                "0",
            ],
            strict=True,
        )
    )
    return row | overrides


def write_csv(path: Path, *rows: dict[str, str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_load_and_preserve_roles(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "synthetic data.csv", synthetic_row(num="4"))
    dataset = load_dataset(path)
    before = dataset.frame.copy()
    assert dataset.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert tuple(dataset.frame.columns) == COLUMNS
    audit = audit_dataset(dataset)
    pd.testing.assert_frame_equal(before, dataset.frame)
    assert audit.fields["dataset"].role == "hospital metadata"
    assert dataset.frame.loc[0, "dataset"] == "Hungary"
    assert set(PREDICTORS) == set(COLUMNS) - {"id", "dataset", "num"}
    assert audit.original_target["4"] == 1
    assert audit.binary_target["1"] == 1
    assert audit.matches_kaggle_v6 is False
    assert any("Expected 920" in error for error in audit.errors)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "absent.csv")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("id,age\n1,50\n", "Missing columns"),
        (",".join(COLUMNS) + ",private\n", "Extra columns"),
        ("id,id\n1,1\n", "Duplicate column"),
        (",".join(COLUMNS) + "\n1,2\n", "has 2 fields"),
        (",".join(COLUMNS) + '\n"unfinished', "Invalid UTF-8 CSV"),
        (",".join(COLUMNS), "no data records"),
    ],
)
def test_bad_csv(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_dataset(path)


def test_binary_target_keeps_unknowns_and_index() -> None:
    original = pd.Series(
        ["0", "1", "2", "3", "4", "-1", "5", "", "x", "1.5", "NaN"],
        index=range(20, 31),
        dtype="string",
    )
    expected = pd.Series(
        [0, 1, 1, 1, 1, None, None, None, None, None, None],
        index=original.index,
        dtype="Int64",
        name="heart_disease_recorded",
    )
    pd.testing.assert_series_equal(binary_target(original), expected)


def test_missingness_and_prevalence_use_explicit_denominators(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "fixture.csv",
        synthetic_row(num="1", chol="0"),
        synthetic_row(id="9002", num="?", chol=""),
        synthetic_row(id="9003", num="7", chol="?", dataset=""),
    )
    audit = audit_dataset(load_dataset(path))
    assert audit.unknown_target == 2
    assert audit.invalid_target_tokens == {"7": 1}
    assert audit.fields["num"].invalid == {"target outside 0–4": 1}
    assert audit.fields["num"].missing.count == 1
    assert audit.fields["chol"].missing.percent == 66.667
    assert audit.fields["chol"].review == {"zero": 1}
    site, unknown_site = audit.hospitals
    assert (site.hospital, site.rows, site.labeled, site.positive) == ("Hungary", 2, 1, 1)
    assert site.prevalence_percent == 100
    assert site.missing["chol"].percent == 50
    assert site.review["chol"] == {"zero": 1}
    assert unknown_site.hospital is None
    assert unknown_site.prevalence_percent is None


def test_exact_duplicates_and_identifiers(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "fixture.csv",
        synthetic_row(),
        synthetic_row(),
        synthetic_row(id="9002"),
        synthetic_row(id="09002", age="55"),
    )
    audit = audit_dataset(load_dataset(path))
    assert audit.duplicate_excess == {
        "all_fields": 1,
        "excluding_id": 2,
        "repeated_valid_identifier": 2,
    }


def test_invalid_types_and_categories(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "fixture.csv",
        synthetic_row(
            age="24.5",
            cp="unknown category",
            dataset="unknown site",
            trestbps="inf",
            chol="NaN",
            thalch="not measured",
            ca="4",
            oldpeak="-1",
            fbs="unknown",
        ),
    )
    audit = audit_dataset(load_dataset(path))
    for name in ("age", "ca", "cp", "dataset", "trestbps", "chol", "thalch", "fbs"):
        assert audit.fields[name].invalid
    assert audit.fields["trestbps"].numeric is None
    assert audit.fields["chol"].missing.count == 0
    assert audit.fields["oldpeak"].review == {"negative ST depression": 1}


def test_cli_deterministic_reports(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "synthetic input.csv", synthetic_row(chol="NaN"))
    args = ["--input", str(path), "--output-dir", str(tmp_path)]
    raw = path.read_bytes()
    assert main(args) == 1  # Tiny fixture is flagged as a nonconforming combined dataset.
    output = tmp_path / "dataset-audit.json"
    report = tmp_path / "dataset-audit.md"
    summary = json.loads(output.read_bytes())
    assert summary["rows"] == 1 and summary["columns"] == 16
    assert summary["fields"]["chol"]["invalid"] == {"non-finite number": 1}
    before = (output.read_bytes(), report.read_bytes())
    assert main(args) == 1
    assert before == (output.read_bytes(), report.read_bytes())
    assert path.read_bytes() == raw


def test_cli_protects_input_on_output_collision(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "dataset-audit.json", synthetic_row())
    before = path.read_bytes()
    assert main(["--input", str(path), "--output-dir", str(tmp_path)]) == 2
    assert path.read_bytes() == before
