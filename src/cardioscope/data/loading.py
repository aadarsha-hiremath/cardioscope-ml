"""Read one byte snapshot so shape checks, data and digest describe the same input."""

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schema import COLUMNS


class DatasetError(ValueError):
    """Input cannot be interpreted as a rectangular combined-dataset CSV."""


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    sha256: str


def load_dataset(path: Path) -> Dataset:
    raw = path.read_bytes()
    try:
        stream = io.StringIO(raw.decode("utf-8-sig"), newline="")
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if len(set(header)) != len(header):
            raise DatasetError("Duplicate column names in header.")
        missing = sorted(set(COLUMNS) - set(header))
        extra = sorted(set(header) - set(COLUMNS))
        if missing or extra:
            raise DatasetError(
                f"Header mismatch. Missing columns {missing}. Extra columns {extra}."
            )
        records = []
        for row in reader:
            if len(row) != len(header):
                raise DatasetError(
                    f"Line {reader.line_num} has {len(row)} fields; expected {len(header)}."
                )
            records.append(row)
        if not records:
            raise DatasetError("Header has no data records.")
    except (csv.Error, UnicodeError) as exc:
        raise DatasetError(f"Invalid UTF-8 CSV: {exc}") from exc
    # Keep exact cell text for duplicate checks. Interpret missing markers only in analysis.
    frame = pd.DataFrame(records, columns=pd.Index(header), dtype="string")
    return Dataset(frame, hashlib.sha256(raw).hexdigest())
