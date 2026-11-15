"""Print aggregate comparisons only. Supply the unmodified Kaggle CSV and UCI ZIP."""

import argparse
import csv
from collections import Counter
from decimal import Decimal
from io import StringIO
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

from cardioscope.data.loading import load_dataset
from cardioscope.data.schema import PREDICTORS
from cardioscope.preprocessing import prepare_data
from cardioscope.splits import conventional_split

FILES = {
    "Cleveland": "processed.cleveland.data",
    "Hungary": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA Long Beach": "processed.va.data",
}
CODES = {
    "sex": {"Male": 1, "Female": 0},
    "cp": {"typical angina": 1, "atypical angina": 2, "non-anginal": 3, "asymptomatic": 4},
    "fbs": {"FALSE": 0, "TRUE": 1},
    "restecg": {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2},
    "exang": {"FALSE": 0, "TRUE": 1},
    "slope": {"upsloping": 1, "flat": 2, "downsloping": 3},
    "thal": {"normal": 3, "fixed defect": 6, "reversable defect": 7},
}


def code(name: str, token: str) -> Decimal | None:
    if token in {"", "?"}:
        return None
    return Decimal(CODES[name][token] if name in CODES else token)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--uci-zip", required=True, type=Path)
    args = parser.parse_args()
    frame = load_dataset(args.input).frame
    keys = (*PREDICTORS, "num")
    combined = frame.loc[:, list(keys)].to_dict("records")
    rows = [tuple(code(name, row[name]) for name in keys) for row in combined]
    sites = frame["dataset"].tolist()
    gains = {}
    losses = {}
    with ZipFile(args.uci_zip) as archive:
        for site, member in FILES.items():
            local = Counter(row for row, label in zip(rows, sites, strict=True) if label == site)
            records = list(csv.reader(StringIO(archive.read(member).decode("ascii"))))
            source = Counter(tuple(None if t == "?" else Decimal(t) for t in r) for r in records)
            gains[site], losses[site] = local - source, source - local
            print(
                site,
                "UCI/Kaggle/matched/UCI-only/Kaggle-only",
                source.total(),
                local.total(),
                (source & local).total(),
                (source - local).total(),
                (local - source).total(),
            )
            print(
                " UCI chol=0, BP=0, oldpeak<0, missing ca/thal/slope",
                sum(Decimal(r[4]) == 0 for r in records if r[4] != "?"),
                sum(Decimal(r[3]) == 0 for r in records if r[3] != "?"),
                sum(Decimal(r[9]) < 0 for r in records if r[9] != "?"),
                *[sum(r[i] == "?" for r in records) for i in (11, 12, 10)],
            )
            print(
                " excess exact repeats (UCI/Kaggle)",
                source.total() - len(source),
                local.total() - len(local),
            )
        for origin in FILES:
            for destination in FILES:
                matches = (losses[origin] & gains[destination]).total()
                if matches:
                    print("Exact multiset transfer", origin, "to", destination, matches)
        for site, member in (("Switzerland", "switzerland.data"), ("VA", "long-beach-va.data")):
            tokens = archive.read(member).decode("ascii").split()
            if len(tokens) % 76 or any(t != "name" for t in tokens[75::76]):
                raise ValueError("Unexpected UCI long-form record boundaries")
            print(
                site,
                "raw cholesterol zero / -9",
                tokens[11::76].count("0"),
                tokens[11::76].count("-9"),
            )

    pairs: Counter[tuple[str, str, str]] = Counter()
    held = set(conventional_split(prepare_data(frame)).test)
    for i, j in combinations(range(len(rows)), 2):
        a, b = rows[i][:-1], rows[j][:-1]
        difference = sum(x != y for x, y in zip(a, b, strict=True))
        observed = sum(x is not None and y is not None for x, y in zip(a, b, strict=True))
        if difference == 0 or (difference == 1 and observed >= 9):
            kind = "exact" if difference == 0 else "one field differs, >=9 jointly observed"
            location = "within site" if sites[i] == sites[j] else "cross site"
            target = "same num" if rows[i][-1] == rows[j][-1] else "different num"
            pairs[kind, location, target] += 1
            if difference == 1:
                fields = [f for f, x, y in zip(PREDICTORS, a, b, strict=True) if x != y]
                print(
                    "Near match:",
                    sites[i],
                    sites[j],
                    fields,
                    "crosses pooled holdout",
                    (i in held) != (j in held),
                )
    for labels, count in sorted(pairs.items()):
        print(*labels, count)


if __name__ == "__main__":
    main()
