"""Console entry point for the CPU-only audit."""

import argparse
import sys
from pathlib import Path

from .data.audit import audit_dataset
from .data.loading import load_dataset
from .report import write_reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the combined UCI heart disease CSV.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the UTF-8 source CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    try:
        audit = audit_dataset(load_dataset(args.input))
        write_reports(audit, args.output_dir, input_path=args.input)
    except (OSError, ValueError) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 2
    print(f"Audited {audit.rows} rows. Reports written to {args.output_dir}.")
    for error in audit.errors:
        print(error, file=sys.stderr)
    return 1 if audit.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
