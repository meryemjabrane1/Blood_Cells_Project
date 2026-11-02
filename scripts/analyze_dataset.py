#!/usr/bin/env python3
"""Validate the blood-cell dataset and write its dashboard report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blood_cells.dataset import write_dataset_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "TXL-PBC")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "reports/dataset_report.json"
    )
    args = parser.parse_args()
    report = write_dataset_report(args.dataset, args.output)
    print(
        f"Validated {report['totals']['images']} images and "
        f"{report['totals']['annotated_cells']} cell annotations; "
        f"report: {args.output}"
    )
    issues = {
        key: value
        for key, value in report["validation"].items()
        if value
    }
    if issues:
        raise SystemExit(f"Dataset validation found issues: {issues}")


if __name__ == "__main__":
    main()
