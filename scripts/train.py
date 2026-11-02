#!/usr/bin/env python3
"""Train YOLO26 and evaluate its best checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blood_cells.training import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "TXL-PBC")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    device = int(args.device) if args.device.isdigit() else args.device
    report = run_training(
        args.dataset,
        args.output,
        model_name=args.model,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        patience=args.patience,
        device=device,
    )
    print(f"Test performance written to {args.output / 'reports/model_performance.json'}")
    print(report["metrics"])


if __name__ == "__main__":
    main()
