"""Dataset inspection and YOLO data-configuration helpers."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

CLASS_NAMES = ["WBC", "RBC", "Platelets"]
SPLITS = ("train", "val", "test")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def summarize_dataset(dataset_root: Path) -> dict[str, Any]:
    """Validate YOLO label files and summarize image and class counts."""
    root = dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    split_reports: dict[str, Any] = {}
    total_class_counts: Counter[str] = Counter()
    validation = Counter()
    image_widths: list[int] = []
    image_heights: list[int] = []

    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Expected images/{split} and labels/{split} under {root}"
            )

        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        image_stems = {image.stem for image in images}
        label_files = sorted(label_dir.glob("*.txt"))
        label_stems = {label.stem for label in label_files}
        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)
        validation["images_missing_labels"] += len(missing_labels)
        validation["orphan_labels"] += len(orphan_labels)
        class_counts: Counter[str] = Counter()
        annotated_cells = 0

        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
                    image_widths.append(width)
                    image_heights.append(height)
            except (OSError, UnidentifiedImageError):
                validation["unreadable_images"] += 1

        for label_path in label_files:
            try:
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 5:
                        validation["malformed_labels"] += 1
                        continue
                    try:
                        class_id = int(fields[0])
                        box = [float(value) for value in fields[1:]]
                    except ValueError:
                        validation["malformed_labels"] += 1
                        continue
                    if (
                        class_id < 0
                        or class_id >= len(CLASS_NAMES)
                        or not all(math.isfinite(value) for value in box)
                        or any(value < 0 or value > 1 for value in box)
                        or box[2] <= 0
                        or box[3] <= 0
                    ):
                        validation["invalid_annotations"] += 1
                        continue
                    class_counts[CLASS_NAMES[class_id]] += 1
                    annotated_cells += 1
            except OSError:
                validation["unreadable_labels"] += 1

        total_class_counts.update(class_counts)
        split_reports[split] = {
            "images": len(images),
            "annotated_cells": annotated_cells,
            "class_counts": {
                name: class_counts.get(name, 0) for name in CLASS_NAMES
            },
            "images_missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
        }

    total_images = sum(report["images"] for report in split_reports.values())
    total_cells = sum(report["annotated_cells"] for report in split_reports.values())
    width_range = (
        [min(image_widths), max(image_widths)] if image_widths else [None, None]
    )
    height_range = (
        [min(image_heights), max(image_heights)] if image_heights else [None, None]
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": root.name,
        "classes": [
            {"id": index, "name": name} for index, name in enumerate(CLASS_NAMES)
        ],
        "splits": split_reports,
        "totals": {
            "images": total_images,
            "annotated_cells": total_cells,
            "class_counts": {
                name: total_class_counts.get(name, 0) for name in CLASS_NAMES
            },
        },
        "image_dimensions": {
            "width_range": width_range,
            "height_range": height_range,
        },
        "validation": dict(validation),
    }


def write_yolo_config(dataset_root: Path, output_path: Path) -> Path:
    """Write an absolute-path YOLO config, suitable for Kaggle input mounts."""
    root = dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {root.as_posix()}",
        f"train: {(root / 'images' / 'train').as_posix()}",
        f"val: {(root / 'images' / 'val').as_posix()}",
        f"test: {(root / 'images' / 'test').as_posix()}",
        f"nc: {len(CLASS_NAMES)}",
        f"names: {json.dumps(CLASS_NAMES)}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_dataset_report(dataset_root: Path, output_path: Path) -> dict[str, Any]:
    report = summarize_dataset(dataset_root)
    _atomic_json(output_path, report)
    return report
