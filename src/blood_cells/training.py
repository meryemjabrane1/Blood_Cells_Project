"""YOLO26 training and held-out test evaluation."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from blood_cells.dataset import CLASS_NAMES, write_yolo_config


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _class_metrics(metrics: Any) -> list[dict[str, Any]]:
    box = metrics.box
    class_metrics = []
    for index, class_id in enumerate(box.ap_class_index):
        precision, recall, ap50, ap = box.class_result(index)
        class_metrics.append(
            {
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[int(class_id)],
                "precision": _as_float(precision),
                "recall": _as_float(recall),
                "map50": _as_float(ap50),
                "map50_95": _as_float(ap),
            }
        )
    return class_metrics


def run_training(
    dataset_root: Path,
    output_dir: Path,
    *,
    model_name: str = "yolo26n.pt",
    epochs: int = 100,
    image_size: int = 640,
    batch_size: int = 16,
    patience: int = 20,
    device: str | int = 0,
) -> dict[str, Any]:
    """Fine-tune a pretrained YOLO26 detector, then evaluate on the test split."""
    from ultralytics import YOLO

    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = write_yolo_config(dataset_root, output_dir / "data.yaml")
    progress_path = reports_dir / "training_progress.json"
    started_at = datetime.now(timezone.utc).isoformat()
    _atomic_json(
        progress_path,
        {
            "status": "starting",
            "epoch": 0,
            "total_epochs": epochs,
            "updated_at": started_at,
        },
    )

    def update_progress(trainer: Any) -> None:
        metrics = {
            str(key): _as_float(value)
            for key, value in (getattr(trainer, "metrics", {}) or {}).items()
        }
        _atomic_json(
            progress_path,
            {
                "status": "training",
                "epoch": int(trainer.epoch) + 1,
                "total_epochs": int(trainer.epochs),
                "metrics": metrics,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    try:
        model = YOLO(model_name)
        model.add_callback("on_fit_epoch_end", update_progress)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            patience=patience,
            device=device,
            workers=2,
            seed=42,
            deterministic=True,
            project=str(output_dir / "runs"),
            name="yolo26n",
            exist_ok=True,
            plots=True,
            save_period=10,
        )

        best_weights = output_dir / "runs" / "yolo26n" / "weights" / "best.pt"
        if not best_weights.is_file():
            raise FileNotFoundError(f"Training did not produce best weights: {best_weights}")

        history_path = output_dir / "runs" / "yolo26n" / "results.csv"
        if history_path.is_file():
            with history_path.open(newline="", encoding="utf-8") as source:
                history = list(csv.DictReader(source))
            _atomic_json(
                reports_dir / "training_history.json",
                {"epochs": history},
            )

        best_model = YOLO(str(best_weights))
        test_metrics = best_model.val(
            data=str(data_yaml),
            split="test",
            imgsz=image_size,
            batch=batch_size,
            device=device,
            plots=True,
            project=str(output_dir / "runs"),
            name="test",
            exist_ok=True,
        )
        performance = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "weights": best_weights.relative_to(output_dir).as_posix(),
            "test_split": "test",
            "metrics": {
                key: _as_float(value)
                for key, value in test_metrics.results_dict.items()
            },
            "per_class": _class_metrics(test_metrics),
            "test_images": len(
                [
                    path
                    for path in (dataset_root / "images" / "test").iterdir()
                    if path.is_file()
                    and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
            ),
            "class_names": CLASS_NAMES,
        }
        _atomic_json(reports_dir / "model_performance.json", performance)
        _atomic_json(
            progress_path,
            {
                "status": "completed",
                "epoch": int(getattr(model.trainer, "epoch", epochs - 1)) + 1,
                "total_epochs": epochs,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "performance": performance["metrics"],
            },
        )
        return performance
    except Exception as error:
        _atomic_json(
            progress_path,
            {
                "status": "failed",
                "epoch": int(getattr(getattr(model, "trainer", None), "epoch", -1))
                + 1,
                "total_epochs": epochs,
                "error": f"{type(error).__name__}: {error}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
