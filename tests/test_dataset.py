from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blood_cells.dataset import summarize_dataset, write_yolo_config


class DatasetSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "dataset"
        for split in ("train", "val", "test"):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
            Image.new("RGB", (20, 10)).save(
                self.root / "images" / split / "sample.png"
            )
            (self.root / "labels" / split / "sample.txt").write_text(
                "0 0.5 0.5 0.25 0.4\n2 0.2 0.3 0.1 0.2\n",
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_summarizes_splits_classes_and_image_dimensions(self) -> None:
        report = summarize_dataset(self.root)
        self.assertEqual(report["totals"]["images"], 3)
        self.assertEqual(report["totals"]["annotated_cells"], 6)
        self.assertEqual(report["totals"]["class_counts"]["WBC"], 3)
        self.assertEqual(report["totals"]["class_counts"]["Platelets"], 3)
        self.assertEqual(report["image_dimensions"]["width_range"], [20, 20])
        self.assertEqual(report["image_dimensions"]["height_range"], [10, 10])
        self.assertFalse(any(report["validation"].values()))

    def test_reports_missing_labels_and_invalid_annotations(self) -> None:
        (self.root / "labels" / "train" / "sample.txt").write_text(
            "7 0.5 0.5 0.3 0.3\nnot-a-label\n", encoding="utf-8"
        )
        (self.root / "labels" / "val" / "extra.txt").write_text(
            "1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
        report = summarize_dataset(self.root)
        self.assertEqual(report["validation"]["invalid_annotations"], 1)
        self.assertEqual(report["validation"]["malformed_labels"], 1)
        self.assertEqual(report["validation"]["orphan_labels"], 1)

    def test_writes_absolute_yolo_paths(self) -> None:
        config = write_yolo_config(self.root, self.root / "generated.yaml")
        contents = config.read_text(encoding="utf-8")
        self.assertIn(f"path: {self.root.resolve().as_posix()}", contents)
        self.assertIn("names: [\"WBC\", \"RBC\", \"Platelets\"]", contents)


if __name__ == "__main__":
    unittest.main()
