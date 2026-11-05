from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from medical.detector_evaluation import evaluate_detector


class DetectorEvaluationTests(unittest.TestCase):
    def _make_dataset(self, root: Path) -> None:
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir(parents=True)
        (root / "images" / "a.jpg").write_bytes(b"x")
        (root / "images" / "b.jpg").write_bytes(b"x")
        (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "labels" / "b.txt").write_text("1 0.4 0.4 0.3 0.3\n", encoding="utf-8")

    def test_missing_layout_raises(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                evaluate_detector("model.pt", temp_dir)

    def test_missing_label_raises(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_dataset(root)
            (root / "labels" / "b.txt").unlink()
            with self.assertRaises(ValueError):
                evaluate_detector("model.pt", root)

    def test_uses_model_names_and_returns_map(self) -> None:
        box = SimpleNamespace(
            p=[0.9, 0.8], r=[0.7, 0.6], ap50=[0.7, 0.6], ap=[0.5, 0.4], map50=0.65, map=0.45, mp=0.85, mr=0.65
        )
        fake_model = SimpleNamespace(names={0: "tumor", 1: "cyst"}, val=lambda **kwargs: SimpleNamespace(box=box))
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_dataset(root)
            with patch("ultralytics.YOLO", return_value=fake_model), patch(
                "medical.detector_evaluation._run_yolo_val", return_value=SimpleNamespace(box=box)
            ) as val_mock:
                report = evaluate_detector("model.pt", root, conf=0.3)
        self.assertAlmostEqual(report["map50"], 0.65)
        self.assertAlmostEqual(report["map"], 0.45)
        self.assertEqual(report["image_count"], 2)
        self.assertEqual([item["label"] for item in report["per_class"]], ["tumor", "cyst"])
        self.assertEqual(val_mock.call_args.kwargs["conf"], 0.3)

    def test_non_contiguous_label_ids_get_contiguous_names(self) -> None:
        zeros = lambda n: [0.5] * n
        box = SimpleNamespace(p=zeros(6), r=zeros(6), ap50=zeros(6), ap=zeros(6), map50=0.5, map=0.5, mp=0.5, mr=0.5)
        fake_model = SimpleNamespace(names={}, val=lambda **kwargs: SimpleNamespace(box=box))
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_dataset(root)
            # id 5 xuất hiện nhưng thiếu 1-4 — ultralytics cần 0..5 liên tục
            (root / "labels" / "b.txt").write_text("5 0.4 0.4 0.3 0.3\n", encoding="utf-8")
            with patch("ultralytics.YOLO", return_value=fake_model), patch(
                "medical.detector_evaluation._run_yolo_val", return_value=SimpleNamespace(box=box)
            ):
                report = evaluate_detector("model.pt", root)
        self.assertEqual(sorted(report["class_names"]), list(range(6)))
        self.assertIn("5", report["class_names"].values())


    def test_per_class_maps_via_ap_class_index_when_gt_missing_class(self) -> None:
        box = SimpleNamespace(
            p=[0.9, 0.1], r=[0.8, 0.2], ap50=[0.7, 0.1], ap=[0.6, 0.1],
            map50=0.4, map=0.35, mp=0.5, mr=0.5,
            ap_class_index=[0, 2],  # id 1 không có GT trong bộ ảnh
        )
        fake_model = SimpleNamespace(
            names={0: "tumor", 1: "cyst", 2: "stone"}, val=lambda **kwargs: SimpleNamespace(box=box)
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_dataset(root)
            (root / "labels" / "b.txt").write_text("2 0.4 0.4 0.3 0.3\n", encoding="utf-8")
            with patch("ultralytics.YOLO", return_value=fake_model), patch(
                "medical.detector_evaluation._run_yolo_val", return_value=SimpleNamespace(box=box)
            ):
                report = evaluate_detector("model.pt", root)
        self.assertEqual([item["label"] for item in report["per_class"]], ["tumor", "stone"])
        self.assertAlmostEqual(report["per_class"][1]["ap50"], 0.1)


if __name__ == "__main__":
    unittest.main()
