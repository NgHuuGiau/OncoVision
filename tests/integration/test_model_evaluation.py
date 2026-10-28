from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from medical.model_evaluation import evaluate_classifier


class _StubWrapper:
    class_labels = ("a", "b")
    image_size = 96

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def predict(self, source, *, top_k=1):
        label = self._mapping[Path(source).name]
        return [{"label": label, "confidence": 0.9}]


class EvaluateClassifierTests(unittest.TestCase):
    def _make_dataset(self, root: Path) -> None:
        (root / "a").mkdir(parents=True)
        (root / "b").mkdir(parents=True)
        (root / "a" / "x1.jpg").write_bytes(b"fake")
        (root / "a" / "x2.jpg").write_bytes(b"fake")
        (root / "b" / "y1.jpg").write_bytes(b"fake")
        (root / "b" / "y2.jpg").write_bytes(b"fake")

    def test_reports_confusion_matrix_and_per_class_metrics(self) -> None:
        wrapper = _StubWrapper({"x1.jpg": "a", "x2.jpg": "b", "y1.jpg": "b", "y2.jpg": "b"})
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_dataset(root)
            report = evaluate_classifier(wrapper, root)
        self.assertEqual(report["total"], 4)
        self.assertEqual(report["correct"], 3)
        self.assertAlmostEqual(report["accuracy"], 0.75)
        self.assertEqual(report["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(report["image_size"], 96)
        per_class = {item["label"]: item for item in report["per_class"]}
        self.assertAlmostEqual(per_class["a"]["sensitivity"], 0.5)
        self.assertAlmostEqual(per_class["b"]["sensitivity"], 1.0)
        self.assertAlmostEqual(per_class["b"]["specificity"], 0.5)

    def test_unknown_class_directory_raises(self) -> None:
        wrapper = _StubWrapper({})
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "c").mkdir()
            with self.assertRaises(ValueError):
                evaluate_classifier(wrapper, root)

    def test_per_class_limit_slices_files(self) -> None:
        wrapper = _StubWrapper({"x1.jpg": "a", "x2.jpg": "a", "y1.jpg": "b", "y2.jpg": "b"})
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_dataset(root)
            report = evaluate_classifier(wrapper, root, per_class_limit=1)
        self.assertEqual(report["total"], 2)

    def test_missing_image_files_raises(self) -> None:
        wrapper = _StubWrapper({})
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a").mkdir()
            (root / "b").mkdir()
            with self.assertRaises(ValueError):
                evaluate_classifier(wrapper, root)


if __name__ == "__main__":
    unittest.main()
