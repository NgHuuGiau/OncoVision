from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medical import evaluation
from medical.training import MedicalTrainingPaths


class _FakeModel:
    """Model gia lap tra ve du doan xac dinh theo ten file."""

    def __init__(self, class_labels: tuple[str, ...]) -> None:
        self.class_labels = class_labels

    def predict(self, image_path, top_k: int = 3):
        # Ten file dang '<label_index>_...': doan dung lop do voi conf cao.
        stem = Path(image_path).stem
        idx = int(stem.split("_")[0])
        results = []
        for i, label in enumerate(self.class_labels):
            conf = 0.9 if i == idx else 0.02
            results.append({"label": label, "confidence": conf})
        results.sort(key=lambda item: -item["confidence"])
        return results[:top_k]


class EvaluationTests(unittest.TestCase):
    def _paths(self, root: Path) -> MedicalTrainingPaths:
        return MedicalTrainingPaths(
            dataset_root=root,
            trained_model_path=root / "model.pt",
            class_names=("A", "B"),
        )

    def test_evaluate_on_test_set_perfect_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.pt"
            model_path.write_bytes(b"stub")
            paths = self._paths(root)
            samples = [
                (root / "0_a1.jpg", 0),
                (root / "0_a2.jpg", 0),
                (root / "1_b1.jpg", 1),
                (root / "1_b2.jpg", 1),
            ]
            fake_model = _FakeModel(paths.class_names)
            with patch.object(evaluation, "_samples_for_split", return_value=samples), \
                 patch.object(evaluation, "_resolve_model", return_value=(fake_model, True)):
                report = evaluation.evaluate_on_test_set(paths, model_path=model_path, split="test")

            self.assertEqual(report["num_samples"], 4)
            self.assertAlmostEqual(report["accuracy"], 1.0)
            self.assertEqual(len(report["per_class"]), 2)
            for entry in report["per_class"]:
                self.assertAlmostEqual(entry["sensitivity"], 1.0)
                self.assertAlmostEqual(entry["specificity"], 1.0)

    def test_evaluate_raises_when_no_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.pt"
            model_path.write_bytes(b"stub")
            paths = self._paths(root)
            with patch.object(evaluation, "_samples_for_split", return_value=[]):
                with self.assertRaises(FileNotFoundError):
                    evaluation.evaluate_on_test_set(paths, model_path=model_path)

    def test_evaluate_raises_when_model_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._paths(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                evaluation.evaluate_on_test_set(paths, model_path=Path(tmp) / "missing.pt")

    def test_write_evaluation_report_creates_json_and_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = {
                "model_path": "m.pt",
                "split": "test",
                "num_samples": 2,
                "class_labels": ["A", "B"],
                "accuracy": 1.0,
                "macro_precision": 1.0,
                "macro_recall": 1.0,
                "macro_f1": 1.0,
                "micro_f1": 1.0,
                "macro_roc_auc": 1.0,
                "macro_pr_auc": 1.0,
                "per_class": [
                    {"label": "A", "support": 1, "precision": 1.0, "recall": 1.0,
                     "sensitivity": 1.0, "specificity": 1.0, "f1_score": 1.0,
                     "roc_auc": 1.0, "pr_auc": 1.0},
                ],
                "confusion_matrix": [[1, 0], [0, 1]],
                "evaluated_at": "2026-01-01T00:00:00",
            }
            json_path, md_path = evaluation.write_evaluation_report(report, reports_dir=tmp)
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["accuracy"], 1.0)
            self.assertIn("Per-Class", md_path.read_text(encoding="utf-8"))


class PredictScoresTests(unittest.TestCase):
    def test_out_of_vocab_label_returns_negative_index(self) -> None:
        class _ForeignModel:
            class_labels = ("X", "Y")

            def predict(self, image_path, top_k: int = 3):
                return [{"label": "X", "confidence": 0.9}, {"label": "Y", "confidence": 0.1}]

        idx, scores = evaluation._predict_scores(_ForeignModel(), Path("0_a.jpg"), ("A", "B"))
        self.assertEqual(idx, -1)
        # Khong nhan nao khop nen scores van la vector 0.
        self.assertEqual(float(scores.sum()), 0.0)

    def test_in_vocab_label_maps_correctly(self) -> None:
        class _Model:
            def predict(self, image_path, top_k: int = 3):
                return [{"label": "B", "confidence": 0.8}, {"label": "A", "confidence": 0.2}]

        idx, scores = evaluation._predict_scores(_Model(), Path("1_b.jpg"), ("A", "B"))
        self.assertEqual(idx, 1)
        self.assertAlmostEqual(float(scores[1]), 0.8)


if __name__ == "__main__":
    unittest.main()
