from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from medical.classifier import load_medical_classifier
from medical.training import (
    MedicalTrainingPaths,
    audit_medical_raw_dataset,
    medical_training_paths,
    prepare_medical_training_dataset,
    run_full_medical_training_pipeline,
    train_medical_model,
    validate_medical_model,
)


class MedicalTrainingTests(unittest.TestCase):
    def _paths(self, root: Path) -> MedicalTrainingPaths:
        dataset_root = root / "dataset"
        return MedicalTrainingPaths(
            dataset_root=dataset_root,
            data_yaml_path=dataset_root / "data.yaml",
            trained_model_path=root / "medical_7_cancers.pt",
            class_names=(
                "Ung thư gan",
                "Ung thư phổi",
                "Ung thư vú",
                "Ung thư dạ dày",
                "Ung thư đại trực tràng",
                "Ung thư tuyến tiền liệt",
                "Ung thư cổ tử cung",
            ),
            feature_size=16,
        )

    def _seed_dataset(self, paths: MedicalTrainingPaths) -> None:
        for index, class_name in enumerate(paths.class_names):
            color = (index * 30 % 255, index * 60 % 255, index * 90 % 255)
            for split in ("train", "val", "test"):
                split_dir = paths.dataset_root / class_name / "processed" / "images" / split
                split_dir.mkdir(parents=True, exist_ok=True)
                image_path = split_dir / f"{class_name}_{split}.jpg"
                Image.new("RGB", (24, 24), color).save(image_path)
        paths.data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        paths.data_yaml_path.write_text("path: dataset/medical\n", encoding="utf-8")

    def test_audit_medical_raw_dataset_reports_class_counts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            paths = self._paths(Path(temp_dir))
            self._seed_dataset(paths)

            audit = audit_medical_raw_dataset(paths)

            self.assertEqual(audit["missing_classes"], [])
            self.assertEqual(audit["class_counts"][paths.class_names[0]], 3)
            self.assertEqual(len(audit["train_images"][paths.class_names[0]]), 1)

    def test_prepare_medical_training_dataset_reports_split_counts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            paths = self._paths(Path(temp_dir))
            self._seed_dataset(paths)

            summary = prepare_medical_training_dataset(paths)

            self.assertEqual(summary.class_count, 7)
            self.assertEqual(summary.train_count, 7)
            self.assertEqual(summary.val_count, 7)
            self.assertEqual(summary.test_count, 7)
            self.assertTrue(paths.data_yaml_path.exists())

    def test_prepare_medical_training_dataset_populates_splits_from_raw_class_images(self) -> None:
        with TemporaryDirectory() as temp_dir:
            paths = self._paths(Path(temp_dir))
            class_dir = paths.dataset_root / paths.class_names[0] / "raw" / "images"
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(6):
                image_path = class_dir / f"sample_{index}.jpg"
                Image.new("RGB", (64, 64), (index * 20 % 256, 50, 100)).save(image_path)

            summary = prepare_medical_training_dataset(paths)

            self.assertGreater(summary.train_count, 0)
            self.assertGreater(summary.val_count, 0)
            self.assertGreater(summary.test_count, 0)

    def test_train_medical_model_saves_classifier(self) -> None:
        with TemporaryDirectory() as temp_dir:
            paths = self._paths(Path(temp_dir))
            self._seed_dataset(paths)

            model_path = train_medical_model(paths)
            classifier = load_medical_classifier(model_path)

            self.assertTrue(model_path.exists())
            self.assertEqual(len(classifier.class_labels), 7)

    def test_validate_medical_model_uses_current_model_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            paths = self._paths(Path(temp_dir))
            self._seed_dataset(paths)
            train_medical_model(paths)

            metrics = validate_medical_model(paths)

            self.assertEqual(metrics["class_count"], 7)
            self.assertGreaterEqual(metrics["accuracy"], 0.9)

    @patch("medical.training.validate_medical_model", return_value={"accuracy": 0.8})
    @patch("medical.training.train_medical_model")
    @patch("medical.training.prepare_medical_training_dataset")
    @patch("medical.training.medical_training_paths")
    def test_run_full_medical_training_pipeline_returns_combined_report(
        self,
        paths_mock,
        prepare_mock,
        train_mock,
        validate_mock,
    ) -> None:
        fake_paths = self._paths(Path("D:/YOLO/tmp-medical"))
        paths_mock.return_value = fake_paths
        prepare_mock.return_value = SimpleNamespace(train_count=7, val_count=7, test_count=7)
        train_mock.return_value = fake_paths.trained_model_path

        report = run_full_medical_training_pipeline()

        self.assertEqual(report["train_count"], 7)
        self.assertEqual(report["trained_model_path"], fake_paths.trained_model_path)
        self.assertEqual(report["validation_metrics"]["accuracy"], 0.8)

    @patch("medical.training._load_medical_settings")
    def test_medical_training_paths_uses_configured_dataset_and_model_names(self, load_settings_mock) -> None:
        load_settings_mock.return_value = {
            "dataset_root": "dataset/custom_medical",
            "disease_name": "custom_medical",
            "feature_size": 24,
        }

        paths = medical_training_paths()

        self.assertEqual(paths.dataset_root, Path("dataset/custom_medical"))
        self.assertEqual(paths.trained_model_path, Path("medical_7_cancers.pt"))
        self.assertEqual(paths.feature_size, 24)


if __name__ == "__main__":
    unittest.main()
