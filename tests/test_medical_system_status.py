from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from medical.system_status import get_medical_system_status, recommended_medical_commands


class MedicalSystemStatusTests(unittest.TestCase):
    def test_get_medical_system_status_returns_zero_cases_when_db_is_unavailable(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset" / "medical"
            reports_dir = root / "output" / "medical" / "reports"
            normalized_dir = root / "output" / "medical" / "normalized_images"
            overlay_dir = root / "output" / "medical" / "processed_images"
            exports_dir = root / "output" / "medical" / "exports"
            for directory in (
                dataset_root / "raw" / "images",
                dataset_root / "raw" / "labels",
                dataset_root / "processed" / "images" / "train",
                dataset_root / "processed" / "images" / "val",
                dataset_root / "processed" / "images" / "test",
                reports_dir,
                normalized_dir,
                overlay_dir,
                exports_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (dataset_root / "data.yaml").write_text("path: .", encoding="utf-8")
            model_path = root / "models" / "trained" / "skin.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text("weights", encoding="utf-8")

            with patch(
                "medical.system_status.build_default_medical_analyzer_config",
                return_value=type(
                    "MedicalConfig",
                    (),
                    {
                        "model_path": model_path,
                        "working_dir": root / "output" / "medical",
                        "reports_dir": reports_dir,
                        "processed_dir": normalized_dir,
                        "overlay_dir": overlay_dir,
                        "fallback_model_path": None,
                        "allow_fallback_model": False,
                    },
                )(),
            ), patch(
                "medical.system_status.medical_training_paths",
                return_value=type(
                    "MedicalTrainingPaths",
                    (),
                    {
                        "dataset_root": dataset_root,
                        "data_yaml_path": dataset_root / "data.yaml",
                        "raw_images_dir": dataset_root / "raw" / "images",
                        "raw_labels_dir": dataset_root / "raw" / "labels",
                        "processed_images_dir": dataset_root / "processed" / "images",
                    },
                )(),
            ), patch(
                "medical.system_status.medical_output_directories",
                return_value=[reports_dir, normalized_dir, overlay_dir, exports_dir],
            ), patch(
                "medical.system_status.resolve_medical_runtime_model_path",
                return_value=model_path,
            ), patch(
                "medical.system_status.sqlite3.connect",
                side_effect=sqlite3.OperationalError("db unavailable"),
            ):
                status = get_medical_system_status()

        self.assertEqual(status.case_count, 0)

    def test_get_medical_system_status_counts_outputs_and_cases(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset" / "medical"
            reports_dir = root / "output" / "medical" / "reports"
            normalized_dir = root / "output" / "medical" / "normalized_images"
            overlay_dir = root / "output" / "medical" / "processed_images"
            exports_dir = root / "output" / "medical" / "exports"
            for directory in (dataset_root / "raw" / "images", dataset_root / "raw" / "labels", dataset_root / "processed" / "images" / "train", dataset_root / "processed" / "images" / "val", dataset_root / "processed" / "images" / "test", reports_dir, normalized_dir, overlay_dir, exports_dir):
                directory.mkdir(parents=True, exist_ok=True)
            (dataset_root / "data.yaml").write_text("path: .", encoding="utf-8")
            (dataset_root / "raw" / "images" / "img1.jpg").write_text("img", encoding="utf-8")
            (dataset_root / "raw" / "labels" / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (dataset_root / "processed" / "images" / "train" / "img1.jpg").write_text("train", encoding="utf-8")
            (dataset_root / "processed" / "images" / "val" / "img2.jpg").write_text("val", encoding="utf-8")
            (reports_dir / "case.json").write_text("{}", encoding="utf-8")
            (normalized_dir / "normalized.jpg").write_text("x", encoding="utf-8")
            (overlay_dir / "overlay.jpg").write_text("x", encoding="utf-8")
            (exports_dir / "case.zip").write_text("x", encoding="utf-8")
            model_path = root / "models" / "trained" / "skin.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text("weights", encoding="utf-8")
            db_path = root / "output" / "medical" / "medical_cases.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("CREATE TABLE medical_cases (id INTEGER PRIMARY KEY AUTOINCREMENT)")
                conn.execute("INSERT INTO medical_cases DEFAULT VALUES")
                conn.commit()
            finally:
                conn.close()

            with patch(
                "medical.system_status.build_default_medical_analyzer_config",
                return_value=type(
                    "MedicalConfig",
                    (),
                    {
                        "model_path": model_path,
                        "working_dir": root / "output" / "medical",
                        "reports_dir": reports_dir,
                        "processed_dir": normalized_dir,
                        "overlay_dir": overlay_dir,
                        "fallback_model_path": None,
                        "allow_fallback_model": False,
                    },
                )(),
            ), patch(
                "medical.system_status.medical_training_paths",
                return_value=type(
                    "MedicalTrainingPaths",
                    (),
                    {
                        "dataset_root": dataset_root,
                        "data_yaml_path": dataset_root / "data.yaml",
                        "raw_images_dir": dataset_root / "raw" / "images",
                        "raw_labels_dir": dataset_root / "raw" / "labels",
                        "processed_images_dir": dataset_root / "processed" / "images",
                    },
                )(),
            ), patch(
                "medical.system_status.medical_output_directories",
                return_value=[reports_dir, normalized_dir, overlay_dir, exports_dir],
            ), patch(
                "medical.system_status.resolve_medical_runtime_model_path",
                return_value=model_path,
            ):
                status = get_medical_system_status()

        self.assertTrue(status.model_ready)
        self.assertEqual(status.case_count, 1)
        self.assertEqual(status.report_files, 1)
        self.assertEqual(status.normalized_files, 1)
        self.assertEqual(status.overlay_files, 1)
        self.assertEqual(status.export_files, 1)
        self.assertEqual(status.raw_images, 1)
        self.assertEqual(status.train_images, 1)
        self.assertEqual(status.val_images, 1)

    def test_recommended_medical_commands_prioritize_bootstrap_and_training(self) -> None:
        status = type(
            "MedicalStatus",
            (),
            {
                "dataset_initialized": False,
                "raw_dataset_ready": False,
                "processed_dataset_ready": False,
                "model_ready": False,
                "report_files": 0,
                "normalized_files": 0,
                "overlay_files": 0,
                "export_files": 0,
            },
        )()

        commands = recommended_medical_commands(status)

        self.assertEqual(commands[0], "python run_medical.py init-dataset")
        self.assertIn("python run_medical.py audit-dataset", commands)
        self.assertIn("python run_medical.py train-all", commands)

    def test_get_medical_system_status_includes_screening_targets(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset" / "medical"
            reports_dir = root / "output" / "medical" / "reports"
            normalized_dir = root / "output" / "medical" / "normalized_images"
            overlay_dir = root / "output" / "medical" / "processed_images"
            exports_dir = root / "output" / "medical" / "exports"
            for directory in (dataset_root / "raw" / "images", dataset_root / "raw" / "labels", dataset_root / "processed" / "images" / "train", dataset_root / "processed" / "images" / "val", dataset_root / "processed" / "images" / "test", reports_dir, normalized_dir, overlay_dir, exports_dir):
                directory.mkdir(parents=True, exist_ok=True)
            (dataset_root / "data.yaml").write_text("path: .", encoding="utf-8")
            model_path = root / "models" / "trained" / "skin.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text("weights", encoding="utf-8")

            with patch(
                "medical.system_status.build_default_medical_analyzer_config",
                return_value=type(
                    "MedicalConfig",
                    (),
                    {
                        "model_path": model_path,
                        "working_dir": root / "output" / "medical",
                        "reports_dir": reports_dir,
                        "processed_dir": normalized_dir,
                        "overlay_dir": overlay_dir,
                        "fallback_model_path": None,
                        "allow_fallback_model": False,
                    },
                )(),
            ), patch(
                "medical.system_status.medical_training_paths",
                return_value=type(
                    "MedicalTrainingPaths",
                    (),
                    {
                        "dataset_root": dataset_root,
                        "data_yaml_path": dataset_root / "data.yaml",
                        "raw_images_dir": dataset_root / "raw" / "images",
                        "raw_labels_dir": dataset_root / "raw" / "labels",
                        "processed_images_dir": dataset_root / "processed" / "images",
                    },
                )(),
            ), patch(
                "medical.system_status.medical_output_directories",
                return_value=[reports_dir, normalized_dir, overlay_dir, exports_dir],
            ), patch(
                "medical.system_status.resolve_medical_runtime_model_path",
                return_value=model_path,
            ):
                status = get_medical_system_status()

        self.assertTrue(status.screening_targets)
        self.assertIn(("Ung thu da", True), status.screening_targets)

    def test_dataset_counts_are_separated_between_medical_and_object_detection(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            root = Path(temp_dir)
            medical_root = root / "dataset" / "medical" / "skin_lesion"
            object_root = root / "dataset" / "object_detection"
            for directory in (
                medical_root / "raw" / "images",
                medical_root / "raw" / "labels",
                medical_root / "processed" / "images" / "train",
                medical_root / "processed" / "images" / "val",
                medical_root / "processed" / "images" / "test",
                object_root / "raw" / "images",
                object_root / "raw" / "labels",
                object_root / "processed" / "images" / "train",
                object_root / "processed" / "images" / "val",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (medical_root / "data.yaml").write_text("path: .", encoding="utf-8")
            (medical_root / "raw" / "images" / "skin.jpg").write_text("skin", encoding="utf-8")
            (medical_root / "raw" / "labels" / "skin.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (object_root / "raw" / "images" / "box.jpg").write_text("box", encoding="utf-8")
            (object_root / "raw" / "labels" / "box.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            with patch("medical.system_status.medical_training_paths") as training_paths_mock, patch(
                "medical.system_status.build_default_medical_analyzer_config"
            ) as config_mock, patch(
                "medical.system_status.medical_output_directories", return_value=[root / "output" / "medical" / "reports", root / "output" / "medical" / "normalized_images", root / "output" / "medical" / "processed_images", root / "output" / "medical" / "exports"]
            ), patch("medical.system_status.resolve_medical_runtime_model_path", return_value=root / "models" / "trained" / "skin.pt"):
                training_paths_mock.return_value = type(
                    "TrainingPaths",
                    (),
                    {
                        "dataset_root": medical_root,
                        "data_yaml_path": medical_root / "data.yaml",
                        "raw_images_dir": medical_root / "raw" / "images",
                        "raw_labels_dir": medical_root / "raw" / "labels",
                        "processed_images_dir": medical_root / "processed" / "images",
                    },
                )()
                config_mock.return_value = type(
                    "MedicalConfig",
                    (),
                    {
                        "model_path": root / "models" / "trained" / "skin.pt",
                        "working_dir": root / "output" / "medical",
                        "reports_dir": root / "output" / "medical" / "reports",
                        "processed_dir": root / "output" / "medical" / "normalized_images",
                        "overlay_dir": root / "output" / "medical" / "processed_images",
                        "fallback_model_path": None,
                        "allow_fallback_model": False,
                    },
                )()
                status = get_medical_system_status()

            object_images = sum(1 for _ in (object_root / "raw" / "images").glob("*") if _.is_file())
            object_labels = sum(1 for _ in (object_root / "raw" / "labels").glob("*") if _.is_file())

        self.assertEqual(status.raw_images, 1)
        self.assertEqual(status.raw_labels, 1)
        self.assertEqual(object_images, 1)
        self.assertEqual(object_labels, 1)
