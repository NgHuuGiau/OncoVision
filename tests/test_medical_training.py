from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from medical.training import (
    MedicalTrainingPaths,
    audit_medical_raw_dataset,
    prepare_medical_training_dataset,
    run_full_medical_training_pipeline,
    medical_training_paths,
    sync_medical_model_config,
    train_medical_model,
    validate_medical_model,
)


class _FakeYOLO:
    def __init__(self, path: str) -> None:
        self.path = path

    def train(self, **kwargs):
        return SimpleNamespace(save_dir=kwargs["project"] + "/medical-run")

    def val(self, **kwargs):
        return {"map50": 0.77, "project": kwargs["project"]}


class MedicalTrainingTests(unittest.TestCase):
    def _paths(self, root: Path) -> MedicalTrainingPaths:
        dataset_root = root / "dataset"
        return MedicalTrainingPaths(
            dataset_root=dataset_root,
            raw_images_dir=dataset_root / "raw" / "images",
            raw_labels_dir=dataset_root / "raw" / "labels",
            processed_images_dir=dataset_root / "processed" / "images",
            processed_labels_dir=dataset_root / "processed" / "labels",
            data_yaml_path=dataset_root / "data.yaml",
            train_runs_dir=root / "runs" / "train",
            val_runs_dir=root / "runs" / "val",
            trained_model_path=root / "models" / "trained" / "skin_cancer_screening_best.pt",
        )

    def _seed_raw_dataset(self, paths: MedicalTrainingPaths, count: int = 4) -> None:
        paths.raw_images_dir.mkdir(parents=True, exist_ok=True)
        paths.raw_labels_dir.mkdir(parents=True, exist_ok=True)
        paths.data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        paths.data_yaml_path.write_text(
            "path: .\ntrain: processed/images/train\nval: processed/images/val\ntest: processed/images/test\nnames:\n  0: lesion\n",
            encoding="utf-8",
        )
        for index in range(count):
            (paths.raw_images_dir / f"img{index}.jpg").write_text("img", encoding="utf-8")
            (paths.raw_labels_dir / f"img{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def test_audit_medical_raw_dataset_reports_eligible_pairs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            paths = self._paths(Path(temp_dir))
            self._seed_raw_dataset(paths, count=2)

            audit = audit_medical_raw_dataset(paths)

            self.assertEqual(len(audit["eligible"]), 2)
            self.assertEqual(audit["missing_labels"], [])
            self.assertEqual(audit["invalid_labels"], [])

    def test_prepare_medical_training_dataset_creates_split_dirs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            paths = self._paths(Path(temp_dir))
            self._seed_raw_dataset(paths, count=5)

            summary = prepare_medical_training_dataset(paths)

            self.assertEqual(summary.train_count + summary.val_count + summary.test_count, 5)
            self.assertTrue((paths.processed_images_dir / "train").exists())
            self.assertTrue((paths.processed_labels_dir / "test").exists())

    @patch("medical.training.sync_medical_model_config")
    @patch("medical.training.resolve_medical_base_model", return_value=Path("yolo11n.pt"))
    def test_train_medical_model_copies_best_weight(self, _resolve_base_model_mock, sync_mock) -> None:
        with TemporaryDirectory() as temp_dir:
            paths = self._paths(Path(temp_dir))
            paths.train_runs_dir.mkdir(parents=True, exist_ok=True)
            run_dir = paths.train_runs_dir / "medical-run" / "weights"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "best.pt").write_text("weight", encoding="utf-8")

            with patch(
                "medical.training.load_medical_settings",
                return_value={"base_model": "yolo11n.pt", "fallback_model": "yolo11n.pt", "run_name": "medical-run"},
            ):
                model_path = train_medical_model(paths, yolo_cls=_FakeYOLO)

            self.assertTrue(model_path.exists())
            self.assertEqual(model_path.read_text(encoding="utf-8"), "weight")
            sync_mock.assert_called_once()

    @patch("medical.training.resolve_medical_runtime_model_path", side_effect=lambda config: Path(config.model_path))
    def test_validate_medical_model_uses_current_model_path(self, _resolve_runtime_model_mock) -> None:
        with TemporaryDirectory() as temp_dir:
            paths = self._paths(Path(temp_dir))
            paths.trained_model_path.parent.mkdir(parents=True, exist_ok=True)
            paths.trained_model_path.write_text("weight", encoding="utf-8")

            with patch("medical.training.load_medical_settings", return_value={"model": str(paths.trained_model_path)}):
                metrics = validate_medical_model(paths, yolo_cls=_FakeYOLO)

            self.assertEqual(metrics["map50"], 0.77)

    @patch("medical.training.validate_medical_model", return_value={"map50": 0.8})
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
        prepare_mock.return_value = SimpleNamespace(train_count=3, val_count=1, test_count=1)
        train_mock.return_value = fake_paths.trained_model_path

        report = run_full_medical_training_pipeline(yolo_cls=_FakeYOLO)

        self.assertEqual(report["train_count"], 3)
        self.assertEqual(report["trained_model_path"], fake_paths.trained_model_path)
        self.assertEqual(report["validation_metrics"]["map50"], 0.8)

    def test_sync_medical_model_config_updates_model_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "medical_settings.yaml"
            config_path.write_text("medical:\n  model: old.pt\n", encoding="utf-8")

            with patch("medical.training.load_yaml", return_value={"medical": {"model": "old.pt"}}), patch(
                "medical.training.save_yaml"
            ) as save_mock:
                sync_medical_model_config("models/trained/new.pt")

            saved_payload = save_mock.call_args.args[1]
            self.assertEqual(Path(saved_payload["medical"]["model"]), Path("models/trained/new.pt"))

    @patch("medical.training.load_medical_settings")
    def test_medical_training_paths_uses_configured_dataset_and_model_names(self, load_settings_mock) -> None:
        load_settings_mock.return_value = {
            "dataset_root": "dataset/custom_medical",
            "disease_name": "custom_medical",
        }

        paths = medical_training_paths()

        self.assertEqual(paths.dataset_root, Path("dataset/custom_medical"))
        self.assertEqual(paths.raw_images_dir, Path("dataset/custom_medical/raw/images"))
        self.assertEqual(paths.trained_model_path, Path("models/trained/custom_medical_best.pt"))




