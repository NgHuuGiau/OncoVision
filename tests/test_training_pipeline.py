from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from training import export_model, train_model, validate_model


class TrainingPipelineTests(unittest.TestCase):
    @patch("training.train_model.PROCESSED_VAL_DIR")
    @patch("training.train_model.PROCESSED_TRAIN_DIR")
    @patch("training.train_model._copy_best_weight")
    @patch("training.train_model.resolve_data_config_path", return_value=Path("training/.generated_data.yaml"))
    @patch("training.train_model.YOLO")
    @patch("training.train_model.load_yaml")
    def test_train_main_falls_back_to_lighter_model(
        self, load_yaml_mock, yolo_mock, resolve_data_config_path_mock, copy_best_mock, processed_train_mock, processed_val_mock
    ) -> None:
        load_yaml_mock.return_value = {
            "model": "yolo11s.pt",
            "fallback_model": "yolo11n.pt",
            "data": "training/data.yaml",
            "epochs": 2,
            "imgsz": 512,
            "batch": 8,
            "device": 0,
            "workers": 2,
            "cache": False,
            "amp": True,
            "patience": 20,
            "project": "runs/train",
            "name": "test-run",
        }

        primary_model = MagicMock()
        primary_model.train.side_effect = RuntimeError("oom")
        fallback_model = MagicMock()
        fallback_model.train.return_value = SimpleNamespace(save_dir="runs/train/test-run")
        yolo_mock.side_effect = [primary_model, fallback_model]
        processed_train_mock.exists.return_value = True
        processed_train_mock.iterdir.return_value = iter([Path("train-item")])
        processed_val_mock.exists.return_value = True
        processed_val_mock.iterdir.return_value = iter([Path("val-item")])

        train_model.main()

        fallback_model.train.assert_called_once()
        kwargs = fallback_model.train.call_args.kwargs
        self.assertEqual(kwargs["model"], "yolo11n.pt")
        self.assertEqual(kwargs["imgsz"], 416)
        self.assertEqual(kwargs["batch"], 4)
        copy_best_mock.assert_called_once()

    def test_copy_best_weight_copies_when_present(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            best_path = weights_dir / "best.pt"
            best_path.write_text("fake-weight", encoding="utf-8")
            target = run_dir / "trained-best.pt"
            with patch("training.train_model.TRAINED_BEST_MODEL_PATH", target):
                train_model._copy_best_weight(run_dir)
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "fake-weight")

    @patch("training.validate_model._ensure_validation_dataset_ready")
    @patch("training.validate_model.resolve_data_config_path", return_value=Path("training/.generated_data.yaml"))
    @patch("training.validate_model.YOLO")
    def test_validate_model_falls_back_to_yolo11n_when_best_missing(self, yolo_mock, resolve_data_config_path_mock, ensure_validation_ready_mock) -> None:
        model = MagicMock()
        yolo_mock.return_value = model
        ensure_validation_ready_mock.return_value = None
        with patch("training.validate_model.resolve_trained_model_path", return_value=Path("yolo11n.pt")):
            validate_model.main()
        yolo_mock.assert_called_once_with(str(Path("models/pretrained/yolo11n.pt")))
        model.val.assert_called_once()

    @patch("training.export_model.YOLO")
    def test_export_model_uses_best_weight(self, yolo_mock) -> None:
        model = MagicMock()
        yolo_mock.return_value = model
        with patch("training.export_model.resolve_trained_model_path", return_value=Path("models/trained/best.pt")):
            export_model.main()
        yolo_mock.assert_called_once_with(str(Path("models/trained/best.pt")))
        model.export.assert_called_once_with(format="onnx")
