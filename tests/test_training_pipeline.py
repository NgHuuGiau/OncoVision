from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from training import export_model, train_model, validate_model


class TrainingPipelineTests(unittest.TestCase):
    @patch("training.train_model._auto_prepare_training_dataset")
    @patch("training.train_model._ensure_training_dataset_ready")
    @patch("training.train_model._copy_best_weight")
    @patch("training.train_model.resolve_data_config_path", return_value=Path("training/.generated_data.yaml"))
    @patch("training.train_model.YOLO")
    @patch("training.train_model.load_yaml")
    def test_train_main_falls_back_to_lighter_model(
        self, load_yaml_mock, yolo_mock, resolve_data_config_path_mock, copy_best_mock, ensure_dataset_ready_mock, auto_prepare_mock
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
        ensure_dataset_ready_mock.return_value = None
        auto_prepare_mock.return_value = {"raw_images": 0, "auto_labeled": 0, "eligible": 0, "no_detection": []}

        train_model.main()

        fallback_model.train.assert_called_once()
        kwargs = fallback_model.train.call_args.kwargs
        self.assertEqual(kwargs["model"], "yolo11n.pt")
        self.assertEqual(kwargs["imgsz"], 416)
        self.assertEqual(kwargs["batch"], 4)
        copy_best_mock.assert_called_once()
        auto_prepare_mock.assert_called_once()

    @patch("training.train_model._copy_split")
    @patch("training.train_model._split_items", return_value={"train": [("img", "lbl")], "val": [], "test": []})
    @patch("training.train_model._reset_processed_dirs")
    @patch("training.train_model._save_auto_prepare_state")
    @patch("training.train_model._auto_label_device", return_value="cuda:0")
    @patch("training.train_model._load_auto_prepare_state", return_value=None)
    @patch("training.train_model._dataset_signature", side_effect=["before", "after"])
    @patch("training.train_model.auto_label_raw_images", return_value={"generated": 2, "no_detection": ["c.jpg"]})
    @patch("training.train_model.audit_raw_dataset")
    def test_auto_prepare_training_dataset_auto_labels_and_splits(
        self,
        audit_mock,
        auto_label_mock,
        dataset_signature_mock,
        load_state_mock,
        auto_label_device_mock,
        save_state_mock,
        reset_mock,
        split_items_mock,
        copy_split_mock,
    ) -> None:
        first_audit = SimpleNamespace(raw_image_count=2, missing_labels=[Path("a.jpg")], eligible=[])
        second_audit = SimpleNamespace(raw_image_count=2, missing_labels=[], eligible=[("img", "lbl")])
        audit_mock.side_effect = [first_audit, second_audit]

        report = train_model._auto_prepare_training_dataset()

        auto_label_mock.assert_called_once_with(overwrite=False, conf=0.25, device="cuda:0")
        reset_mock.assert_called_once()
        split_items_mock.assert_called_once_with([("img", "lbl")])
        copy_split_mock.assert_any_call("train", [("img", "lbl")])
        self.assertEqual(copy_split_mock.call_count, 3)
        self.assertEqual(report["raw_images"], 2)
        self.assertEqual(report["auto_labeled"], 2)
        self.assertEqual(report["eligible"], 1)
        self.assertEqual(report["no_detection"], ["c.jpg"])
        self.assertEqual(report["device"], "cuda:0")
        save_state_mock.assert_called_once_with("after")

    @patch("training.train_model._save_auto_prepare_state")
    @patch("training.train_model._load_auto_prepare_state", return_value="same-signature")
    @patch("training.train_model._dataset_signature", return_value="same-signature")
    @patch("training.train_model.auto_label_raw_images")
    @patch("training.train_model.audit_raw_dataset")
    @patch("training.train_model.PROCESSED_VAL_DIR")
    @patch("training.train_model.PROCESSED_TRAIN_DIR")
    @patch("training.train_model._reset_processed_dirs")
    def test_auto_prepare_training_dataset_skips_rebuild_when_dataset_unchanged(
        self,
        reset_mock,
        processed_train_mock,
        processed_val_mock,
        audit_mock,
        auto_label_mock,
        dataset_signature_mock,
        load_state_mock,
        save_state_mock,
    ) -> None:
        processed_train_mock.exists.return_value = True
        processed_train_mock.iterdir.return_value = iter([Path("train-item")])
        processed_val_mock.exists.return_value = True
        processed_val_mock.iterdir.return_value = iter([Path("val-item")])
        audit_mock.return_value = SimpleNamespace(raw_image_count=3, missing_labels=[], eligible=[("img", "lbl")])

        report = train_model._auto_prepare_training_dataset()

        self.assertTrue(report["skipped_rebuild"])
        self.assertEqual(report["eligible"], 1)
        auto_label_mock.assert_not_called()
        reset_mock.assert_not_called()
        save_state_mock.assert_not_called()

    def test_copy_best_weight_copies_when_present(self) -> None:
        with TemporaryDirectory(dir="D:\\OncoVision") as temp_dir:
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
