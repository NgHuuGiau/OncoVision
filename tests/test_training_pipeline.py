from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

from training import export_model, train_model, validate_model


class TrainingPipelineTests(unittest.TestCase):
    @patch("training.train_model.train_medical_model", return_value=Path("medical_7_cancers.pt"))
    def test_train_model_main_calls_medical_training(self, train_medical_model_mock) -> None:
        result = train_model.main()

        self.assertEqual(result, Path("medical_7_cancers.pt"))
        train_medical_model_mock.assert_called_once_with()

    @patch("training.validate_model.validate_medical_model")
    def test_validate_model_main_calls_medical_validation(self, validate_medical_model_mock) -> None:
        validate_medical_model_mock.return_value = {"accuracy": 0.75, "model_path": Path("medical_7_cancers.pt")}

        result = validate_model.main()

        self.assertEqual(result["accuracy"], 0.75)
        validate_medical_model_mock.assert_called_once_with()

    @patch("training.export_model.medical_training_paths")
    @patch("training.export_model.ensure_utf8_console")
    def test_export_model_copies_medical_model(self, ensure_console_mock, medical_training_paths_mock) -> None:
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "medical_7_cancers.pt"
            source_path.write_text("model-bytes", encoding="utf-8")
            medical_training_paths_mock.return_value = SimpleNamespace(trained_model_path=source_path)

            export_model.main()

            export_path = Path(temp_dir) / "medical_7_cancers_exported.pt"
            self.assertTrue(export_path.exists())
            self.assertEqual(export_path.read_text(encoding="utf-8"), "model-bytes")
        ensure_console_mock.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
