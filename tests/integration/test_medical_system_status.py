from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from medical.system_status import (
    get_medical_system_status,
    recommended_medical_commands,
)


class MedicalSystemStatusTests(unittest.TestCase):
    def _config(self, root: Path, *, model_path: Path | None = None, brain_model_path: Path | None = None):
        return type(
            "MedicalConfig",
            (),
            {
                "model_path": model_path or root / "models" / "medical.pt",
                "working_dir": root / "output" / "medical",
                "fallback_model_path": None,
                "allow_fallback_model": False,
                "brain_model_path": brain_model_path,
                "target_model_root": root / "models" / "cancers",
            },
        )()

    def test_status_counts_outputs_cases_and_model(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            working = root / "output" / "medical"
            for directory, filename in (
                (working / "reports", "case.json"),
                (working / "normalized_images", "normalized.jpg"),
                (working / "processed_images", "overlay.jpg"),
                (working / "exports", "case.zip"),
            ):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / filename).write_text("x", encoding="utf-8")
            db_path = root / "output" / "onco.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("CREATE TABLE medical_cases (id INTEGER PRIMARY KEY)")
                conn.execute("INSERT INTO medical_cases DEFAULT VALUES")
                conn.commit()
            finally:
                conn.close()

            config = self._config(root)
            config.model_path.parent.mkdir(parents=True, exist_ok=True)
            config.model_path.touch()
            with patch("medical.system_status.build_default_medical_analyzer_config", return_value=config), patch(
                "medical.system_status.resolve_medical_runtime_model_path", return_value=config.model_path
            ):
                status = get_medical_system_status()

            self.assertTrue(status.model_ready)
            self.assertTrue(status.classification_ready)
            self.assertFalse(status.localization_ready)
            self.assertFalse(status.volume_ready)
            self.assertFalse(status.clinical_validation_ready)
            self.assertEqual(status.case_count, 1)
            self.assertEqual(status.report_files, 1)
            self.assertEqual(status.normalized_files, 1)
            self.assertEqual(status.overlay_files, 1)
            self.assertEqual(status.export_files, 1)
            self.assertTrue(any(label == "Ung thư não" and ready for label, ready in status.screening_targets))

    def test_status_uses_brain_model_as_limited_fallback(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brain_model = root / "brain.pt"
            brain_model.touch()
            config = self._config(root, brain_model_path=brain_model)
            with patch("medical.system_status.build_default_medical_analyzer_config", return_value=config), patch(
                "medical.system_status.resolve_medical_runtime_model_path", side_effect=FileNotFoundError("missing general model")
            ):
                status = get_medical_system_status()
            self.assertTrue(status.model_ready)
            self.assertEqual(status.resolved_model_path, brain_model)
            self.assertIn("Chỉ phân tích ảnh vùng não", status.model_message)

    def test_status_tracks_each_target_independently(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = root / "models" / "cancers" / "lung" / "detector.pt"
            detector.parent.mkdir(parents=True)
            detector.touch()
            config = self._config(root)
            with patch("medical.system_status.build_default_medical_analyzer_config", return_value=config), patch(
                "medical.system_status.resolve_medical_runtime_model_path", side_effect=FileNotFoundError("missing")
            ), patch("medical.system_status.probe_runtime_capability", return_value=(True, "")):
                status = get_medical_system_status()
            self.assertTrue(status.target_capabilities["lung"]["runtime_ready"])
            self.assertTrue(status.target_capabilities["lung"]["localization_ready"])
            self.assertFalse(status.target_capabilities["brain"]["runtime_ready"])
            self.assertEqual(status.resolved_model_path, detector)

    def test_status_warns_when_model_sits_at_organ_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = root / "models" / "cancers" / "lung" / "detector.pt"
            detector.parent.mkdir(parents=True)
            detector.touch()
            config = self._config(root)
            with patch("medical.system_status.build_default_medical_analyzer_config", return_value=config), patch(
                "medical.system_status.resolve_medical_runtime_model_path", side_effect=FileNotFoundError("missing")
            ), patch("medical.system_status.probe_runtime_capability", return_value=(True, "")):
                status = get_medical_system_status()
            self.assertIn("cấp cơ quan", status.model_message)
            self.assertIn("fetch_model.py --modality", status.model_message)

    def test_status_has_no_organ_root_warning_when_modality_slot_ready(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = root / "models" / "cancers" / "lung" / "ct" / "detector.pt"
            detector.parent.mkdir(parents=True)
            detector.touch()
            config = self._config(root)
            with patch("medical.system_status.build_default_medical_analyzer_config", return_value=config), patch(
                "medical.system_status.resolve_medical_runtime_model_path", side_effect=FileNotFoundError("missing")
            ), patch("medical.system_status.probe_runtime_capability", return_value=(True, "")):
                status = get_medical_system_status()
            self.assertNotIn("cấp cơ quan", status.model_message)

    def test_status_has_no_organ_root_warning_for_brain(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            classifier = root / "models" / "cancers" / "brain" / "classifier.pt"
            classifier.parent.mkdir(parents=True)
            classifier.touch()
            config = self._config(root)
            with patch("medical.system_status.build_default_medical_analyzer_config", return_value=config), patch(
                "medical.system_status.resolve_medical_runtime_model_path", side_effect=FileNotFoundError("missing")
            ), patch("medical.system_status.probe_runtime_capability", return_value=(True, "")):
                status = get_medical_system_status()
            self.assertNotIn("cấp cơ quan", status.model_message)
            self.assertTrue(status.classification_ready)

    def test_missing_model_recommendations_never_suggest_training(self) -> None:
        status = type(
            "MedicalStatus",
            (),
            {
                "model_ready": False,
                "configured_model_path": Path("medical.pt"),
                "report_files": 0,
                "normalized_files": 0,
                "overlay_files": 0,
                "export_files": 0,
            },
        )()
        commands = recommended_medical_commands(status)
        self.assertEqual(len(commands), 1)
        self.assertIn("model suy luận", commands[0])
        self.assertFalse(any("train-all" in command.lower() or "split-dataset" in command.lower() for command in commands))


if __name__ == "__main__":
    unittest.main()
