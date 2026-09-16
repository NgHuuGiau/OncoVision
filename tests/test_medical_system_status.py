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
            with patch("medical.system_status.build_default_medical_analyzer_config", return_value=config), patch(
                "medical.system_status.resolve_medical_runtime_model_path", return_value=config.model_path
            ):
                status = get_medical_system_status()

            self.assertTrue(status.model_ready)
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
