from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.chat_ui.paths import build_chat_capture_path
from utils.cleanup_utils import cleanup_directories


class ChatAiAppPathTests(unittest.TestCase):
    def test_build_chat_capture_path_uses_requested_directory_and_unique_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = build_chat_capture_path(base_dir=temp_dir)
            second_path = build_chat_capture_path(base_dir=temp_dir)

        self.assertEqual(first_path.parent, Path(temp_dir))
        self.assertEqual(second_path.parent, Path(temp_dir))
        self.assertTrue(first_path.name.startswith("camera_capture_"))
        self.assertTrue(second_path.name.startswith("camera_capture_"))
        self.assertEqual(first_path.suffix, ".png")
        self.assertEqual(second_path.suffix, ".png")
        self.assertNotEqual(first_path.name, second_path.name)

    def test_cleanup_chat_outputs_removes_old_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            capture_path = Path(temp_dir) / "camera_capture_test.png"
            capture_path.write_text("x", encoding="utf-8")

            summary = cleanup_directories([Path(temp_dir)], older_than_days=None)

        self.assertEqual(summary.removed_files, 1)
        self.assertFalse(capture_path.exists())


if __name__ == "__main__":
    unittest.main()
