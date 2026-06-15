from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.chat_ui.paths import build_chat_capture_path


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


if __name__ == "__main__":
    unittest.main()
