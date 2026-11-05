from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.cleanup_utils import cleanup_directories


class ChatOutputManagementTests(unittest.TestCase):
    def test_cleanup_chat_outputs_passes_older_than_days_to_shared_cleanup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture_path = Path(temp_dir) / "camera_capture_test.png"
            capture_path.write_text("x", encoding="utf-8")

            summary = cleanup_directories([Path(temp_dir)], older_than_days=None)

        self.assertEqual(summary.removed_files, 1)
        self.assertFalse(capture_path.exists())


if __name__ == "__main__":
    unittest.main()
