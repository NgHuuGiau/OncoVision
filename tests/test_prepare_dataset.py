from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from training import prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    @patch("training.prepare_dataset.ensure_project_directories")
    def test_main_prints_help_without_touching_filesystem(self, ensure_dirs_mock) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            prepare_dataset.main()

        ensure_dirs_mock.assert_called_once()
        text = output.getvalue()
        self.assertIn("YOLO DATASET :: CHUẨN BỊ THƯ MỤC", text)
        self.assertIn("Đã tạo sẵn các thư mục dataset.", text)
        self.assertIn("dataset/raw", text)
        self.assertIn("dataset/processed", text)


if __name__ == "__main__":
    unittest.main()
