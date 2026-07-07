from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from training import prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    @patch("training.prepare_dataset.prepare_medical_training_dataset")
    @patch("training.prepare_dataset.ensure_utf8_console")
    def test_main_prints_medical_summary(self, ensure_console_mock, prepare_mock) -> None:
        prepare_mock.return_value = SimpleNamespace(
            class_count=7,
            train_count=10,
            val_count=4,
            test_count=2,
            total_count=16,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            prepare_dataset.main()

        ensure_console_mock.assert_called_once()
        prepare_mock.assert_called_once_with()
        text = output.getvalue()
        self.assertIn("Medical dataset ready", text)
        self.assertIn("Classes: 7", text)
        self.assertIn("Train/val/test: 10/4/2", text)


if __name__ == "__main__":
    unittest.main()
