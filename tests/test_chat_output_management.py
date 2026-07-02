from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.chat_ui.output_management import chat_output_directories, cleanup_chat_outputs


class ChatOutputManagementTests(unittest.TestCase):
    def test_chat_output_directories_uses_explicit_base_dir(self) -> None:
        base_dir = Path("D:/OncoVision/output/chat")

        dirs = chat_output_directories(base_dir=base_dir)

        self.assertEqual(dirs, [base_dir])

    def test_cleanup_chat_outputs_passes_older_than_days_to_shared_cleanup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with patch("app.chat_ui.output_management.cleanup_directories") as cleanup_mock:
                cleanup_mock.return_value = object()
                cleanup_chat_outputs(older_than_days=14, base_dir=temp_dir)

        cleanup_mock.assert_called_once()
        directories = cleanup_mock.call_args.args[0]
        self.assertEqual(directories, [Path(temp_dir)])
        self.assertEqual(cleanup_mock.call_args.kwargs["older_than_days"], 14)


if __name__ == "__main__":
    unittest.main()
