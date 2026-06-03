from __future__ import annotations

import unittest

from utils.console_ui import explain_runtime_failure, progress_bar_colored


class ConsoleUiTests(unittest.TestCase):
    def test_progress_bar_uses_unicode_bar_and_dot(self) -> None:
        bar = progress_bar_colored(50, width=6)
        self.assertIn("█", bar)
        self.assertIn("·", bar)
        self.assertNotIn("â", bar)
        self.assertNotIn("Â", bar)

    def test_explain_runtime_failure_for_camera_error(self) -> None:
        reason, suggestions, commands = explain_runtime_failure(RuntimeError("Không mở được camera."))
        self.assertTrue("webcam" in reason.lower() or "camera" in reason.lower())
        self.assertTrue(any("camera index" in suggestion.lower() or "webcam" in suggestion.lower() for suggestion in suggestions))
        self.assertTrue(any("--camera-index 1" in command for command in commands))
