from __future__ import annotations

import io
import unittest
from unittest.mock import Mock, patch

import run_tests


class DummyPassingTest(unittest.TestCase):
    def test_ok(self) -> None:
        self.assertTrue(True)


class RunTestsDashboardTests(unittest.TestCase):
    def test_meter_uses_unicode_characters(self) -> None:
        text = run_tests.meter(3, 5, width=5)
        self.assertIn("█", text)
        self.assertIn("·", text)
        self.assertNotIn("â", text)
        self.assertNotIn("Â", text)

    def test_pretty_runner_renders_vietnamese_summary(self) -> None:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(DummyPassingTest)
        stream = io.StringIO()
        runner = run_tests.PrettyTestRunner(verbosity=0, total_tests=suite.countTestCases(), stream=stream)

        result = runner.run(suite)
        output = stream.getvalue()

        self.assertTrue(result.wasSuccessful())
        self.assertIn("TỔNG QUAN", output)
        self.assertIn("KẾT QUẢ CUỐI", output)
        self.assertIn("TOÀN BỘ TEST PASS", output)

    @patch("run_tests._open_camera_capture")
    def test_check_camera_reports_warn_when_camera_cannot_open(self, open_camera_mock) -> None:
        capture = Mock()
        capture.isOpened.return_value = False
        open_camera_mock.return_value = capture

        result = run_tests.check_camera(index=1)

        self.assertEqual(result.level, "WARN")
        self.assertIn("Không mở được camera index 1", result.summary)
        capture.release.assert_called_once()

    @patch("run_tests._open_camera_capture")
    def test_check_camera_reports_pass_when_frame_is_read(self, open_camera_mock) -> None:
        frame = Mock()
        frame.shape = (480, 640, 3)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True, frame)]
        open_camera_mock.return_value = capture

        result = run_tests.check_camera(index=0)

        self.assertEqual(result.level, "PASS")
        self.assertIn("Đọc frame thành công", result.summary)
        self.assertIn("640x480", result.detail)
        capture.release.assert_called_once()
