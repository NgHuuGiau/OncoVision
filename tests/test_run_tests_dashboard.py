from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, Mock, patch

import run_tests


class DummyPassingTest(unittest.TestCase):
    def test_ok(self) -> None:
        self.assertTrue(True)


class RunTestsDashboardTests(unittest.TestCase):
    def test_meter_renders_requested_fill_pattern(self) -> None:
        text = run_tests.meter(3, 5, width=5, filled_char="#", empty_char="-")
        self.assertEqual(text, "# # # - -")

    def test_pretty_runner_renders_summary_sections(self) -> None:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(DummyPassingTest)
        stream = io.StringIO()
        runner = run_tests.PrettyTestRunner(verbosity=0, total_tests=suite.countTestCases(), stream=stream)

        result = runner.run(suite)
        output = stream.getvalue()

        self.assertTrue(result.wasSuccessful())
        self.assertIn("SYSTEM TEST DASHBOARD", output)
        self.assertIn("TEST PASS", output)

    @patch("run_tests._open_camera_capture")
    def test_check_camera_reports_warn_when_camera_cannot_open(self, open_camera_mock) -> None:
        capture = Mock()
        capture.isOpened.return_value = False
        open_camera_mock.return_value = capture

        result = run_tests.check_camera(index=1)

        self.assertEqual(result.level, "WARN")
        self.assertIn("camera index 1", result.summary)
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
        self.assertIn("index 0", result.summary)
        self.assertIn("640x480", result.detail)
        capture.release.assert_called_once()

    @patch("run_tests.PrettyTestRunner")
    @patch("run_tests.unittest.defaultTestLoader.discover")
    @patch("run_tests.ensure_project_directories")
    @patch("run_tests.parse_args")
    @patch("run_tests.check_camera")
    def test_main_skips_camera_probe_when_requested(
        self,
        check_camera_mock,
        parse_args_mock,
        _ensure_dirs_mock,
        discover_mock,
        runner_cls_mock,
    ) -> None:
        parse_args_mock.return_value = Mock(camera_index=0, skip_camera_check=True, strict_camera=False)
        suite = MagicMock()
        suite.countTestCases.return_value = 3
        discover_mock.return_value = suite
        result = MagicMock()
        result.wasSuccessful.return_value = True
        runner = MagicMock()
        runner.run.return_value = result
        runner_cls_mock.return_value = runner

        exit_code = run_tests.main()

        self.assertEqual(exit_code, 0)
        check_camera_mock.assert_not_called()
        self.assertIsNone(runner_cls_mock.call_args.kwargs["camera_result"])

    @patch("run_tests.PrettyTestRunner")
    @patch("run_tests.unittest.defaultTestLoader.discover")
    @patch("run_tests.ensure_project_directories")
    @patch("run_tests.parse_args")
    @patch("run_tests.check_camera")
    def test_main_returns_failure_when_strict_camera_check_warns(
        self,
        check_camera_mock,
        parse_args_mock,
        _ensure_dirs_mock,
        discover_mock,
        runner_cls_mock,
    ) -> None:
        parse_args_mock.return_value = Mock(camera_index=1, skip_camera_check=False, strict_camera=True)
        suite = MagicMock()
        suite.countTestCases.return_value = 5
        discover_mock.return_value = suite
        camera_result = run_tests.CameraCheckResult(level="WARN", summary="warn", detail="detail")
        check_camera_mock.return_value = camera_result
        result = MagicMock()
        result.wasSuccessful.return_value = True
        runner = MagicMock()
        runner.run.return_value = result
        runner_cls_mock.return_value = runner

        exit_code = run_tests.main()

        self.assertEqual(exit_code, 1)
        check_camera_mock.assert_called_once_with(index=1)
        self.assertIs(runner_cls_mock.call_args.kwargs["camera_result"], camera_result)
        self.assertTrue(runner_cls_mock.call_args.kwargs["strict_camera"])

    @patch("run_tests.PrettyTestRunner")
    @patch("run_tests.unittest.defaultTestLoader.discover")
    @patch("run_tests.ensure_project_directories")
    @patch("run_tests.parse_args")
    @patch("run_tests.check_camera")
    def test_main_returns_failure_when_test_suite_fails(
        self,
        check_camera_mock,
        parse_args_mock,
        _ensure_dirs_mock,
        discover_mock,
        runner_cls_mock,
    ) -> None:
        parse_args_mock.return_value = Mock(camera_index=0, skip_camera_check=False, strict_camera=False)
        suite = MagicMock()
        suite.countTestCases.return_value = 2
        discover_mock.return_value = suite
        check_camera_mock.return_value = run_tests.CameraCheckResult(level="PASS", summary="ok", detail="ok")
        result = MagicMock()
        result.wasSuccessful.return_value = False
        runner = MagicMock()
        runner.run.return_value = result
        runner_cls_mock.return_value = runner

        exit_code = run_tests.main()

        self.assertEqual(exit_code, 1)
