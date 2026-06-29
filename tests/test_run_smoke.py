from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run_smoke


class RunSmokeTests(unittest.TestCase):
    def test_select_checks_appends_test_suite_when_requested(self) -> None:
        checks = run_smoke.select_checks(include_tests=True)

        self.assertEqual(checks[-1].key, "test-suite")

    def test_select_checks_ci_safe_filters_out_heavy_checks(self) -> None:
        checks = run_smoke.select_checks(ci_safe=True)

        self.assertTrue(all(check.ci_safe for check in checks))
        self.assertNotIn("doctor", [check.key for check in checks])
        self.assertNotIn("medical-status", [check.key for check in checks])

    def test_select_checks_ci_safe_filters_out_heavy_checks(self) -> None:
        checks = run_smoke.select_checks(ci_safe=True)

        self.assertTrue(all(check.ci_safe for check in checks))
        self.assertNotIn("doctor", [check.key for check in checks])
        self.assertNotIn("medical-status", [check.key for check in checks])

    def test_execute_checks_treats_warning_exit_codes_as_nonfatal(self) -> None:
        checks = (
            run_smoke.SmokeCheck(
                key="chat",
                title="Chat",
                description="test",
                command=("run_chat.py", "--check-only"),
                warning_exit_codes=(2,),
            ),
        )
        run_command = MagicMock(return_value=2)

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = run_smoke.execute_checks(checks, run_command_fn=run_command)

        self.assertEqual(result, 0)
        self.assertIn("WARN", stdout.getvalue())
        run_command.assert_called_once_with(checks[0])

    def test_execute_checks_stops_on_first_failure_when_requested(self) -> None:
        checks = (
            run_smoke.SmokeCheck("one", "One", "test", ("run_app.py", "--advisor-only")),
            run_smoke.SmokeCheck("two", "Two", "test", ("run_train.py", "--check-only")),
            run_smoke.SmokeCheck("three", "Three", "test", ("run_medical.py", "status")),
        )
        run_command = MagicMock(side_effect=[0, 1, 0])

        result = run_smoke.execute_checks(checks, stop_on_fail=True, run_command_fn=run_command)

        self.assertEqual(result, 1)
        self.assertEqual(run_command.call_count, 2)

    @patch("run_smoke.execute_checks")
    @patch("run_smoke.parse_args")
    def test_main_passes_selected_checks_to_executor(
        self,
        parse_args_mock,
        execute_checks_mock,
    ) -> None:
        parse_args_mock.return_value = SimpleNamespace(include_tests=True, stop_on_fail=True, ci_safe=False)
        execute_checks_mock.return_value = 7

        result = run_smoke.main()

        self.assertEqual(result, 7)
        checks = execute_checks_mock.call_args.args[0]
        self.assertEqual(checks[-1].key, "test-suite")
        self.assertTrue(execute_checks_mock.call_args.kwargs["stop_on_fail"])

    @patch("run_smoke.execute_checks")
    @patch("run_smoke.parse_args")
    def test_main_passes_ci_safe_to_check_selector(
        self,
        parse_args_mock,
        execute_checks_mock,
    ) -> None:
        parse_args_mock.return_value = SimpleNamespace(include_tests=False, stop_on_fail=False, ci_safe=True)
        execute_checks_mock.return_value = 0

        run_smoke.main()

        checks = execute_checks_mock.call_args.args[0]
        self.assertTrue(all(check.ci_safe for check in checks))

    @patch("run_smoke.execute_checks")
    @patch("run_smoke.parse_args")
    def test_main_passes_ci_safe_to_check_selector(
        self,
        parse_args_mock,
        execute_checks_mock,
    ) -> None:
        parse_args_mock.return_value = SimpleNamespace(include_tests=False, stop_on_fail=False, ci_safe=True)
        execute_checks_mock.return_value = 0

        run_smoke.main()

        checks = execute_checks_mock.call_args.args[0]
        self.assertTrue(all(check.ci_safe for check in checks))


if __name__ == "__main__":
    unittest.main()
