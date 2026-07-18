from __future__ import annotations

import os
import re
import unittest
from unittest.mock import MagicMock, patch

import run_menu
from run_menu import MENU_OPTIONS

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class RunMenuTests(unittest.TestCase):
    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_main_auto_downloads_missing_yolo_models_on_startup(self, exists_mock, download_models_mock) -> None:
        def exists_side_effect(path: str) -> bool:
            return path.endswith(("yolo11n.pt", "yolo11s.pt"))

        exists_mock.side_effect = exists_side_effect
        download_models_mock.return_value = (["yolo11m.pt", "yolo11l.pt", "yolo11x.pt"], [])

        answers = iter(["0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=lambda _: None,
            run_script_fn=MagicMock(),
            clear_terminal_fn=MagicMock(),
        )

        self.assertEqual(result, 0)
        download_models_mock.assert_called_once_with(["yolo11m.pt", "yolo11l.pt", "yolo11x.pt"])

    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_main_does_not_download_when_all_yolo_models_exist(self, exists_mock, download_models_mock) -> None:
        exists_mock.return_value = True

        result = run_menu.main(
            input_fn=lambda _: "0",
            print_fn=lambda _: None,
            run_script_fn=MagicMock(),
            clear_terminal_fn=MagicMock(),
        )

        self.assertEqual(result, 0)
        download_models_mock.assert_not_called()

    def test_main_exits_on_zero(self) -> None:
        outputs: list[str] = []
        clear_terminal = MagicMock()
        result = run_menu.main(
            input_fn=lambda _: "0",
            print_fn=outputs.append,
            run_script_fn=MagicMock(),
            clear_terminal_fn=clear_terminal,
        )

        self.assertEqual(result, 0)
        self.assertTrue(any("Đã thoát menu" in line for line in outputs))
        clear_terminal.assert_not_called()

    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_main_runs_camera_entrypoint(self, exists_mock, download_models_mock) -> None:
        exists_mock.return_value = True
        run_script = MagicMock(return_value=0)
        answers = iter(["1", "0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=lambda _: None,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_app.py")

    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_main_enters_check_menu_and_runs_doctor_option(self, exists_mock, download_models_mock) -> None:
        exists_mock.return_value = True
        outputs: list[str] = []
        run_script = MagicMock(return_value=0)
        clear_terminal = MagicMock()
        answers = iter(["5", "1", "0", "0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=run_script,
            clear_terminal_fn=clear_terminal,
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_doctor.py", "--skip-camera-check")
        self.assertTrue(any("run_doctor.py --skip-camera-check" in line for line in outputs))
        self.assertEqual(clear_terminal.call_count, 2)

    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_main_enters_check_menu_and_runs_test_option(self, exists_mock, download_models_mock) -> None:
        exists_mock.return_value = True
        outputs: list[str] = []
        run_script = MagicMock(return_value=0)
        answers = iter(["5", "2", "0", "0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_tests.py", "--skip-camera-check")
        self.assertTrue(any("run_tests.py --skip-camera-check" in line for line in outputs))

    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_main_enters_check_menu_and_runs_smoke_check_option(self, exists_mock, download_models_mock) -> None:
        exists_mock.return_value = True
        outputs: list[str] = []
        run_script = MagicMock(return_value=0)
        clear_terminal = MagicMock()
        answers = iter(["5", "3", "0", "0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=run_script,
            clear_terminal_fn=clear_terminal,
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_smoke.py")
        self.assertTrue(any("run_smoke.py" in line for line in outputs))
        self.assertEqual(clear_terminal.call_count, 2)

    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_main_enters_check_menu_and_runs_smoke_plus_tests_option(self, exists_mock, download_models_mock) -> None:
        exists_mock.return_value = True
        outputs: list[str] = []
        run_script = MagicMock(return_value=0)
        answers = iter(["5", "4", "0", "0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_smoke.py", "--include-tests")
        self.assertTrue(any("run_smoke.py --include-tests" in line for line in outputs))

    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_main_runs_chat_cleanup_option(self, exists_mock, download_models_mock) -> None:
        exists_mock.return_value = True
        outputs: list[str] = []
        run_script = MagicMock(return_value=0)
        answers = iter(["6", "0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_chat.py", "--cleanup-output")
        self.assertTrue(any("run_chat.py --cleanup-output" in line for line in outputs))

    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_main_retries_on_invalid_choice(self, exists_mock, download_models_mock) -> None:
        exists_mock.return_value = True
        outputs: list[str] = []
        answers = iter(["13", "5", "3", "0", "0"])
        run_script = MagicMock(return_value=0)

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_smoke.py")
        self.assertTrue(any("Lựa chọn không hợp lệ" in line for line in outputs))

    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_check_menu_retries_on_invalid_choice(self, exists_mock, download_models_mock) -> None:
        exists_mock.return_value = True
        outputs: list[str] = []
        answers = iter(["5", "9", "3", "0", "0"])
        run_script = MagicMock(return_value=0)

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_smoke.py")
        self.assertTrue(any("Lựa chọn không hợp lệ" in line for line in outputs))

    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_main_enters_medical_menu_and_runs_report(self, exists_mock, download_models_mock) -> None:
        exists_mock.return_value = True
        outputs: list[str] = []
        run_script = MagicMock(return_value=0)
        answers = iter(["3", "1", "0", "0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_medical.py", "report")
        self.assertTrue(any("Báo cáo nhanh" in line or "report" in line for line in outputs))

    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_main_enters_medical_menu_and_runs_train_all(self, exists_mock, download_models_mock) -> None:
        exists_mock.return_value = True
        run_script = MagicMock(return_value=0)
        answers = iter(["3", "3", "0", "0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=lambda _: None,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_medical.py", "train-all", env={"PYTHONUNBUFFERED": "1"})

    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_main_enters_medical_menu_and_runs_improvement_flow(self, exists_mock, download_models_mock) -> None:
        exists_mock.return_value = True
        run_script = MagicMock(return_value=0)
        answers = iter(["3", "6", "0", "0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=lambda _: None,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )

        self.assertEqual(result, 0)
        called = [call.args for call in run_script.call_args_list]
        self.assertTrue(any("active-learning" in call for call in called))
        self.assertTrue(any("train-modality" in call for call in called))
        self.assertTrue(any("calibrate-modality-tuning" in call for call in called))

    @patch("run_menu.download_models")
    @patch("run_menu.os.path.exists")
    def test_medical_analyze_prompts_for_path_and_patient_code(self, exists_mock, download_models_mock) -> None:
        exists_mock.return_value = True
        run_script = MagicMock(return_value=0)
        answers = iter(["3", "4", "sample.jpg", "BN009", "0", "0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=lambda _: None,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_medical.py", "analyze", "--image", "sample.jpg", "--patient-code", "BN009")

    def test_render_menu_wraps_long_descriptions_on_narrow_terminal(self) -> None:
        outputs: list[str] = []

        with patch("run_menu.os.get_terminal_size", return_value=os.terminal_size((60, 20))):
            run_menu._render_menu(print_fn=outputs.append)

        plain_outputs = [ANSI_RE.sub("", item) for item in outputs]
        non_empty_lines = [line for line in plain_outputs if line.strip()]
        self.assertTrue(len(non_empty_lines) > len(MENU_OPTIONS))
        menu_lines = [line for line in non_empty_lines if line.strip()[0] in "0123456789"]
        self.assertTrue(
            all(len(line) <= 60 for line in menu_lines),
            msg="\n".join(plain_outputs),
        )


if __name__ == "__main__":
    unittest.main()
