from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import run_menu


class RunMenuTests(unittest.TestCase):
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

    def test_main_runs_selected_script(self) -> None:
        outputs: list[str] = []
        run_script = MagicMock(return_value=0)
        clear_terminal = MagicMock()
        answers = iter(["4", "0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=run_script,
            clear_terminal_fn=clear_terminal,
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_doctor.py")
        self.assertTrue(any("Quay lại menu" in line for line in outputs))
        clear_terminal.assert_called_once()

    def test_main_runs_training_via_run_train(self) -> None:
        outputs: list[str] = []
        run_script = MagicMock(return_value=0)
        clear_terminal = MagicMock()
        answers = iter(["5", "0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=run_script,
            clear_terminal_fn=clear_terminal,
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_train.py")
        self.assertTrue(any("run_train.py" in line for line in outputs))
        clear_terminal.assert_called_once()

    def test_main_retries_on_invalid_choice(self) -> None:
        outputs: list[str] = []
        answers = iter(["9", "3", "0"])
        run_script = MagicMock(return_value=0)
        clear_terminal = MagicMock()

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=run_script,
            clear_terminal_fn=clear_terminal,
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_tests.py")
        self.assertTrue(any("Lựa chọn không hợp lệ" in line for line in outputs))
        clear_terminal.assert_called_once()

    def test_main_returns_to_menu_after_nonzero_exit(self) -> None:
        outputs: list[str] = []
        answers = iter(["2", "0"])
        run_script = MagicMock(return_value=1)
        clear_terminal = MagicMock()

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=run_script,
            clear_terminal_fn=clear_terminal,
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_chat.py")
        self.assertTrue(any("kết thúc với mã" in line for line in outputs))
        clear_terminal.assert_called_once()

    def test_menu_restores_run_app_entrypoint(self) -> None:
        outputs: list[str] = []
        run_script = MagicMock(return_value=0)
        answers = iter(["1", "0"])

        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )

        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_app.py")
        self.assertTrue(any("run_app.py" in line for line in outputs))
        self.assertFalse(any("run_detect.py" in line for line in outputs))


if __name__ == "__main__":
    unittest.main()
