from __future__ import annotations

import re
import unittest
from unittest.mock import MagicMock, patch

import run_menu

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class RunMenuTests(unittest.TestCase):
    def test_main_exits_without_downloading_models(self) -> None:
        run_script = MagicMock()
        result = run_menu.main(
            input_fn=lambda _: "0",
            print_fn=lambda _: None,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )
        self.assertEqual(result, 0)
        run_script.assert_not_called()

    def test_medical_analyze_prompts_for_image_and_patient_code(self) -> None:
        run_script = MagicMock(return_value=0)
        answers = iter(["3", "1", "sample.jpg", "BN009", "0", "0"])
        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=lambda _: None,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )
        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_medical.py", "analyze", "--image", "sample.jpg", "--patient-code", "BN009")

    def test_medical_menu_does_not_offer_training(self) -> None:
        outputs: list[str] = []
        run_menu._render_medical_menu(print_fn=outputs.append)
        rendered = ANSI_RE.sub("", "\n".join(outputs)).lower()
        self.assertIn("phân tích ảnh", rendered)
        self.assertNotIn("huấn luyện", rendered)
        self.assertNotIn("tuning", rendered)

    def test_main_enters_medical_menu_and_runs_report(self) -> None:
        run_script = MagicMock(return_value=0)
        answers = iter(["3", "3", "0", "0"])
        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=lambda _: None,
            run_script_fn=run_script,
            clear_terminal_fn=MagicMock(),
        )
        self.assertEqual(result, 0)
        run_script.assert_called_once_with("run_medical.py", "report")

    def test_invalid_main_menu_choice_is_rejected(self) -> None:
        outputs: list[str] = []
        answers = iter(["99", "0"])
        result = run_menu.main(
            input_fn=lambda _: next(answers),
            print_fn=outputs.append,
            run_script_fn=MagicMock(),
            clear_terminal_fn=MagicMock(),
        )
        self.assertEqual(result, 0)
        self.assertTrue(any("Lựa chọn không hợp lệ" in line for line in outputs))

    def test_render_menu_wraps_long_descriptions(self) -> None:
        outputs: list[str] = []
        with patch("run_menu.os.get_terminal_size", return_value=type("Size", (), {"columns": 60})()):
            run_menu._render_menu(print_fn=outputs.append)
        lines = [ANSI_RE.sub("", item) for item in outputs if ANSI_RE.sub("", item).strip()]
        self.assertGreater(len(lines), len(run_menu.MENU_OPTIONS))
        self.assertTrue(all(len(line) <= 60 for line in lines))


if __name__ == "__main__":
    unittest.main()
