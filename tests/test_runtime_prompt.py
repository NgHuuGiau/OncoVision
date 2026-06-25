from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.runtime_prompt import prompt_runtime_mode as tool_prompt_runtime_mode, prompt_runtime_model
from utils.console_ui import mode_to_ui_defaults, prompt_launch_target, prompt_runtime_mode as console_prompt_runtime_mode


class RuntimePromptTests(unittest.TestCase):
    def test_prompt_runtime_mode_exits_on_zero(self) -> None:
        answers = iter(["0"])
        printed: list[str] = []
        with self.assertRaises(SystemExit) as ctx:
            console_prompt_runtime_mode(input_fn=lambda _: next(answers), print_fn=printed.append)
        self.assertEqual(ctx.exception.code, 0)

    def test_prompt_runtime_mode_accepts_valid_choice(self) -> None:
        answers = iter(["2"])
        printed: list[str] = []
        mode = console_prompt_runtime_mode(input_fn=lambda _: next(answers), print_fn=printed.append)
        self.assertEqual(mode, "medium")
        self.assertTrue(any("OncoVision" in line for line in printed))

    def test_prompt_runtime_mode_retries_on_invalid_choice(self) -> None:
        answers = iter(["9", "", "3"])
        printed: list[str] = []
        mode = console_prompt_runtime_mode(input_fn=lambda _: next(answers), print_fn=printed.append)
        self.assertEqual(mode, "low")
        self.assertGreater(len(printed), 1)

    def test_prompt_launch_target_accepts_camera(self) -> None:
        answers = iter(["2"])
        printed: list[str] = []
        target = prompt_launch_target(
            selected_mode="medium",
            selected_model="yolo11s.pt",
            preferred_target="camera",
            input_fn=lambda _: next(answers),
            print_fn=printed.append,
        )
        self.assertEqual(target, "camera")
        self.assertGreater(len(printed), 0)

    def test_mode_to_ui_defaults_maps_values(self) -> None:
        self.assertEqual(mode_to_ui_defaults("auto"), ("auto", "medium"))
        self.assertEqual(mode_to_ui_defaults("high"), ("manual", "high"))

    def test_prompt_runtime_model_accepts_recommended_choice(self) -> None:
        answers = iter(["1"])
        printed: list[str] = []
        model = prompt_runtime_model(
            selected_mode="medium",
            recommendations={
                "medium": type(
                    "Runtime",
                    (),
                    {"primary_model_name": "yolo11s.pt", "candidate_models": ["yolo11s.pt", "yolo11n.pt"]},
                )()
            },
            input_fn=lambda _: next(answers),
            print_fn=printed.append,
        )
        self.assertEqual(model, "yolo11s.pt")
        self.assertTrue(any("CHỌN MODEL SẼ CHẠY" in line for line in printed))

    @patch("core.runtime_prompt._available_models", return_value=(["yolo11s.pt"], ["yolo11x.pt"]))
    @patch(
        "core.runtime_prompt.load_yaml_cached",
        return_value={"preferred_models": {"primary_gpu": "yolo11s.pt"}, "priority_order": ["models/pretrained/yolo11s.pt"]},
    )
    @patch(
        "core.runtime_prompt.load_settings",
        return_value={
            "models": {
                "high": {"model": "yolo11l.pt", "imgsz": 768},
                "medium": {"model": "yolo11s.pt", "imgsz": 640},
                "low": {"model": "yolo11n.pt", "imgsz": 416},
            }
        },
    )
    def test_tool_prompt_runtime_mode_shows_runtime_advisor_and_accepts_choice(
        self,
        _load_settings_mock,
        _load_yaml_mock,
        _available_models_mock,
    ) -> None:
        answers = iter(["2"])
        printed: list[str] = []
        hardware = SimpleNamespace(
            cpu_name="Intel Core i7",
            ram_gb=16.0,
            gpu_name="RTX 3050",
            vram_gb=4.0,
            cuda_available=True,
            os_name="Windows 11",
            torch_version="2.0",
            torch_cuda_version="12.1",
            cpu_usage_percent=25.0,
            ram_usage_percent=40.0,
            gpu_usage_percent=30.0,
            vram_usage_percent=35.0,
        )

        def runtime(mode: str, model: str, device: str, imgsz: int, max_det: int):
            return SimpleNamespace(
                mode=mode,
                primary_model_name=model,
                resolved_device=device,
                imgsz=imgsz,
                max_det=max_det,
                candidate_models=[model],
            )

        recommendations = {
            "high": runtime("high", "yolo11l.pt", "cuda:0", 768, 180),
            "medium": runtime("medium", "yolo11s.pt", "cuda:0", 640, 150),
            "low": runtime("low", "yolo11n.pt", "cuda:0", 416, 100),
            "auto": runtime("medium", "yolo11s.pt", "cuda:0", 640, 150),
        }

        mode = tool_prompt_runtime_mode(
            hardware=hardware,
            recommendations=recommendations,
            input_fn=lambda _: next(answers),
            print_fn=printed.append,
        )

        self.assertEqual(mode, "medium")
        self.assertTrue(any("BỘ TƯ VẤN RUNTIME OncoVision" in line for line in printed))
        self.assertTrue(any("CHỌN CHẾ ĐỘ SẼ CHẠY" in line for line in printed))
        self.assertTrue(any("Model / Imgsz" in line for line in printed))
        self.assertTrue(any("Nên chọn ngay" in line for line in printed))


if __name__ == "__main__":
    unittest.main()
