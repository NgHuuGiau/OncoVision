from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import run_app
import run_chat
import run_train
from utils.console_ui import mode_to_ui_defaults


class RunEntrypointsTests(unittest.TestCase):
    @patch("run_app.resolve_start_bundle")
    def test_run_app_start_bundle_uses_runtime_tool_prompt(self, resolve_start_bundle_mock) -> None:
        run_app.resolve_run_app_start_bundle(
            requested_mode="high",
            requested_model="yolo11l.pt",
            requested_target="camera",
            preferred_target="camera",
        )

        resolve_start_bundle_mock.assert_called_once_with(
            requested_mode="high",
            requested_model="yolo11l.pt",
            requested_target="camera",
            preferred_target="camera",
            prompt_runtime_mode_fn=run_app.prompt_runtime_mode,
        )

    @patch("run_app.BootProgress")
    @patch("run_app.run_camera_session")
    @patch("run_app.print_runtime_dashboard")
    @patch("run_app.resolve_run_app_start_bundle")
    @patch("run_app.parse_args")
    def test_run_app_main_runs_camera_only_flow(
        self,
        parse_args_mock,
        resolve_start_bundle_mock,
        print_dashboard_mock,
        run_camera_session_mock,
        boot_progress_mock,
    ) -> None:
        progress = boot_progress_mock.return_value
        args = type("Args", (), {"mode": None, "model": None, "camera_index": 2})()
        parse_args_mock.return_value = args
        runtime = SimpleNamespace(show_fps=True)
        hardware = object()
        resolve_start_bundle_mock.return_value = SimpleNamespace(
            selected_mode="high",
            selected_model="models/trained/best.pt",
            launch_target="camera",
            runtime=runtime,
            hardware=hardware,
        )

        exit_code = run_app.main()

        self.assertEqual(exit_code, 0)
        resolve_start_bundle_mock.assert_called_once_with(
            requested_mode=None,
            requested_model=None,
            requested_target="camera",
            preferred_target="camera",
        )
        boot_progress_mock.assert_called_once_with("OncoVision Camera Realtime")
        progress.advance_to.assert_any_call(16, "Dang nhan cau hinh khoi dong")
        progress.advance_to.assert_any_call(42, "Dang kiem tra CPU / GPU / CUDA")
        progress.advance_to.assert_any_call(68, "Dang chon model va runtime phu hop")
        progress.advance_to.assert_any_call(88, "Dang chuan bi mo camera")
        progress.finish.assert_called_once_with("San sang mo camera")
        print_dashboard_mock.assert_called_once_with(
            title="OncoVision Camera Realtime",
            runtime=runtime,
            hardware=hardware,
            camera_index=2,
            launch_target="camera",
        )
        run_camera_session_mock.assert_called_once_with(runtime=runtime, camera_index=2)

    @patch("run_train.main")
    def test_run_train_module_exposes_training_main(self, train_main_mock) -> None:
        run_train.main()
        train_main_mock.assert_called_once()

    @patch("run_train.resolve_model_source", side_effect=lambda value: Path(value))
    @patch("run_train.load_yaml")
    @patch("run_train.module_available")
    @patch("run_train.ensure_project_directories")
    def test_run_train_preflight_warns_when_models_missing_but_does_not_fail(
        self,
        ensure_dirs_mock,
        module_available_mock,
        load_yaml_mock,
        _resolve_model_source_mock,
    ) -> None:
        module_available_mock.return_value = True
        load_yaml_mock.return_value = {"model": "models/pretrained/yolo11s.pt", "fallback_model": "yolo11n.pt"}

        with patch("run_train._count_files", side_effect=[0, 0, 0, 0]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = run_train.run_train_preflight()

        self.assertEqual(result, 0)
        ensure_dirs_mock.assert_called_once_with()
        self.assertIn("Warning: thiếu pretrained model", stdout.getvalue())
        self.assertIn("training\\download_models.py", stdout.getvalue())

    @patch("run_chat.launch_chat_app")
    @patch("run_chat.build_chat_arg_parser")
    def test_run_chat_main_forces_medium_mode(
        self,
        build_parser_mock,
        launch_chat_app_mock,
    ) -> None:
        parser = build_parser_mock.return_value
        parser.parse_args.return_value = SimpleNamespace(camera_index=3, cleanup_output=False, older_than_days=None, model="skin.pt")
        launch_chat_app_mock.return_value = 5

        result = run_chat.main()

        self.assertEqual(result, 5)
        launch_chat_app_mock.assert_called_once_with(
            window_title="OncoVision Chat AI",
            camera_index=3,
            app_mode="medium",
            selected_model="skin.pt",
        )

    @patch("run_chat.cleanup_medical_outputs")
    @patch("run_chat.cleanup_chat_outputs")
    @patch("run_chat.build_chat_arg_parser")
    def test_run_chat_cleanup_output_reports_summary(
        self,
        build_parser_mock,
        cleanup_mock,
        cleanup_medical_mock,
    ) -> None:
        parser = build_parser_mock.return_value
        parser.parse_args.return_value = SimpleNamespace(
            camera_index=0,
            cleanup_output=True,
            older_than_days=14,
            model=None,
        )
        cleanup_mock.return_value = SimpleNamespace(
            removed_files=4,
            removed_dirs=1,
            freed_bytes=2048,
        )
        cleanup_medical_mock.return_value = SimpleNamespace(
            removed_files=2,
            removed_dirs=0,
            freed_bytes=1024,
        )

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = run_chat.main()

        self.assertEqual(result, 0)
        cleanup_mock.assert_called_once_with(older_than_days=14)
        cleanup_medical_mock.assert_called_once_with(older_than_days=14)
        self.assertIn("Đã xóa file chat: 4", stdout.getvalue())
        self.assertIn("Đã xóa file medical: 2", stdout.getvalue())

    def test_mode_to_ui_defaults_maps_modes(self) -> None:
        self.assertEqual(mode_to_ui_defaults("auto"), ("auto", "medium"))
        self.assertEqual(mode_to_ui_defaults("high"), ("manual", "high"))


if __name__ == "__main__":
    unittest.main()
