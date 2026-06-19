import argparse

from app.camera_runtime.bootstrap import resolve_start_bundle
from app.camera_runtime.launching import run_camera_launch_flow
from app.chat_ui import build_chat_arg_parser
from core.camera_runner import run_camera_session
from tools.runtime_tool import prompt_runtime_mode
from utils.console_ui import BootProgress, print_runtime_dashboard


def parse_args() -> argparse.Namespace:
    parser = build_chat_arg_parser("Chay YOLO thoi gian thuc voi camera desktop.")
    return parser.parse_args()


def resolve_run_app_start_bundle(**kwargs):
    return resolve_start_bundle(
        **kwargs,
        prompt_runtime_mode_fn=prompt_runtime_mode,
    )


def main() -> int:
    args = parse_args()
    start_options = resolve_run_app_start_bundle(
        requested_mode=args.mode,
        requested_model=args.model,
        requested_target="camera",
        preferred_target="camera",
    )
    return run_camera_launch_flow(
        dashboard_title="YOLO Camera Realtime",
        runtime=start_options.runtime,
        hardware=start_options.hardware,
        camera_index=args.camera_index,
        launch_target="camera",
        print_runtime_dashboard_fn=print_runtime_dashboard,
        run_camera_session_fn=run_camera_session,
        boot_progress_cls=BootProgress,
    )


if __name__ == "__main__":
    raise SystemExit(main())
