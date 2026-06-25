from __future__ import annotations

import argparse

from app.camera_runtime.cli import build_camera_arg_parser
from app.camera_runtime.bootstrap import resolve_start_bundle
from app.camera_runtime.launching import run_camera_launch_flow
from core.camera_runner import run_camera_session
from core.hardware_info import detect_hardware
from core.runtime_prompt import prompt_runtime_mode
from core.runtime_advisor import build_recommendations
from utils.console_ui import BootProgress, print_runtime_dashboard


def parse_args() -> argparse.Namespace:
    parser = build_camera_arg_parser("Chay YOLO thoi gian thuc voi camera desktop.")
    parser.add_argument(
        "--advisor-only",
        action="store_true",
        help="Chi in khuyen nghi runtime, khong mo camera.",
    )
    return parser.parse_args()


def resolve_run_app_start_bundle(**kwargs):
    return resolve_start_bundle(
        **kwargs,
        prompt_runtime_mode_fn=prompt_runtime_mode,
    )


def run_runtime_advisor(print_fn=print) -> int:
    hardware = detect_hardware()
    recommendations = build_recommendations(hardware)
    print_fn("YOLO runtime advisor")
    for label, runtime in recommendations.items():
        print_fn(
            f"- {label}: model={runtime.primary_model_name}, device={runtime.resolved_device}, "
            f"imgsz={runtime.imgsz}, max_det={runtime.max_det}"
        )
    return 0


def main() -> int:
    args = parse_args()
    if getattr(args, "advisor_only", False):
        return run_runtime_advisor()
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
