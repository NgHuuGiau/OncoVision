from __future__ import annotations

import argparse

from app.camera_runtime.launching import CAMERA_BOOT_FINISH_MESSAGE, CAMERA_BOOT_STEPS, run_camera_launch_flow
from app.chat_ui.cli import build_chat_arg_parser
from utils.console_ui import BootProgress


def build_targeted_parser(description: str) -> argparse.ArgumentParser:
    parser = build_chat_arg_parser(description)
    parser.add_argument(
        "--target",
        default=None,
        choices=["ui", "camera"],
        help="Kieu khoi dong: giao dien desktop hoac camera thoi gian thuc.",
    )
    return parser


def run_targeted_entrypoint(
    *,
    args,
    preferred_target: str,
    ui_title: str,
    dashboard_title: str,
    resolve_start_bundle_fn,
    launch_chat_app_fn,
    print_runtime_dashboard_fn,
    run_camera_session_fn,
) -> int:
    start_options = resolve_start_bundle_fn(
        requested_mode=args.mode,
        requested_model=args.model,
        requested_target=args.target,
        preferred_target=preferred_target,
    )
    if start_options.launch_target == "ui":
        return launch_chat_app_fn(
            window_title=ui_title,
            camera_index=args.camera_index,
            app_mode=start_options.selected_mode,
            selected_model=start_options.selected_model,
        )

    return run_camera_launch_flow(
        dashboard_title=dashboard_title,
        runtime=start_options.runtime,
        hardware=start_options.hardware,
        camera_index=args.camera_index,
        launch_target=start_options.launch_target,
        print_runtime_dashboard_fn=print_runtime_dashboard_fn,
        run_camera_session_fn=run_camera_session_fn,
        boot_progress_cls=BootProgress,
        boot_steps=CAMERA_BOOT_STEPS,
        finish_message=CAMERA_BOOT_FINISH_MESSAGE,
    )


def run_camera_entrypoint(
    *,
    args,
    ui_title: str,
    dashboard_title: str,
    resolve_start_bundle_fn,
    launch_chat_app_fn,
    print_runtime_dashboard_fn,
    run_camera_session_fn,
) -> int:
    return run_targeted_entrypoint(
        args=args,
        preferred_target="camera",
        ui_title=ui_title,
        dashboard_title=dashboard_title,
        resolve_start_bundle_fn=resolve_start_bundle_fn,
        launch_chat_app_fn=launch_chat_app_fn,
        print_runtime_dashboard_fn=print_runtime_dashboard_fn,
        run_camera_session_fn=run_camera_session_fn,
    )
