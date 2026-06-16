import argparse

from app.chat_ui import build_chat_arg_parser
from app.camera_runtime.bootstrap import resolve_start_bundle
from core.camera_runner import run_camera_session
from tools.runtime_tool import main as runtime_tool_main
from tools.runtime_tool import prompt_runtime_mode
from utils.console_ui import BootProgress, print_runtime_dashboard


def parse_args() -> argparse.Namespace:
    parser = build_chat_arg_parser("Chạy YOLO thời gian thực với camera desktop.")
    parser.add_argument(
        "--advisor-only",
        action="store_true",
        help="Chi mo bo tu van runtime, khong mo camera.",
    )
    return parser.parse_args()


def resolve_run_app_start_bundle(**kwargs):
    return resolve_start_bundle(
        **kwargs,
        prompt_runtime_mode_fn=prompt_runtime_mode,
    )


def main() -> int:
    args = parse_args()
    if args.advisor_only:
        runtime_tool_main()
        return 0

    start_options = resolve_run_app_start_bundle(
        requested_mode=args.mode,
        requested_model=args.model,
        requested_target="camera",
        preferred_target="camera",
    )

    progress = BootProgress("YOLO Camera Realtime")
    progress.advance_to(16, "Dang nhan cau hinh khoi dong")
    progress.advance_to(42, "Dang kiem tra CPU / GPU / CUDA")
    progress.advance_to(68, "Dang chon model va runtime phu hop")
    progress.advance_to(88, "Dang chuan bi mo camera")
    progress.finish("San sang mo camera")

    print_runtime_dashboard(
        title="YOLO Camera Realtime",
        runtime=start_options.runtime,
        hardware=start_options.hardware,
        camera_index=args.camera_index,
        launch_target="camera",
    )
    run_camera_session(runtime=start_options.runtime, camera_index=args.camera_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
