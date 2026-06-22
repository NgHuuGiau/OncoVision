from __future__ import annotations

import argparse
import io
import sys

from app.chat_ui import build_chat_arg_parser, launch_chat_app
from app.chat_ui.output_management import cleanup_chat_outputs
from utils.entrypoint_checks import chat_preflight_status


def run_chat_preflight(print_fn=print) -> int:
    required_modules, icons_count, medical_ready, medical_detail, capture_dir = chat_preflight_status()

    print_fn("YOLO chat preflight")
    print_fn(f"- Capture dir: {capture_dir}")
    print_fn(f"- Icons: {icons_count} svg file")
    print_fn("- Required deps: " + ", ".join(f"{name}={ok}" for name, ok in required_modules.items()))
    print_fn(f"- Medical model: detail={medical_detail}")

    if not all(required_modules.values()):
        print_fn("- Status: thiếu dependency bắt buộc, chat chưa sẵn sàng để mở GUI.")
        return 1

    if icons_count == 0:
        print_fn("- Status: thiếu icon UI, chat chưa sẵn sàng để mở GUI.")
        return 1

    if not medical_ready:
        print_fn("- Status: GUI có thể mở nhưng phân tích ảnh y khoa chưa sẵn sàng.")
        return 2

    print_fn("- Status: chat UI và luồng medical đã sẵn sàng.")
    return 0


def main() -> int:
    parser = build_chat_arg_parser("YOLO Chat AI")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Chi kiem tra nhanh dependency, icon, output va medical model; khong mo GUI.",
    )
    parser.add_argument(
        "--cleanup-output",
        action="store_true",
        help="Xoa file output chat da tao tu camera capture.",
    )
    parser.add_argument(
        "--older-than-days",
        type=int,
        default=None,
        help="Chi xoa file cu hon so ngay nay khi dung voi --cleanup-output.",
    )
    args = parser.parse_args()
    if getattr(args, "check_only", False):
        return run_chat_preflight()
    if args.cleanup_output:
        summary = cleanup_chat_outputs(older_than_days=args.older_than_days)
        output = io.StringIO()
        output.write(f"Da xoa file chat: {summary.removed_files}\n")
        output.write(f"Da xoa thu muc rong: {summary.removed_dirs}\n")
        output.write(f"Dung luong giai phong: {summary.freed_bytes} bytes\n")
        sys.stdout.write(output.getvalue())
        return 0
    return launch_chat_app(
        window_title="YOLO Chat AI",
        camera_index=args.camera_index,
        app_mode="medium",
        selected_model=args.model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
