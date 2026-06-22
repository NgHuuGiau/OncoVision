from __future__ import annotations

import argparse
import importlib.util
import io
import sys
from pathlib import Path

from app.chat_ui import build_chat_arg_parser, launch_chat_app
from app.chat_ui.output_management import cleanup_chat_outputs
from medical.system_status import get_medical_system_status
from utils.file_utils import ensure_project_directories


ICONS_DIR = Path("assets/icons")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _count_svg_icons() -> int:
    if not ICONS_DIR.exists():
        return 0
    return sum(1 for item in ICONS_DIR.glob("*.svg") if item.is_file())


def run_chat_preflight(print_fn=print) -> int:
    ensure_project_directories()
    capture_dir = Path("output/chat_captures")
    capture_dir.mkdir(parents=True, exist_ok=True)
    medical_status = get_medical_system_status()
    icons_count = _count_svg_icons()
    required_modules = {
        "PySide6": _module_available("PySide6"),
        "cv2": _module_available("cv2"),
        "numpy": _module_available("numpy"),
        "ultralytics": _module_available("ultralytics"),
        "torch": _module_available("torch"),
    }

    print_fn("YOLO chat preflight")
    print_fn(f"- Capture dir: {capture_dir}")
    print_fn(f"- Icons: {icons_count} svg file")
    print_fn("- Required deps: " + ", ".join(f"{name}={ok}" for name, ok in required_modules.items()))
    print_fn(f"- Medical model: ready={medical_status.model_ready}, detail={medical_status.model_message}")

    if not all(required_modules.values()):
        print_fn("- Status: thiếu dependency bắt buộc, chat chưa sẵn sàng để mở GUI.")
        return 1

    if icons_count == 0:
        print_fn("- Status: thiếu icon UI, chat chưa sẵn sàng để mở GUI.")
        return 1

    if not medical_status.model_ready:
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
