from __future__ import annotations

import io
import sys

from app.chat_ui import build_chat_arg_parser, launch_chat_app
from app.chat_ui.output_management import cleanup_chat_outputs
from medical.output_management import cleanup_medical_outputs
from utils.entrypoint_common import run_entrypoint
from utils.entrypoint_checks import chat_preflight_status
from utils.icons import create_default_icons, ICONS_DIR


def run_chat_preflight(print_fn=print, auto_fix_icons: bool = False) -> int:
    required_modules, icons_count, medical_ready, medical_detail, capture_dir = chat_preflight_status()

    print_fn("OncoVision kiểm tra trước khi mở chat")
    print_fn(f"- Capture dir: {capture_dir}")
    print_fn(f"- Icons: {icons_count} file svg")
    print_fn("- Thư viện bắt buộc: " + ", ".join(f"{name}={ok}" for name, ok in required_modules.items()))
    print_fn(f"- Model medical: chi tiết={medical_detail}")

    if not all(required_modules.values()):
        print_fn("- Trạng thái: thiếu thư viện bắt buộc, chat chưa sẵn sàng để mở GUI.")
        return 1

    if icons_count < 10:
        if auto_fix_icons:
            print_fn("- Trạng thái: thiếu icon UI, đang tự động tạo icon...")
            ICONS_DIR.mkdir(parents=True, exist_ok=True)
            create_default_icons()
            icons_count = sum(1 for _ in ICONS_DIR.iterdir() if _.is_file() and _.suffix == ".svg")
            print_fn(f"- Trạng thái: đã tạo {icons_count} icon.")
        else:
            print_fn("- Trạng thái: thiếu icon UI, chat chưa sẵn sàng để mở GUI.")
            return 1

    if not medical_ready:
        print_fn("- Trạng thái: GUI có thể mở nhưng phân tích ảnh y khoa chưa sẵn sàng.")
        return 2

    print_fn("- Trạng thái: chat UI và luồng medical đã sẵn sàng.")
    return 0


def main() -> int:
    parser = build_chat_arg_parser("OncoVision Chat AI")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Chỉ kiểm tra nhanh thư viện, icon, output và model medical; không mở GUI.",
    )
    parser.add_argument(
        "--auto-fix-icons",
        action="store_true",
        help="Tự động tạo icon nếu thiếu khi kiểm tra.",
    )
    parser.add_argument(
        "--cleanup-output",
        action="store_true",
        help="Xóa file output chat và medical đã tạo.",
    )
    parser.add_argument(
        "--older-than-days",
        type=int,
        default=None,
        help="Chỉ xóa file cũ hơn số ngày này khi dùng với --cleanup-output.",
    )
    args = parser.parse_args()
    if getattr(args, "check_only", False):
        return run_chat_preflight(auto_fix_icons=getattr(args, "auto_fix_icons", False))
    if args.cleanup_output:
        output = io.StringIO()
        output.write("=== Chat output ===\n")
        chat_summary = cleanup_chat_outputs(older_than_days=args.older_than_days)
        output.write(f"Đã xóa file chat: {chat_summary.removed_files}\n")
        output.write(f"Đã xóa thư mục rỗng: {chat_summary.removed_dirs}\n")
        output.write(f"Dung lượng giải phóng: {chat_summary.freed_bytes} bytes\n")
        output.write("\n=== Medical output ===\n")
        medical_summary = cleanup_medical_outputs(older_than_days=args.older_than_days)
        output.write(f"Đã xóa file medical: {medical_summary.removed_files}\n")
        output.write(f"Đã xóa thư mục rỗng: {medical_summary.removed_dirs}\n")
        output.write(f"Dung lượng giải phóng: {medical_summary.freed_bytes} bytes\n")
        sys.stdout.write(output.getvalue())
        return 0
    return launch_chat_app(
        window_title="OncoVision Chat AI",
        camera_index=args.camera_index,
        app_mode="medium",
        selected_model=args.model,
    )


if __name__ == "__main__":
    raise SystemExit(run_entrypoint(main))
