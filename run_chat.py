from __future__ import annotations

import io
import sys

from app.chat_ui import build_chat_arg_parser, launch_chat_app
from app.chat_ui.output_management import cleanup_chat_outputs


def main() -> int:
    parser = build_chat_arg_parser("YOLO Chat AI")
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
