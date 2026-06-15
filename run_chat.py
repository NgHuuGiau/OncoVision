from __future__ import annotations

import sys

from app.chat_ai_app import build_chat_arg_parser, launch_chat_ai_app


def main() -> int:
    parser = build_chat_arg_parser("YOLO Chat AI")
    args = parser.parse_args(None)
    return launch_chat_ai_app(
        window_title="YOLO Chat AI",
        camera_index=args.camera_index,
        app_mode="medium",
    )


if __name__ == "__main__":
    raise SystemExit(main())
