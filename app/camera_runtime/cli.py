from __future__ import annotations

import argparse


def build_camera_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        default=None,
        choices=["auto", "high", "medium", "low"],
        help="Chế độ runtime cho camera realtime.",
    )
    parser.add_argument(
        "--camera-index",
        default=0,
        type=int,
        help="Camera index để mở luồng realtime.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model local hoặc tên model để ưu tiên khi khởi động camera.",
    )
    return parser
