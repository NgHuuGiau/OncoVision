from __future__ import annotations

import argparse


def build_camera_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        default=None,
        choices=["auto", "high", "medium", "low"],
        help="Che do runtime cho camera realtime.",
    )
    parser.add_argument(
        "--camera-index",
        default=0,
        type=int,
        help="Camera index de mo luong realtime.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model local hoac ten model de uu tien khi khoi dong camera.",
    )
    return parser
