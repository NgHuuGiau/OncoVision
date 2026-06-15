from __future__ import annotations

import argparse


def build_chat_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        default=None,
        choices=["auto", "high", "medium", "low"],
        help="Kept for compatibility. Chat interface does not use this parameter.",
    )
    parser.add_argument(
        "--camera-index",
        default=0,
        type=int,
        help="Default camera index when opening photo capture dialog.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model selected for display before opening the interface.",
    )
    return parser
