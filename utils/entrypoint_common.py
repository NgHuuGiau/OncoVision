from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from utils.terminal_encoding import ensure_utf8_console


T = TypeVar("T")


def configure_entrypoint() -> None:
    ensure_utf8_console()


def run_entrypoint(main_fn: Callable[[], T]) -> T:
    configure_entrypoint()
    return main_fn()

