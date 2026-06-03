from __future__ import annotations

import sys

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
CARD_WIDTH = 88

STATUS_OK = GREEN
STATUS_WARN = YELLOW
STATUS_ERROR = RED


def _ensure_utf8_console() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        return


_ensure_utf8_console()


def line(text: str = "", color: str = "") -> str:
    return f"{color}{text}{RESET}" if color else text


def pad(text: str, width: int = CARD_WIDTH) -> str:
    return text[:width].ljust(width)


def rule(char: str = "=") -> str:
    return char * CARD_WIDTH


def section(title: str, color: str = CYAN) -> str:
    return line(pad(f"[ {title} ]"), BOLD + color)


def row(label: str, value: str = "", color: str = "", *, bounded: bool = True) -> str:
    content = f"{label:<18} {value}".rstrip()
    return line(pad(content) if bounded else content, color)


def status_color(ok: bool | None) -> str:
    if ok is True:
        return STATUS_OK
    if ok is False:
        return STATUS_ERROR
    return STATUS_WARN


def header(title: str, *, color: str = CYAN) -> list[str]:
    return [line(rule("="), color), line(pad(title), BOLD + color), line(rule("="), color)]


def command_row(index: int, command: str) -> str:
    return row(f"Lệnh {index}", command, CYAN, bounded=False)
