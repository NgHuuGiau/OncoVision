from __future__ import annotations

import os
import textwrap

from utils.terminal_encoding import ensure_utf8_console

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[38;5;81m"
DIM = "\033[2m"
CARD_WIDTH = 88
MIN_CARD_WIDTH = 40
LABEL_WIDTH = 16


ensure_utf8_console()


def line(text: str = "", color: str = "") -> str:
    return f"{color}{text}{RESET}" if color else text


def pad(text: str, width: int | None = None) -> str:
    active_width = _card_width(width)
    return _truncate(text, active_width).ljust(active_width)


def rule(char: str = "=") -> str:
    glyph = {
        "=": "\u2550",
        "-": "\u2500",
        ".": "\u00b7",
    }.get(char, char)
    return glyph * _card_width()


def _card_width(width: int | None = None) -> int:
    if width is not None:
        return max(0, width)
    try:
        columns = os.get_terminal_size().columns
    except OSError:
        return CARD_WIDTH
    return max(MIN_CARD_WIDTH, min(CARD_WIDTH, columns - 2))


def _truncate(text: str, width: int) -> str:
    if len(text) <= width or width <= 3:
        return text[:width]
    return f"{text[: width - 3].rstrip()}..."


def _wrap_row(label: str, value: str, width: int) -> list[str]:
    label_text = _truncate(label, LABEL_WIDTH).ljust(LABEL_WIDTH)
    prefix = f"\u2502 {label_text} "
    continuation_prefix = f"\u2502 {' ' * LABEL_WIDTH} "
    available_width = max(12, width - len(prefix))
    wrapped_value = textwrap.wrap(
        value,
        width=available_width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]
    lines = [f"{prefix}{wrapped_value[0]}".rstrip()]
    lines.extend(f"{continuation_prefix}{chunk}".rstrip() for chunk in wrapped_value[1:])
    return lines


def section(title: str, color: str = CYAN) -> str:
    return line(pad(f"\u25c6 {title}"), BOLD + color)


def row(label: str, value: str = "", color: str = "", *, bounded: bool = True) -> str:
    active_width = _card_width()
    content_lines = _wrap_row(label, value, active_width)
    if bounded:
        rendered = "\n".join(pad(item, active_width) for item in content_lines)
    else:
        rendered = "\n".join(content_lines)
    return line(rendered, color)


def header(title: str, *, color: str = CYAN) -> list[str]:
    active_width = _card_width()
    border = "\u2550" * max(0, active_width - 2)
    return [
        line(f"\u2554{border}\u2557", color),
        line(f"\u2551 {pad(title, active_width - 4)} \u2551", BOLD + color),
        line(f"\u255a{border}\u255d", color),
    ]


def command_row(index: int, command: str) -> str:
    return row(f"L\u1ec7nh {index}", command, BLUE if index == 1 else CYAN, bounded=False)
