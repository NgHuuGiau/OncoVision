from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from training.terminal_ui import CYAN, GREEN, RED, YELLOW, header, line, row, rule, section
from utils.terminal_encoding import ensure_utf8_console


@dataclass(frozen=True)
class MenuOption:
    script: str
    title: str
    description: str
    group: str
    color: str
    args: tuple[str, ...] = ()


MENU_OPTIONS: dict[str, MenuOption] = {
    "1": MenuOption(
        script="run_app.py",
        title="Camera nhan dien",
        description="Mo camera realtime, chay YOLO va hien thi FPS canh box nhan dien.",
        group="CHAY CHINH",
        color=GREEN,
    ),
    "2": MenuOption(
        script="run_chat.py",
        title="Desktop chat",
        description="Mo giao dien chat, dinh kem anh, van ban hoac anh chup camera.",
        group="CHAY CHINH",
        color=GREEN,
    ),
    "3": MenuOption(
        script="run_tests.py",
        title="Kiem thu",
        description="Chay toan bo unit test va kiem tra camera that.",
        group="KIEM TRA",
        color=YELLOW,
    ),
    "4": MenuOption(
        script="run_doctor.py",
        title="Doctor",
        description="Ra soat phan cung, model, camera, du lieu va goi y runtime.",
        group="KIEM TRA",
        color=YELLOW,
    ),
    "5": MenuOption(
        script="run_app.py",
        title="Tu van runtime",
        description="Phan tich runtime phu hop ma khong mo camera.",
        group="KIEM TRA",
        color=YELLOW,
        args=("--advisor-only",),
    ),
    "6": MenuOption(
        script="run_train.py",
        title="Huan luyen",
        description="Chuan bi du lieu, huan luyen, danh gia va xuat model custom.",
        group="HUAN LUYEN",
        color=CYAN,
    ),
    "0": MenuOption(
        script="",
        title="Thoat",
        description="Dong menu terminal.",
        group="HE THONG",
        color=RED,
    ),
}
PRIMARY_KEYS = tuple(key for key in MENU_OPTIONS if key != "0")
TESTED_EXIT_TEXT = "Da thoat menu."
TESTED_INVALID_TEXT = "Lua chon khong hop le. Hay nhap lai."
TESTED_BACK_TEXT = "Quay lai menu."
MENU_PROMPT = f"Chon tac vu ({'/'.join(('0', *PRIMARY_KEYS))}): "


def _configure_terminal_encoding() -> None:
    ensure_utf8_console()


def _command_text(option: MenuOption) -> str:
    if not option.script:
        return "exit"
    return f"python {option.script}{(' ' + ' '.join(option.args)) if option.args else ''}"


def _option_label(key: str, option: MenuOption) -> tuple[str, str]:
    return f"{key} | {option.title}", f"{option.description} [{_command_text(option)}]"


def _render_menu(print_fn=print) -> None:
    for item in header("YOLO HUB :: MENU DIEU KHIEN"):
        print_fn(item)
    current_group = ""
    for key in (*PRIMARY_KEYS, "0"):
        option = MENU_OPTIONS[key]
        if option.group != current_group:
            if current_group:
                print_fn(line(rule("-"), CYAN))
            print_fn(section(option.group, option.color))
            current_group = option.group
        label, description = _option_label(key, option)
        print_fn(row(label, description, option.color, bounded=False))
    print_fn(line(rule("="), CYAN))


def _clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _run_script(script_name: str, *script_args: str) -> int:
    return subprocess.call([sys.executable, script_name, *script_args])


def main(input_fn=input, print_fn=print, run_script_fn=_run_script, clear_terminal_fn=_clear_terminal) -> int:
    _configure_terminal_encoding()
    while True:
        _render_menu(print_fn=print_fn)
        choice = input_fn(MENU_PROMPT).strip()
        if choice == "0":
            print_fn(line(TESTED_EXIT_TEXT, YELLOW))
            return 0
        option = MENU_OPTIONS.get(choice)
        if option is None:
            print_fn(line(TESTED_INVALID_TEXT, RED))
            continue
        clear_terminal_fn()
        print_fn(line(rule("="), CYAN))
        print_fn(section(f"DANG CHAY: {option.title}", option.color))
        print_fn(row("Lenh", _command_text(option), option.color, bounded=False))
        print_fn(row("Ghi chu", option.description, option.color, bounded=False))
        print_fn(line(rule("-"), CYAN))
        exit_code = run_script_fn(option.script, *option.args)
        message = (
            f"Da chay xong {option.script}. {TESTED_BACK_TEXT}"
            if exit_code == 0
            else f"{option.script} ket thuc voi ma {exit_code}. {TESTED_BACK_TEXT}"
        )
        print_fn(line(message, GREEN if exit_code == 0 else YELLOW))


if __name__ == "__main__":
    raise SystemExit(main())
