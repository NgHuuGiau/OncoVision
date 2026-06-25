"""Main interactive menu for the OncoVision application."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from utils.terminal_encoding import ensure_utf8_console

PROJECT_ROOT = Path(__file__).resolve().parent

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"


@dataclass(frozen=True)
class MenuOption:
    script: str
    title: str
    description: str
    group: str
    color: str
    args: tuple[str, ...] = ()


MENU_OPTIONS: dict[str, MenuOption] = {
    "1": MenuOption("run_app.py", "Camera thời gian thực", "Mở camera thời gian thực, chạy model và hiển thị FPS cùng khung phát hiện.", "CHẠY CHÍNH", GREEN),
    "2": MenuOption("run_chat.py", "Chat y dược", "Mở giao diện chat để tải ảnh y khoa lên phân tích và xem kết quả.", "CHẠY CHÍNH", GREEN),
    "3": MenuOption("run_tests.py", "Kiểm thử", "Chạy toàn bộ unit test và các kiểm tra hồi quy chính.", "KIỂM TRA", YELLOW),
    "4": MenuOption("run_doctor.py", "Doctor", "Rà soát phần cứng, model, camera, dữ liệu và gợi ý runtime.", "KIỂM TRA", YELLOW),
    "5": MenuOption("run_train.py", "Huấn luyện chung", "Chuẩn bị dữ liệu, huấn luyện, đánh giá và xuất model tổng quát.", "HUẤN LUYỆN", CYAN),
    "6": MenuOption("run_medical.py", "Medical menu", "Mở menu y dược: dataset, train, validate, phân tích ảnh và lịch sử ca.", "Y DƯỢC", CYAN),
    "7": MenuOption("run_chat.py", "Dọn output", "Xóa nhanh file output chat và medical cũ (camera capture, report, ảnh xử lý).", "BẢO TRÌ", YELLOW, ("--cleanup-output",)),
    "8": MenuOption("run_smoke.py", "Smoke check", "Chạy chuỗi kiểm tra an toàn cho các entrypoint chính.", "KIỂM TRA", YELLOW),
    "0": MenuOption("", "Thoát", "Đóng menu terminal.", "HỆ THỐNG", RED),
}

PRIMARY_KEYS = tuple(key for key in MENU_OPTIONS if key != "0")
TESTED_EXIT_TEXT = "Đã thoát menu."
TESTED_INVALID_TEXT = "Lựa chọn không hợp lệ. Hãy nhập lại."
TESTED_BACK_TEXT = "Quay lại menu chính."
MENU_PROMPT = f"Chọn tác vụ [{ '/'.join(('0', *PRIMARY_KEYS)) }]: "
MEDICAL_BACK_TEXT = "Quay lại menu chính."

MEDICAL_OPTIONS: dict[str, MenuOption] = {
    "1": MenuOption("run_medical.py", "Khởi tạo dataset", "Tạo cấu trúc dataset y dược mặc định.", "DATASET", GREEN, ("init-dataset",)),
    "2": MenuOption("run_medical.py", "Kiểm tra dataset", "Rà soát raw images/raw labels cho pipeline medical.", "DATASET", GREEN, ("audit-dataset",)),
    "3": MenuOption("run_medical.py", "Chia train val test", "Chia dataset medical từ raw sang processed.", "DATASET", GREEN, ("split-dataset",)),
    "4": MenuOption("run_medical.py", "Huấn luyện toàn bộ", "Chạy split, train, validate cho model y dược.", "HUẤN LUYỆN", CYAN, ("train-all",)),
    "5": MenuOption("run_medical.py", "Xem lịch sử ca", "Hiển thị các ca bệnh đã phân tích gần đây.", "KẾT QUẢ", YELLOW, ("history",)),
    "6": MenuOption("run_medical.py", "Phân tích ảnh", "Phân tích một ảnh y khoa với mã bệnh nhân nhập tay.", "KẾT QUẢ", YELLOW),
    "7": MenuOption("run_medical.py", "Medical status", "Kiểm tra nhanh model, dataset, case database và output medical.", "BẢO TRÌ", YELLOW, ("status",)),
    "0": MenuOption("", "Quay lại", "Trở về menu chính.", "HỆ THỐNG", RED),
}
MEDICAL_PRIMARY_KEYS = tuple(key for key in MEDICAL_OPTIONS if key != "0")
MEDICAL_PROMPT = f"Chọn tác vụ medical [{ '/'.join(('0', *MEDICAL_PRIMARY_KEYS)) }]: "


def _configure_terminal_encoding() -> None:
    ensure_utf8_console()


def _command_text(option: MenuOption) -> str:
    if not option.script:
        return "exit"
    return f"python {option.script}{(' ' + ' '.join(option.args)) if option.args else ''}"


def _get_terminal_width() -> int:
    try:
        return max(60, min(120, os.get_terminal_size().columns))
    except OSError:
        return 90


def _wrap_text(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [""]


def _render_options(
    options: dict[str, MenuOption],
    primary_keys: tuple[str, ...],
    title: str,
    print_fn=print,
) -> None:
    width = _get_terminal_width()
    print_fn("")
    print_fn(f"{BOLD}{'=' * width}{RESET}")
    print_fn(f"{BOLD}  {title}{RESET}")
    print_fn(f"{BOLD}{'=' * width}{RESET}")
    print_fn("")

    current_group = ""
    for key in (*primary_keys, "0"):
        option = options[key]
        if option.group != current_group:
            current_group = option.group
            print_fn(f"{BOLD}{option.color}  [{current_group}]{RESET}")
            print_fn(f"{option.color}  {'-' * min(len(current_group) + 2, width - 4)}{RESET}")

        label = f"{key}. {option.title}"
        cmd = _command_text(option)
        desc = f"{option.description}  {DIM}[{cmd}]{RESET}"

        indent = "  " + " " * len(label) + " "
        wrap_width = max(20, width - len(indent) - 2)
        wrapped_desc = _wrap_text(desc, wrap_width)

        for i, line in enumerate(wrapped_desc):
            if i == 0:
                print_fn(f"{option.color}  {label} {line}{RESET}")
            else:
                print_fn(f"{option.color}{indent}{line}{RESET}")

    print_fn("")
    print_fn(f"{DIM}{'-' * width}{RESET}")
    print_fn("")


def _render_menu(print_fn=print) -> None:
    _render_options(MENU_OPTIONS, PRIMARY_KEYS, "OncoVision : MENU ĐIỀU KHIỂN", print_fn=print_fn)


def _render_medical_menu(print_fn=print) -> None:
    _render_options(MEDICAL_OPTIONS, MEDICAL_PRIMARY_KEYS, "OncoVision : MEDICAL MENU", print_fn=print_fn)


def _clear_terminal() -> None:
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def _run_script(script_name: str, *script_args: str) -> int:
    return subprocess.call([sys.executable, script_name, *script_args], cwd=PROJECT_ROOT)


def _resolve_medical_args(option: MenuOption, input_fn=input, print_fn=print) -> tuple[str, ...] | None:
    if option.script != "run_medical.py" or option.args:
        return option.args
    image_path = input_fn("Nhập đường dẫn ảnh y khoa: ").strip()
    if not image_path:
        print_fn(f"{RED}Chưa nhập đường dẫn ảnh. Quay lại menu medical.{RESET}")
        return None
    patient_code = input_fn("Nhập mã bệnh nhân: ").strip() or "BN001"
    return ("analyze", "--image", image_path, "--patient-code", patient_code)


def _run_selected_option(
    option: MenuOption,
    *,
    args: tuple[str, ...],
    print_fn=print,
    run_script_fn=_run_script,
    clear_terminal_fn=_clear_terminal,
    back_text: str = TESTED_BACK_TEXT,
) -> None:
    clear_terminal_fn()
    width = _get_terminal_width()
    print_fn(f"{BOLD}{option.color}{'-' * width}{RESET}")
    print_fn(f"{BOLD}{option.color}  ĐANG CHẠY: {option.title}{RESET}")
    command_text = f"python {option.script} {' '.join(args)}".strip()
    print_fn(f"{option.color}  Lệnh   : {command_text}{RESET}")
    print_fn(f"{option.color}  Ghi chú: {option.description}{RESET}")
    print_fn(f"{BOLD}{option.color}{'-' * width}{RESET}")
    try:
        exit_code = run_script_fn(option.script, *args)
    except OSError as exc:
        print_fn(f"{RED}Không thể chạy {option.script}: {exc}{RESET}")
        return
    status = "Đã chạy xong" if exit_code == 0 else "Kết thúc với lỗi"
    color = GREEN if exit_code == 0 else YELLOW
    print_fn(f"{color}{status} {option.script}. {back_text} (exit={exit_code}){RESET}")


def _run_medical_menu(input_fn=input, print_fn=print, run_script_fn=_run_script, clear_terminal_fn=_clear_terminal) -> None:
    while True:
        _render_medical_menu(print_fn=print_fn)
        choice = input_fn(MEDICAL_PROMPT).strip()
        if choice == "0":
            print_fn(f"{YELLOW}{MEDICAL_BACK_TEXT}{RESET}")
            return
        if not choice:
            continue
        option = MEDICAL_OPTIONS.get(choice)
        if option is None:
            print_fn(f"{RED}{TESTED_INVALID_TEXT}{RESET}")
            continue
        resolved_args = _resolve_medical_args(option, input_fn=input_fn, print_fn=print_fn)
        if resolved_args is None:
            continue
        _run_selected_option(
            option,
            args=resolved_args,
            print_fn=print_fn,
            run_script_fn=run_script_fn,
            clear_terminal_fn=clear_terminal_fn,
            back_text="Quay lại menu medical.",
        )


def _run_menu_choice(
    choice: str,
    *,
    input_fn=input,
    print_fn=print,
    run_script_fn=_run_script,
    clear_terminal_fn=_clear_terminal,
) -> bool:
    option = MENU_OPTIONS.get(choice)
    if option is None:
        print_fn(f"{RED}{TESTED_INVALID_TEXT}{RESET}")
        return False
    if choice == "6":
        clear_terminal_fn()
        _run_medical_menu(
            input_fn=input_fn,
            print_fn=print_fn,
            run_script_fn=run_script_fn,
            clear_terminal_fn=clear_terminal_fn,
        )
        return True
    _run_selected_option(
        option,
        args=option.args,
        print_fn=print_fn,
        run_script_fn=run_script_fn,
        clear_terminal_fn=clear_terminal_fn,
    )
    return True


def main(input_fn=input, print_fn=print, run_script_fn=_run_script, clear_terminal_fn=_clear_terminal) -> int:
    _configure_terminal_encoding()
    try:
        while True:
            _render_menu(print_fn=print_fn)
            choice = input_fn(MENU_PROMPT).strip()
            if choice == "0":
                print_fn(f"{YELLOW}{TESTED_EXIT_TEXT}{RESET}")
                return 0
            _run_menu_choice(
                choice,
                input_fn=input_fn,
                print_fn=print_fn,
                run_script_fn=run_script_fn,
                clear_terminal_fn=clear_terminal_fn,
            )
    except KeyboardInterrupt:
        print_fn(f"{YELLOW}\nĐã thoát menu.{RESET}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
