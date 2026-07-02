"""Menu chính của OncoVision."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from core.model_catalog import YOLO11_MODELS_ASC
from training.download_models import download_models
from utils.entrypoint_common import run_entrypoint
from utils.file_utils import ensure_project_directories
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
    "1": MenuOption("run_app.py", "Camera", "Mở camera thời gian thực, chạy model và hiển thị kết quả.", "CHẠY", GREEN),
    "2": MenuOption("run_chat.py", "Chat y dược", "Mở giao diện chat và phân tích ảnh y khoa.", "CHẠY", GREEN),
    "3": MenuOption("run_medical.py", "Y dược", "Dataset, train, phân tích ảnh và lịch sử ca.", "Y DƯỢC", CYAN),
    "4": MenuOption("run_train.py", "Huấn luyện", "Chuẩn bị dữ liệu, train, đánh giá và xuất model.", "HUẤN LUYỆN", CYAN),
    "5": MenuOption("", "Kiểm tra", "Mở submenu Doctor, Test và Smoke.", "KIỂM TRA", YELLOW),
    "6": MenuOption("run_chat.py", "Dọn output", "Xóa nhanh output chat và medical cũ.", "BẢO TRÌ", YELLOW, ("--cleanup-output",)),
    "0": MenuOption("", "Thoát", "Đóng menu terminal.", "HỆ THỐNG", RED),
}

PRIMARY_KEYS = tuple(key for key in MENU_OPTIONS if key != "0")
TESTED_EXIT_TEXT = "Đã thoát menu."
TESTED_INVALID_TEXT = "Lựa chọn không hợp lệ. Hãy nhập lại."
MENU_PROMPT = f"Chọn tác vụ [{'/'.join(('0', *PRIMARY_KEYS))}]: "

MEDICAL_OPTIONS: dict[str, MenuOption] = {
    "1": MenuOption("run_medical.py", "Báo cáo nhanh", "Tóm tắt dataset, model và mức sẵn sàng train.", "KIỂM TRA", GREEN, ("report",)),
    "2": MenuOption("run_medical.py", "Dataset y dược", "Kiểm tra dataset raw, split và sẵn sàng train.", "DATASET", GREEN, ("ready",)),
    "3": MenuOption("run_medical.py", "Khởi tạo dataset", "Tạo cấu trúc dataset/medical/skin_lesion.", "DATASET", GREEN, ("init-dataset",)),
    "4": MenuOption("run_medical.py", "Chia dữ liệu", "Chia raw sang train/val/test.", "DATASET", GREEN, ("split-dataset",)),
    "5": MenuOption("run_medical.py", "Huấn luyện", "Chạy split, train và validate model y dược.", "HUẤN LUYỆN", CYAN, ("train-all",)),
    "6": MenuOption("run_medical.py", "Phân tích ảnh", "Phân tích một ảnh y khoa với mã bệnh nhân nhập tay.", "KẾT QUẢ", YELLOW),
    "7": MenuOption("run_medical.py", "Lịch sử ca", "Hiển thị các ca bệnh đã phân tích gần đây.", "KẾT QUẢ", YELLOW, ("history",)),
    "0": MenuOption("", "Quay lại", "Trở về menu chính.", "HỆ THỐNG", RED),
}
MEDICAL_PRIMARY_KEYS = tuple(key for key in MEDICAL_OPTIONS if key != "0")
MEDICAL_PROMPT = f"Chọn tác vụ y dược [{'/'.join(('0', *MEDICAL_PRIMARY_KEYS))}]: "
MEDICAL_BACK_TEXT = "Quay lại menu chính."

CHECK_OPTIONS: dict[str, MenuOption] = {
    "1": MenuOption("run_doctor.py", "Doctor", "Rà soát môi trường, model và dataset mà không cần webcam thật.", "KIỂM TRA", YELLOW, ("--skip-camera-check",)),
    "2": MenuOption("run_tests.py", "Test", "Chạy dashboard unit test và regression mà không yêu cầu camera thật.", "KIỂM TRA", YELLOW, ("--skip-camera-check",)),
    "3": MenuOption("run_smoke.py", "Smoke", "Chạy smoke check qua các entrypoint chính để kiểm tra nhanh toàn luồng.", "KIỂM TRA", YELLOW),
    "4": MenuOption("run_smoke.py", "Smoke + tests", "Chạy smoke check và nối thêm run_tests để rà soát sau cùng.", "KIỂM TRA", YELLOW, ("--include-tests",)),
    "0": MenuOption("", "Quay lại", "Trở về menu chính.", "HỆ THỐNG", RED),
}
CHECK_PRIMARY_KEYS = tuple(key for key in CHECK_OPTIONS if key != "0")
CHECK_PROMPT = f"Chọn tác vụ kiểm tra [{'/'.join(('0', *CHECK_PRIMARY_KEYS))}]: "


def _progress_bar(percent: int, width: int = 20) -> str:
    percent = max(0, min(100, percent))
    filled = round(width * percent / 100)
    return f"[{'#' * filled}{'.' * (width - filled)}] {percent:3d}%"


def _model_exists(model_name: str) -> bool:
    return os.path.exists(str(Path("models/pretrained") / model_name))


def _ensure_yolo11_models() -> None:
    missing_models = [model_name for model_name in YOLO11_MODELS_ASC if not _model_exists(model_name)]
    if not missing_models:
        return
    total = len(missing_models)
    print(f"{BOLD}{CYAN}{'=' * 78}{RESET}")
    print(f"{BOLD}{CYAN}  ĐANG KIỂM TRA VÀ TẢI MODEL YOLO11{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 78}{RESET}")
    for index, model_name in enumerate(missing_models, start=1):
        print(f"{YELLOW}  [{index}/{total}] {model_name} {_progress_bar(0)}{RESET}")
    downloaded, skipped = download_models(missing_models)
    for index, model_name in enumerate(downloaded, start=1):
        print(f"{GREEN}  ✓ [{index}/{total}] {model_name} {_progress_bar(100)}{RESET}")
    for model_name in skipped:
        print(f"{CYAN}  • Đã có sẵn {model_name}, bỏ qua{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 78}{RESET}")


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


def _render_options(options: dict[str, MenuOption], primary_keys: tuple[str, ...], title: str, print_fn=print) -> None:
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


def _render_check_menu(print_fn=print) -> None:
    _render_options(CHECK_OPTIONS, CHECK_PRIMARY_KEYS, "OncoVision : MENU KIỂM TRA", print_fn=print_fn)


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
        print_fn(f"{RED}Chưa nhập đường dẫn ảnh. Quay lại menu y dược.{RESET}")
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
    back_text: str = TESTED_EXIT_TEXT,
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
            back_text="Quay lại menu y dược.",
        )


def _run_check_menu(input_fn=input, print_fn=print, run_script_fn=_run_script, clear_terminal_fn=_clear_terminal) -> None:
    while True:
        _render_check_menu(print_fn=print_fn)
        choice = input_fn(CHECK_PROMPT).strip()
        if choice == "0":
            print_fn(f"{YELLOW}Quay lại menu chính.{RESET}")
            return
        if not choice:
            continue
        option = CHECK_OPTIONS.get(choice)
        if option is None:
            print_fn(f"{RED}{TESTED_INVALID_TEXT}{RESET}")
            continue
        _run_selected_option(
            option,
            args=option.args,
            print_fn=print_fn,
            run_script_fn=run_script_fn,
            clear_terminal_fn=clear_terminal_fn,
            back_text="Quay lại menu kiểm tra.",
        )


def _run_menu_choice(choice: str, *, input_fn=input, print_fn=print, run_script_fn=_run_script, clear_terminal_fn=_clear_terminal) -> bool:
    option = MENU_OPTIONS.get(choice)
    if option is None:
        print_fn(f"{RED}{TESTED_INVALID_TEXT}{RESET}")
        return False
    if choice == "3":
        clear_terminal_fn()
        _run_medical_menu(input_fn=input_fn, print_fn=print_fn, run_script_fn=run_script_fn, clear_terminal_fn=clear_terminal_fn)
        return True
    if choice == "5":
        clear_terminal_fn()
        _run_check_menu(input_fn=input_fn, print_fn=print_fn, run_script_fn=run_script_fn, clear_terminal_fn=clear_terminal_fn)
        return True
    _run_selected_option(option, args=option.args, print_fn=print_fn, run_script_fn=run_script_fn, clear_terminal_fn=clear_terminal_fn)
    return True


def main(input_fn=input, print_fn=print, run_script_fn=_run_script, clear_terminal_fn=_clear_terminal) -> int:
    _configure_terminal_encoding()
    ensure_project_directories()
    _ensure_yolo11_models()
    try:
        while True:
            _render_menu(print_fn=print_fn)
            choice = input_fn(MENU_PROMPT).strip()
            if choice == "0":
                print_fn(f"{YELLOW}{TESTED_EXIT_TEXT}{RESET}")
                return 0
            _run_menu_choice(choice, input_fn=input_fn, print_fn=print_fn, run_script_fn=run_script_fn, clear_terminal_fn=clear_terminal_fn)
    except KeyboardInterrupt:
        print_fn(f"{YELLOW}\nĐã thoát menu.{RESET}")
        return 0


if __name__ == "__main__":
    raise SystemExit(run_entrypoint(main))
