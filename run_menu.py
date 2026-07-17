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
    "1": MenuOption("run_app.py", "Camera realtime", "Mở camera, chạy model và xem kết quả ngay.", "CHẠY NHANH", GREEN),
    "2": MenuOption("run_chat.py", "Chat y dược", "Mở chat UI và luồng phân tích ảnh y khoa.", "CHẠY NHANH", GREEN),
    "3": MenuOption("run_medical.py", "Y dược", "Quản lý dataset, lịch sử ca và luồng medical.", "Y DƯỢC", CYAN),
    "5": MenuOption("", "Kiểm tra", "Mở menu kiểm tra (Doctor, Test, Smoke).", "KIỂM TRA", YELLOW),
    "6": MenuOption("run_chat.py", "Dọn cache", "Xóa output chat và medical cũ cho repo gọn hơn.", "BẢO TRÌ", YELLOW, ("--cleanup-output",)),
    "0": MenuOption("", "Thoát", "Đóng menu terminal.", "HỆ THỐNG", RED),
}

PRIMARY_KEYS = tuple(key for key in MENU_OPTIONS if key != "0")
TESTED_EXIT_TEXT = "Đã thoát menu."
TESTED_INVALID_TEXT = "Lựa chọn không hợp lệ. Hãy nhập lại."
MENU_PROMPT = f"Chọn tác vụ [{'/'.join(('0', *PRIMARY_KEYS))}]: "

MEDICAL_OPTIONS: dict[str, MenuOption] = {
    "1": MenuOption("run_medical.py", "Báo cáo nhanh", "Tóm tắt dataset, model và mức sẵn sàng train.", "KIỂM TRA", GREEN, ("report",)),
    "2": MenuOption("run_medical.py", "Kiểm tra ảnh", "Kiểm tra ảnh y khoa hợp lệ và phân loại modality/vùng cơ thể.", "KIỂM TRA", GREEN, ("validate-image",)),
    "3": MenuOption("run_medical.py", "Dữ liệu & Huấn luyện", "Khởi tạo, chia split và train/validate model y dược (trong terminal).", "Y DƯỢNG", CYAN, ("train-all",)),
    "4": MenuOption("run_medical.py", "Phân tích ảnh", "Phân tích ảnh y khoa với mã bệnh nhân nhập tay.", "KẾT QUẢ", YELLOW, ("analyze",)),
    "5": MenuOption("run_medical.py", "Lịch sử ca", "Xem các ca bệnh đã phân tích gần đây.", "KẾT QUẢ", YELLOW, ("history",)),
    "6": MenuOption("run_medical.py", "Cải tiến (AL + modality)", "Gợi ý gán nhãn -> train classifier modality -> hiệu chỉnh ngưỡng tuning (gộp 1 luồng).", "CẢI TIẾN", CYAN, ("active-learning", "|", "train-modality", "--epochs", "12", "--verbose", "|", "calibrate-modality-tuning", "--apply")),
    "7": MenuOption("run_medical.py", "Train nhận diện ảnh", "Train classifier nhận diện modality (CT/MRI/X-ray/...) từ dataset y khoa.", "CẢI TIẾN", CYAN, ("train-modality",)),
    "0": MenuOption("", "Quay lại", "Trở về menu chính.", "HỆ THỐNG", RED),
}
MEDICAL_PRIMARY_KEYS = tuple(key for key in MEDICAL_OPTIONS if key != "0")
MEDICAL_PROMPT = f"Chọn tác vụ y dược [{'/'.join(('0', *MEDICAL_PRIMARY_KEYS))}]: "
MEDICAL_BACK_TEXT = "Quay lại menu chính."

CHECK_OPTIONS: dict[str, MenuOption] = {
    "1": MenuOption("run_doctor.py", "Doctor", "Rà soát môi trường, model và dataset mà không cần webcam thật.", "KIỂM TRA", YELLOW, ("--skip-camera-check",)),
    "2": MenuOption("run_tests.py", "Test", "Chạy dashboard unit test và regression mà không cần camera thật.", "KIỂM TRA", YELLOW, ("--skip-camera-check",)),
    "3": MenuOption("run_smoke.py", "Smoke", "Chạy smoke check qua các entrypoint chính để rà nhanh toàn luồng.", "KIỂM TRA", YELLOW),
    "4": MenuOption("run_smoke.py", "Smoke + tests", "Chạy smoke check và nối thêm run_tests để rà soát sau cùng.", "KIỂM TRA", YELLOW, ("--include-tests",)),
    "5": MenuOption("run_medical.py", "Kiểm tra ảnh y khoa", "Nhập đường dẫn ảnh để kiểm tra hợp lệ và phân loại modality/body region.", "KIỂM TRA", YELLOW, ("validate-image",)),
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
        return "menu"
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
    ordered_keys = (*primary_keys, "0")
    label_width = max(len(f"{key}. {options[key].title}") for key in ordered_keys)
    print_fn("")
    print_fn(f"{BOLD}{'=' * width}{RESET}")
    print_fn(f"{BOLD}  {title}{RESET}")
    print_fn(f"{BOLD}{'=' * width}{RESET}")
    print_fn("")
    print_fn(f"{DIM}  Nhập số để chọn, 0 để quay lại hoặc thoát.{RESET}")
    print_fn("")

    current_group = ""
    for key in ordered_keys:
        option = options[key]
        if option.group != current_group:
            current_group = option.group
            print_fn(f"{BOLD}{option.color}  [{current_group}]{RESET}")
            print_fn(f"{option.color}  {'-' * min(len(current_group) + 2, width - 4)}{RESET}")

        label = f"{key}. {option.title}"
        cmd = _command_text(option)
        label_pad = " " * (label_width - len(label))
        desc_indent = " " * (label_width + 6)
        wrap_width = max(24, width - len(desc_indent) - 2)
        wrapped_desc = _wrap_text(option.description, wrap_width)

        print_fn(f"{option.color}  {label}{label_pad}  {wrapped_desc[0]}{RESET}")
        for line in wrapped_desc[1:]:
            print_fn(f"{option.color}{desc_indent}{line}{RESET}")
        print_fn(f"{DIM}{desc_indent}[{cmd}]{RESET}")

    print_fn("")
    print_fn(f"{DIM}{'-' * width}{RESET}")
    print_fn("")


def _render_menu(print_fn=print) -> None:
    _render_options(MENU_OPTIONS, PRIMARY_KEYS, "OncoVision : MENU CHÍNH", print_fn=print_fn)


def _render_medical_menu(print_fn=print) -> None:
    _render_options(MEDICAL_OPTIONS, MEDICAL_PRIMARY_KEYS, "OncoVision : MEDICAL MENU", print_fn=print_fn)


def _render_check_menu(print_fn=print) -> None:
    _render_options(CHECK_OPTIONS, CHECK_PRIMARY_KEYS, "OncoVision : MENU KIỂM TRA", print_fn=print_fn)


def _clear_terminal() -> None:
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def _run_script(script_name: str, *script_args: str, env: dict[str, str] | None = None) -> int:
    merged_env = None
    if env:
        merged_env = {**os.environ, **env}
    return subprocess.call([sys.executable, script_name, *script_args], cwd=PROJECT_ROOT, env=merged_env)


def _resolve_medical_args(option: MenuOption, input_fn=input, print_fn=print) -> tuple[str, ...] | None:
    if option.script != "run_medical.py":
        return option.args
    if option.args and option.args[0] not in {"analyze", "validate-image"}:
        return option.args
    image_path = input_fn("Nhập đường dẫn ảnh y khoa: ").strip()
    if not image_path:
        print_fn(f"{RED}Chưa nhập đường dẫn ảnh. Quay lại menu y dược.{RESET}")
        return None
    if option.args and option.args[0] == "validate-image":
        return ("validate-image", "--image", image_path)
    patient_code = input_fn("Nhập mã bệnh nhân: ").strip() or "BN001"
    return ("analyze", "--image", image_path, "--patient-code", patient_code)


def _is_training_command(args: tuple[str, ...]) -> bool:
    return bool(args) and args[0] in {"train", "train-all"}


def _split_steps(args: tuple[str, ...]) -> list[tuple[str, ...]]:
    steps: list[tuple[str, ...]] = []
    current: list[str] = []
    for token in args:
        if token == "|":
            if current:
                steps.append(tuple(current))
                current = []
            continue
        current.append(token)
    if current:
        steps.append(tuple(current))
    return steps


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
    steps = _split_steps(args)
    multi_step = len(steps) > 1
    for step_index, step_args in enumerate(steps, start=1):
        if not step_args:
            continue
        if multi_step:
            print_fn(f"{BOLD}{option.color}{'=' * width}{RESET}")
            print_fn(f"{BOLD}{option.color}  BƯỚC {step_index}/{len(steps)}: {' '.join(step_args)}{RESET}")
            print_fn(f"{BOLD}{option.color}{'-' * width}{RESET}")
        training = _is_training_command(step_args)
        if training:
            print_fn(f"{BOLD}{YELLOW}{'=' * width}{RESET}")
            print_fn(f"{BOLD}{YELLOW}  ĐANG TRAIN ... (quá trình có thể mất vài phút){RESET}")
            print_fn(f"{BOLD}{YELLOW}  Tiến trình từng epoch/batch sẽ hiển thị bên dưới.{RESET}")
            print_fn(f"{BOLD}{YELLOW}{'=' * width}{RESET}")
        print_fn(f"{BOLD}{option.color}{'-' * width}{RESET}")
        print_fn(f"{BOLD}{option.color}  ĐANG CHẠY: {option.title}{RESET}")
        command_text = f"python {option.script} {' '.join(step_args)}".strip()
        print_fn(f"{option.color}  Lệnh   : {command_text}{RESET}")
        print_fn(f"{option.color}  Ghi chú: {option.description}{RESET}")
        print_fn(f"{BOLD}{option.color}{'-' * width}{RESET}")
        try:
            if training:
                exit_code = run_script_fn(option.script, *step_args, env={"PYTHONUNBUFFERED": "1"})
            else:
                exit_code = run_script_fn(option.script, *step_args)
        except OSError as exc:
            print_fn(f"{RED}Không thể chạy {option.script}: {exc}{RESET}")
            return
        status = "Đã chạy xong" if exit_code == 0 else "Kết thúc với lỗi"
        color = GREEN if exit_code == 0 else YELLOW
        print_fn(f"{color}{status} {option.script}. {back_text} (exit={exit_code}){RESET}")
        if multi_step and step_index < len(steps) and exit_code != 0:
            print_fn(f"{YELLOW}Bước {step_index} lỗi, dừng luồng cải tiến.{RESET}")
            return
    if multi_step:
        print_fn(f"{GREEN}{BOLD}Hoàn tất luồng {option.title} ({len(steps)} bước).{RESET}")


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
    except EOFError:
        print_fn(f"{YELLOW}{TESTED_EXIT_TEXT}{RESET}")
        return 0
    except KeyboardInterrupt:
        print_fn(f"{YELLOW}\nĐã thoát menu.{RESET}")
        return 0


if __name__ == "__main__":
    raise SystemExit(run_entrypoint(main))
