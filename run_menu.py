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
DIM = "\033[2m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"

TESTED_EXIT_TEXT = "Đã thoát menu."
TESTED_INVALID_TEXT = "Lựa chọn không hợp lệ. Hãy nhập lại."


@dataclass(frozen=True)
class MenuOption:
    script: str
    title: str
    description: str
    group: str
    color: str
    icon: str
    args: tuple[str, ...] = ()


MENU_OPTIONS: dict[str, MenuOption] = {
    "1": MenuOption("run_app.py", "Camera realtime", "Mở camera, chạy model và xem kết quả ngay.", "CHẠY NHANH", GREEN, "📷"),
    "2": MenuOption("run_chat.py", "Chat y dược", "Mở chat UI và luồng phân tích ảnh y khoa.", "CHẠY NHANH", GREEN, "💬"),
    "3": MenuOption("run_medical.py", "Y dược", "Quản lý dataset, lịch sử ca và luồng medical.", "Y DƯỢC", CYAN, "🏥"),
    "4": MenuOption("", "Kiểm tra", "Mở menu kiểm tra (Doctor, Test, Smoke).", "KIỂM TRA", YELLOW, "🔍"),
    "5": MenuOption("run_chat.py", "Dọn cache", "Xóa output chat và medical cũ cho repo gọn hơn.", "BẢO TRÌ", YELLOW, "🧹", ("--cleanup-output",)),
    "0": MenuOption("", "Thoát", "Đóng menu terminal.", "HỆ THỐNG", RED, "🚪"),
}

PRIMARY_KEYS = tuple(key for key in MENU_OPTIONS if key != "0")
MENU_PROMPT = f"  Nhập lựa chọn [{'/'.join(('0', *PRIMARY_KEYS))}]: "

MEDICAL_OPTIONS: dict[str, MenuOption] = {
    "1": MenuOption("run_medical.py", "Phân tích ca bệnh", "Thực hiện phân tích ảnh y khoa theo ID", "PHÂN TÍCH", CYAN, "🔬", ("analyze",)),
    "2": MenuOption("run_medical.py", "Lịch sử ca bệnh", "Xem danh sách các ca đã phân tích", "KẾT QUẢ", YELLOW, "📋", ("history",)),
    "3": MenuOption("run_medical.py", "Kiểm tra & Báo cáo", "Xem tóm tắt dữ liệu / Kiểm tra tính hợp lệ ảnh", "KIỂM TRA", GREEN, "✅", ("report",)),
    "4": MenuOption("run_medical.py", "Huấn luyện mô hình", "Khởi tạo split và train pipeline y khoa", "Y DƯỢC", CYAN, "🎓", ("train-all",)),
    "5": MenuOption("run_medical.py", "Cải tiến & Tuning", "Active learning, train modality & hiệu chỉnh", "CẢI TIẾN", YELLOW, "🔧", ("active-learning", "|", "train-modality", "--epochs", "12", "--verbose", "|", "calibrate-modality-tuning", "--apply")),
    "0": MenuOption("", "Quay lại menu chính", "Trở về menu chính.", "HỆ THỐNG", RED, "↩"),
}
MEDICAL_PRIMARY_KEYS = tuple(key for key in MEDICAL_OPTIONS if key != "0")
MEDICAL_PROMPT = f"  Nhập lựa chọn [{'/'.join(('0', *MEDICAL_PRIMARY_KEYS))}]: "
MEDICAL_BACK_TEXT = "Quay lại menu chính."

CHECK_OPTIONS: dict[str, MenuOption] = {
    "1": MenuOption("run_doctor.py", "Doctor", "Rà soát môi trường, model và dataset.", "KIỂM TRA", YELLOW, "🩺", ("--skip-camera-check",)),
    "2": MenuOption("run_tests.py", "Test", "Chạy unit test và regression.", "KIỂM TRA", YELLOW, "🧪", ("--skip-camera-check",)),
    "3": MenuOption("run_smoke.py", "Smoke", "Kiểm tra nhanh các entrypoint chính.", "KIỂM TRA", YELLOW, "💨", ()),
    "4": MenuOption("run_smoke.py", "Smoke + tests", "Smoke check và nối thêm test suite.", "KIỂM TRA", YELLOW, "🔬", ("--include-tests",)),
    "5": MenuOption("run_medical.py", "Kiểm tra ảnh y khoa", "Nhập đường dẫn ảnh để kiểm tra hợp lệ.", "KIỂM TRA", YELLOW, "🖼️", ("validate-image",)),
    "6": MenuOption("", "Kiểm tra toàn bộ", "Chạy doctor → test → smoke → medical status.", "KIỂM TRA", YELLOW, "🚀"),
    "0": MenuOption("", "Quay lại", "Trở về menu chính.", "HỆ THỐNG", RED, "↩"),
}
CHECK_PRIMARY_KEYS = tuple(key for key in CHECK_OPTIONS if key != "0")
CHECK_PROMPT = f"  Nhập lựa chọn [{'/'.join(('0', *CHECK_PRIMARY_KEYS))}]: "


def _progress_bar(percent: int, width: int = 20) -> str:
    percent = max(0, min(100, percent))
    filled = round(width * percent / 100)
    return f"[{'█' * filled}{'░' * (width - filled)}] {percent:3d}%"


def _model_exists(model_name: str) -> bool:
    return os.path.exists(str(Path("models/pretrained") / model_name))


def _ensure_yolo11_models() -> None:
    missing_models = [model_name for model_name in YOLO11_MODELS_ASC if not _model_exists(model_name)]
    if not missing_models:
        return
    total = len(missing_models)
    print(f"{BOLD}{CYAN}{'═' * 78}{RESET}")
    print(f"{BOLD}{CYAN}  ĐANG KIỂM TRA VÀ TẢI MODEL YOLO11{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 78}{RESET}")
    for index, model_name in enumerate(missing_models, start=1):
        print(f"{YELLOW}  [{index}/{total}] {model_name} {_progress_bar(0)}{RESET}")
    downloaded, skipped = download_models(missing_models)
    for index, model_name in enumerate(downloaded, start=1):
        print(f"{GREEN}  ✔ [{index}/{total}] {model_name} {_progress_bar(100)}{RESET}")
    for model_name in skipped:
        print(f"{CYAN}  • Đã có sẵn {model_name}, bỏ qua{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 78}{RESET}")


def _configure_terminal_encoding() -> None:
    ensure_utf8_console()


def _get_terminal_width() -> int:
    try:
        return max(60, min(120, os.get_terminal_size().columns))
    except OSError:
        return 90


def _wrap_text(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [""]


def _center(text: str, width: int) -> str:
    return f"{(width - len(text)) // 2 * ' '}{text}"


def _print_menu_lines(title: str, items: list[tuple[str, str, str]], header_color: str, item_color: str, print_fn=print) -> None:
    width = _get_terminal_width()
    content_width = max(width - 8, 48)
    inner = content_width + 2

    print_fn("")
    print_fn(f"{BOLD}{header_color}  {title}{RESET}")
    print_fn(f"{BOLD}{header_color}  {'─' * (inner - 4)}{RESET}")

    for num, label, desc in items:
        print_fn(f"{BOLD}{item_color}  [{num}] {RESET}{label}")
        wrapped = _wrap_text(desc, content_width - 6)
        for line in wrapped:
            print_fn(f"{BOLD}{item_color}      {DIM}{line}{RESET}")
        print_fn("")

    print_fn(f"{BOLD}{RED}  [0] 🚪 Thoát menu{RESET}")
    print_fn("")


def _render_main_menu(print_fn=print) -> None:
    items = [
        ("1", "Camera realtime", "Mở camera, chạy model và xem kết quả ngay."),
        ("2", "Chat y dược", "Mở chat UI và luồng phân tích ảnh y khoa."),
        ("3", "Y dược", "Quản lý dataset, lịch sử ca và luồng medical."),
        ("4", "Kiểm tra", "Mở menu kiểm tra (Doctor, Test, Smoke)."),
        ("5", "Dọn cache", "Xóa output chat và medical cũ cho repo gọn hơn."),
    ]
    _print_menu_lines("OncoVision", items, CYAN, GREEN, print_fn=print_fn)


def _render_menu(print_fn=print) -> None:
    _render_main_menu(print_fn=print_fn)


def _render_medical_menu(print_fn=print) -> None:
    items = [
        ("1", "Phân tích ca bệnh", "Thực hiện phân tích ảnh y khoa theo ID."),
        ("2", "Lịch sử ca bệnh", "Xem danh sách các ca đã phân tích."),
        ("3", "Kiểm tra & Báo cáo", "Xem tóm tắt dữ liệu / Kiểm tra tính hợp lệ ảnh."),
        ("4", "Huấn luyện mô hình", "Khởi tạo split và train pipeline y khoa."),
        ("5", "Cải tiến & Tuning", "Active learning, train modality & hiệu chỉnh."),
    ]
    _print_menu_lines("MENU Y DƯỢC", items, MAGENTA, CYAN, print_fn=print_fn)


def _render_check_menu(print_fn=print) -> None:
    items = [
        ("1", "Doctor", "Rà soát môi trường, model và dataset."),
        ("2", "Test", "Chạy unit test và regression."),
        ("3", "Smoke", "Kiểm tra nhanh các entrypoint chính."),
        ("4", "Smoke + tests", "Smoke check và nối thêm test suite."),
        ("5", "Kiểm tra ảnh y khoa", "Nhập đường dẫn ảnh để kiểm tra hợp lệ."),
        ("6", "Kiểm tra toàn bộ", "Chạy doctor → test → smoke → medical status."),
    ]
    _print_menu_lines("MENU KIỂM TRA", items, BLUE, YELLOW, print_fn=print_fn)


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
    if option.args and option.args[0] != "analyze":
        return option.args
    image_path = input_fn("Nhập đường dẫn ảnh y khoa: ").strip()
    if not image_path:
        print_fn(f"{RED}Chưa nhập đường dẫn ảnh. Quay lại menu y dược.{RESET}")
        return None
    patient_code = input_fn("Nhập mã bệnh nhân: ").strip() or "BN001"
    return ("analyze", "--image", image_path, "--patient-code", patient_code)


def _is_training_command(args: tuple[str, ...]) -> bool:
    return bool(args) and args[0] in {"train", "train-all"}


def _split_steps(args: tuple[str, ...]) -> list[tuple[str, ...]]:
    if not args:
        return [()]
    steps: list[tuple[str, ...]] = []
    current: list[str] = []
    for token in args:
        if token == "|":
            steps.append(tuple(current))
            current = []
            continue
        current.append(token)
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
        if multi_step and not step_args:
            continue
        if multi_step:
            print_fn(f"{BOLD}{option.color}{'═' * width}{RESET}")
            print_fn(f"{BOLD}{option.color}  BƯỚC {step_index}/{len(steps)}: {' '.join(step_args)}{RESET}")
            print_fn(f"{BOLD}{option.color}{'═' * width}{RESET}")
        training = _is_training_command(step_args)
        if training:
            print_fn(f"{BOLD}{YELLOW}{'═' * width}{RESET}")
            print_fn(f"{BOLD}{YELLOW}  ĐANG TRAIN ... (quá trình có thể mất vài phút){RESET}")
            print_fn(f"{BOLD}{YELLOW}  Tiến trình từng epoch/batch sẽ hiển thị bên dưới.{RESET}")
            print_fn(f"{BOLD}{YELLOW}{'═' * width}{RESET}")
        print_fn(f"{BOLD}{option.color}{'─' * width}{RESET}")
        print_fn(f"{BOLD}{option.color}  ĐANG CHẠY: {option.title}{RESET}")
        command_text = f"python {option.script} {' '.join(step_args)}".strip()
        print_fn(f"{option.color}  Lệnh   : {command_text}{RESET}")
        print_fn(f"{option.color}  Ghi chú: {option.description}{RESET}")
        print_fn(f"{BOLD}{option.color}{'─' * width}{RESET}")
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
        if choice == "6":
            _run_all_checks(input_fn=input_fn, print_fn=print_fn, run_script_fn=run_script_fn, clear_terminal_fn=clear_terminal_fn)
            return
        _run_selected_option(
            option,
            args=option.args,
            print_fn=print_fn,
            run_script_fn=run_script_fn,
            clear_terminal_fn=clear_terminal_fn,
            back_text="Quay lại menu kiểm tra.",
        )


def _run_all_checks(
    *,
    input_fn=input,
    print_fn=print,
    run_script_fn=_run_script,
    clear_terminal_fn=_clear_terminal,
) -> None:
    clear_terminal_fn()
    width = _get_terminal_width()
    steps = [
        ("run_doctor.py", ("--skip-camera-check",), "Bác sĩ kiểm tra"),
        ("run_tests.py", ("--skip-camera-check",), "Bộ test hệ thống"),
        ("run_smoke.py", (), "Kiểm tra khói"),
        ("run_medical.py", ("status",), "Trạng thái y dược"),
    ]
    overall_results: list[tuple[str, str, float]] = []
    start_time = __import__("time").time()

    print_fn(f"{BOLD}{CYAN}{'═' * width}{RESET}")
    print_fn(f"{BOLD}{CYAN}  🩺 OncoVision – KIỂM TRA TOÀN HỆ THỐNG{RESET}")
    print_fn(f"{BOLD}{CYAN}{'═' * width}{RESET}")
    print_fn("")

    for step_index, (script, script_args, title) in enumerate(steps, start=1):
        print_fn(f"{BOLD}{BLUE}{'─' * width}{RESET}")
        print_fn(f"{BOLD}{BLUE}  ▶ Bước {step_index}/{len(steps)}: {title}{RESET}")
        command_text = f"python {script} {' '.join(script_args)}".strip()
        print_fn(f"{DIM}  Lệnh: {command_text}{RESET}")
        print_fn(f"{BOLD}{BLUE}{'─' * width}{RESET}")

        step_start = __import__("time").time()
        try:
            exit_code = run_script_fn(script, *script_args)
        except OSError as exc:
            print_fn(f"{RED}  ✗ Không thể chạy {script}: {exc}{RESET}")
            elapsed = __import__("time").time() - step_start
            overall_results.append((title, "LỖI", elapsed))
            break

        elapsed = __import__("time").time() - step_start
        status = "PASS" if exit_code == 0 else "LỖI"
        color = GREEN if exit_code == 0 else RED
        overall_results.append((title, status, elapsed))
        print_fn(f"{color}  ✔ {title} – {status} (exit={exit_code}, {elapsed:.1f}s){RESET}")

        if exit_code != 0:
            print_fn(f"{YELLOW}  ⚠ Dừng kiểm tra toàn bộ do bước {step_index} lỗi.{RESET}")
            break

    total_time = __import__("time").time() - start_time
    passed = sum(1 for _, s, _ in overall_results if s == "PASS")
    failed = sum(1 for _, s, _ in overall_results if s == "LỖI")

    print_fn("")
    print_fn(f"{BOLD}{CYAN}{'═' * width}{RESET}")
    print_fn(f"{BOLD}{CYAN}  📊 TỔNG KẾT KIỂM TRA TOÀN BỘ{RESET}")
    print_fn(f"{BOLD}{CYAN}{'═' * width}{RESET}")

    for title, status, elapsed in overall_results:
        color = GREEN if status == "PASS" else RED
        icon = "✔" if status == "PASS" else "✗"
        print_fn(f"{color}  {icon} {title}: {status} ({elapsed:.1f}s){RESET}")

    print_fn(f"{CYAN}{'─' * width}{RESET}")
    print_fn(f"{BOLD}  Tổng thời gian: {total_time:.1f}s | Thành công: {passed} | Lỗi: {failed}{RESET}")
    print_fn(f"{BOLD}{CYAN}{'═' * width}{RESET}")


def _run_menu_choice(choice: str, *, input_fn=input, print_fn=print, run_script_fn=_run_script, clear_terminal_fn=_clear_terminal) -> bool:
    option = MENU_OPTIONS.get(choice)
    if option is None:
        print_fn(f"{RED}{TESTED_INVALID_TEXT}{RESET}")
        return False
    if choice == "3":
        clear_terminal_fn()
        _run_medical_menu(input_fn=input_fn, print_fn=print_fn, run_script_fn=run_script_fn, clear_terminal_fn=clear_terminal_fn)
        return True
    if choice == "4":
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
            _render_main_menu(print_fn=print_fn)
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
