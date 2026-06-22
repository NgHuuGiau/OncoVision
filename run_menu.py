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
    "1": MenuOption("run_app.py", "Camera nhận diện", "Mở camera realtime, chạy YOLO và hiển thị FPS cùng khung phát hiện.", "CHẠY CHÍNH", GREEN),
    "2": MenuOption("run_chat.py", "Chat y dược", "Mở giao diện chat để tải ảnh y khoa lên phân tích và xem kết quả.", "CHẠY CHÍNH", GREEN),
    "3": MenuOption("run_tests.py", "Kiểm thử", "Chạy toàn bộ unit test và các kiểm tra hồi quy chính.", "KIỂM TRA", YELLOW),
    "4": MenuOption("run_doctor.py", "Doctor", "Rà soát phần cứng, model, camera, dữ liệu và gợi ý runtime.", "KIỂM TRA", YELLOW),
    "5": MenuOption("run_train.py", "Huấn luyện chung", "Chuẩn bị dữ liệu, huấn luyện, đánh giá và xuất model tổng quát.", "HUẤN LUYỆN", CYAN),
    "6": MenuOption("run_medical.py", "Medical menu", "Mở menu y dược: dataset, train, validate, phân tích ảnh và lịch sử ca.", "Y DƯỢC", CYAN),
    "7": MenuOption("run_chat.py", "Dọn output chat", "Xóa nhanh ảnh camera capture cũ trong output chat.", "BẢO TRÌ", YELLOW, ("--cleanup-output",)),
    "8": MenuOption("run_medical.py", "Dọn output medical", "Xóa nhanh report, ảnh xử lý và export medical cũ.", "BẢO TRÌ", YELLOW, ("cleanup-output",)),
    "9": MenuOption("run_smoke.py", "Smoke check", "Chạy chuỗi kiểm tra an toàn cho các entrypoint chính.", "KIỂM TRA", YELLOW),
    "0": MenuOption("", "Thoát", "Đóng menu terminal.", "HỆ THỐNG", RED),
}
PRIMARY_KEYS = tuple(key for key in MENU_OPTIONS if key != "0")
TESTED_EXIT_TEXT = "Đã thoát menu."
TESTED_INVALID_TEXT = "Lựa chọn không hợp lệ. Hãy nhập lại."
TESTED_BACK_TEXT = "Quay lại menu."
MENU_PROMPT = f"Chọn tác vụ ({'/'.join(('0', *PRIMARY_KEYS))}): "
MEDICAL_BACK_TEXT = "Quay lại menu chính."

MEDICAL_OPTIONS: dict[str, MenuOption] = {
    "1": MenuOption("run_medical.py", "Khởi tạo dataset", "Tạo cấu trúc dataset y dược mặc định.", "DATASET", GREEN, ("init-dataset",)),
    "2": MenuOption("run_medical.py", "Kiểm tra dataset", "Rà soát raw images/raw labels cho pipeline medical.", "DATASET", GREEN, ("audit-dataset",)),
    "3": MenuOption("run_medical.py", "Chia train val test", "Chia dataset medical từ raw sang processed.", "DATASET", GREEN, ("split-dataset",)),
    "4": MenuOption("run_medical.py", "Huấn luyện toàn bộ", "Chạy split, train, validate cho model y dược.", "HUẤN LUYỆN", CYAN, ("train-all",)),
    "5": MenuOption("run_medical.py", "Xem lịch sử ca", "Hiển thị các ca bệnh đã phân tích gần đây.", "KẾT QUẢ", YELLOW, ("history",)),
    "6": MenuOption("run_medical.py", "Phân tích ảnh", "Phân tích một ảnh y khoa với mã bệnh nhân nhập tay.", "KẾT QUẢ", YELLOW),
    "7": MenuOption("run_medical.py", "Medical status", "Kiểm tra nhanh model, dataset, case database và output medical.", "BẢO TRÌ", YELLOW, ("status",)),
    "8": MenuOption("run_medical.py", "Dọn output medical", "Xóa report, ảnh xử lý và export medical cũ.", "BẢO TRÌ", YELLOW, ("cleanup-output",)),
    "0": MenuOption("", "Quay lại", "Trở về menu chính.", "HỆ THỐNG", RED),
}
MEDICAL_PRIMARY_KEYS = tuple(key for key in MEDICAL_OPTIONS if key != "0")
MEDICAL_PROMPT = f"Chọn tác vụ medical ({'/'.join(('0', *MEDICAL_PRIMARY_KEYS))}): "


def _configure_terminal_encoding() -> None:
    ensure_utf8_console()


def _command_text(option: MenuOption) -> str:
    if not option.script:
        return "exit"
    return f"python {option.script}{(' ' + ' '.join(option.args)) if option.args else ''}"


def _option_label(key: str, option: MenuOption) -> tuple[str, str]:
    return f"{key} | {option.title}", f"{option.description} [{_command_text(option)}]"


def _render_options(options: dict[str, MenuOption], primary_keys: tuple[str, ...], title: str, print_fn=print) -> None:
    for item in header(title):
        print_fn(item)
    current_group = ""
    for key in (*primary_keys, "0"):
        option = options[key]
        if option.group != current_group:
            if current_group:
                print_fn(line(rule("-"), CYAN))
            print_fn(section(option.group, option.color))
            current_group = option.group
        label, description = _option_label(key, option)
        print_fn(row(label, description, option.color, bounded=False))
    print_fn(line(rule("="), CYAN))


def _render_menu(print_fn=print) -> None:
    _render_options(MENU_OPTIONS, PRIMARY_KEYS, "YOLO HUB : MENU ĐIỀU KHIỂN", print_fn=print_fn)


def _render_medical_menu(print_fn=print) -> None:
    _render_options(MEDICAL_OPTIONS, MEDICAL_PRIMARY_KEYS, "YOLO HUB : MEDICAL MENU", print_fn=print_fn)


def _clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _run_script(script_name: str, *script_args: str) -> int:
    return subprocess.call([sys.executable, script_name, *script_args])


def _resolve_medical_args(option: MenuOption, input_fn=input, print_fn=print) -> tuple[str, ...] | None:
    if option.script != "run_medical.py" or option.args:
        return option.args
    image_path = input_fn("Nhập đường dẫn ảnh y khoa: ").strip()
    if not image_path:
        print_fn(line("Chưa nhập đường dẫn ảnh. Quay lại menu medical.", RED))
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
    print_fn(line(rule("="), CYAN))
    print_fn(section(f"ĐANG CHẠY: {option.title}", option.color))
    command_text = f"python {option.script} {' '.join(args)}".strip()
    print_fn(row("Lệnh", command_text, option.color, bounded=False))
    print_fn(row("Ghi chú", option.description, option.color, bounded=False))
    print_fn(line(rule("-"), CYAN))
    exit_code = run_script_fn(option.script, *args)
    message = (
        f"Đã chạy xong {option.script}. {back_text}"
        if exit_code == 0
        else f"{option.script} kết thúc với mã {exit_code}. {back_text}"
    )
    print_fn(line(message, GREEN if exit_code == 0 else YELLOW))


def _run_medical_menu(input_fn=input, print_fn=print, run_script_fn=_run_script, clear_terminal_fn=_clear_terminal) -> None:
    while True:
        _render_medical_menu(print_fn=print_fn)
        choice = input_fn(MEDICAL_PROMPT).strip()
        if choice == "0":
            print_fn(line(MEDICAL_BACK_TEXT, YELLOW))
            return
        option = MEDICAL_OPTIONS.get(choice)
        if option is None:
            print_fn(line(TESTED_INVALID_TEXT, RED))
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
        print_fn(line(TESTED_INVALID_TEXT, RED))
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
    while True:
        _render_menu(print_fn=print_fn)
        choice = input_fn(MENU_PROMPT).strip()
        if choice == "0":
            print_fn(line(TESTED_EXIT_TEXT, YELLOW))
            return 0
        _run_menu_choice(
            choice,
            input_fn=input_fn,
            print_fn=print_fn,
            run_script_fn=run_script_fn,
            clear_terminal_fn=clear_terminal_fn,
        )


if __name__ == "__main__":
    raise SystemExit(main())
