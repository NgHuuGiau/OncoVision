from __future__ import annotations

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from training.terminal_ui import CYAN, GREEN, YELLOW, header, line, row, rule, section
from utils.file_utils import ensure_project_directories
from utils.terminal_encoding import ensure_utf8_console


def main() -> None:
    ensure_utf8_console()
    ensure_project_directories()
    for item in header("YOLO DATASET :: CHUẨN BỊ THƯ MỤC"):
        print(item)
    print(section("TRẠNG THÁI", GREEN))
    print(row("Kết quả", "Đã tạo sẵn các thư mục dataset.", GREEN))
    print(line(rule("-"), CYAN))
    print(section("ĐƯỜNG DẪN", YELLOW))
    print(row("Raw", "dataset/raw"))
    print(row("Processed", "dataset/processed"))
    print(line(rule("-"), CYAN))
    print(section("Ý NGHĨA LỆNH", CYAN))
    print(row("Lệnh này", "Chỉ tạo sẵn raw và processed, chưa train và chưa split.", YELLOW, bounded=False))
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    main()
