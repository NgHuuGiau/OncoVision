from __future__ import annotations

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from training.split_dataset import audit_raw_dataset
    from training.terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section
except ModuleNotFoundError:
    from split_dataset import audit_raw_dataset
    from terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section


def main() -> None:
    report = audit_raw_dataset()
    valid_count = len(report.eligible)
    total_images = report.raw_image_count
    for item in header("YOLO DATASET :: KIỂM TRA DỮ LIỆU RAW"):
        print(item)
    print(section("TỔNG QUAN", GREEN if total_images > 0 else YELLOW))
    print(row("Tổng ảnh raw", str(total_images), GREEN if total_images > 0 else YELLOW))
    print(row("Ảnh hợp lệ", str(valid_count), GREEN if valid_count > 0 else YELLOW))
    print(row("Ảnh thiếu label", str(len(report.missing_labels)), RED if report.missing_labels else GREEN))
    print(row("Label rỗng", str(len(report.empty_labels)), YELLOW if report.empty_labels else GREEN))
    print(row("Label lỗi", str(len(report.invalid_labels)), RED if report.invalid_labels else GREEN))
    print(row("Label mồ côi", str(len(report.orphan_labels)), YELLOW if report.orphan_labels else GREEN))

    if total_images == 0:
        print(line(rule("-"), CYAN))
        print(section("KẾT LUẬN", RED))
        print(row("Lý do không chạy", "Chưa có dữ liệu raw.", RED))
        print(row("Gợi ý", "Bỏ ảnh vào dataset/raw/images và label vào dataset/raw/labels", YELLOW, bounded=False))
        print(line(rule("-"), CYAN))
        print(section("LỆNH TIẾP", CYAN))
        print(command_row(1, r".\.venv\Scripts\python training\prepare_dataset.py"))
        print(command_row(2, r".\.venv\Scripts\python training\validate_dataset.py"))
        print(line(rule("="), CYAN))
        raise SystemExit(1)

    if report.missing_labels or report.invalid_labels:
        print(line(rule("-"), CYAN))
        print(section("CẦN SỬA", RED))
        for path in report.missing_labels[:5]:
            print(row("Lý do không chạy", f"Thiếu label: {path}", RED, bounded=False))
        for path, reason in report.invalid_labels[:5]:
            print(row("Lý do không chạy", f"Label lỗi: {path} | {reason}", RED, bounded=False))
        print(line(rule("-"), CYAN))
        print(section("Ý NGHĨA LỆNH", CYAN))
        print(row("Lệnh này", "Chỉ kiểm tra dữ liệu raw, chưa copy và chưa chia split.", YELLOW, bounded=False))
        print(line(rule("="), CYAN))
        raise SystemExit(1)

    print(line(rule("-"), CYAN))
    print(section("SẴN SÀNG", GREEN))
    print(row("Trạng thái", "Dataset raw hợp lệ để split/train.", GREEN, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("LỆNH TIẾP", CYAN))
    print(command_row(1, r".\.venv\Scripts\python training\split_dataset.py"))
    print(command_row(2, r".\.venv\Scripts\python run_train.py"))
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    main()
