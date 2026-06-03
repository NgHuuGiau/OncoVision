from __future__ import annotations

import importlib
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from training.model_paths import resolve_trained_model_path
except ModuleNotFoundError:
    from model_paths import resolve_trained_model_path

try:
    from training.terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section
except ModuleNotFoundError:
    from terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section

YOLO = None
ULTRALYTICS_IMPORT_ERROR = None
TRAINED_BEST_MODEL_PATH = Path("models/trained/best.pt")


def _require_yolo():
    global YOLO, ULTRALYTICS_IMPORT_ERROR
    if YOLO is None and ULTRALYTICS_IMPORT_ERROR is None:
        try:
            YOLO = importlib.import_module("ultralytics").YOLO
        except Exception as exc:  # pragma: no cover
            ULTRALYTICS_IMPORT_ERROR = exc
    if YOLO is None:
        raise RuntimeError(f"Không khởi tạo được ultralytics/YOLO: {ULTRALYTICS_IMPORT_ERROR}")
    return YOLO


def resolve_export_model_path():
    return resolve_trained_model_path(required=True)


def _print_export_ready_help(error: FileNotFoundError) -> None:
    exists = TRAINED_BEST_MODEL_PATH.exists()
    for item in header("YOLO EXPORT :: MODEL CHƯA SẴN SÀNG", color=RED):
        print(item)
    print(section("LÝ DO", RED))
    print(row("Lý do không chạy", str(error), RED, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("KIỂM TRA NHANH", YELLOW))
    print(row("Best model", f"{TRAINED_BEST_MODEL_PATH} ({'có' if exists else 'chưa có'})", GREEN if exists else RED, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("CÁC BƯỚC CẦN LÀM", GREEN))
    print(row("Bước 1", "Chuẩn bị dataset và chạy train trước", YELLOW, bounded=False))
    print(row("Bước 2", "Đảm bảo có models/trained/best.pt", YELLOW, bounded=False))
    print(row("Bước 3", "Chạy lại training/export_model.py", GREEN))
    print(line(rule("-"), CYAN))
    print(section("Ý NGHĨA LỆNH", CYAN))
    print(row("Lệnh này", "Xuất models/trained/best.pt sang định dạng ONNX để deploy.", YELLOW, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("LỆNH NHANH", CYAN))
    print(command_row(1, r".\.venv\Scripts\python run_train.py"))
    print(command_row(2, r".\.venv\Scripts\python training\export_model.py"))
    print(line(rule("="), CYAN))


def main() -> None:
    try:
        model_path = resolve_export_model_path()
    except FileNotFoundError as exc:
        _print_export_ready_help(exc)
        raise SystemExit(1)
    model = _require_yolo()(str(model_path))
    model.export(format="onnx")
    print("Export model xong.")


if __name__ == "__main__":
    main()
