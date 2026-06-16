from __future__ import annotations

from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from training.common import GREEN, RED, print_help_screen, require_yolo
    from training.model_paths import resolve_trained_model_path
except ModuleNotFoundError:
    from common import GREEN, RED, print_help_screen, require_yolo
    from model_paths import resolve_trained_model_path


YOLO = None
ULTRALYTICS_IMPORT_ERROR = None
TRAINED_BEST_MODEL_PATH = Path("models/trained/best.pt")


def _require_yolo():
    global YOLO, ULTRALYTICS_IMPORT_ERROR
    YOLO, ULTRALYTICS_IMPORT_ERROR = require_yolo(YOLO, ULTRALYTICS_IMPORT_ERROR)
    return YOLO


def resolve_export_model_path():
    return resolve_trained_model_path(required=True)


def _print_export_ready_help(error: FileNotFoundError) -> None:
    exists = TRAINED_BEST_MODEL_PATH.exists()
    print_help_screen(
        title="YOLO EXPORT :: MODEL CHƯA SẴN SÀNG",
        reason=str(error),
        checks=[("Best model", f"{TRAINED_BEST_MODEL_PATH} ({'có' if exists else 'chưa có'})", GREEN if exists else RED)],
        steps=[
            ("Bước 1", "Chuẩn bị dataset và chạy train trước", GREEN if exists else RED),
            ("Bước 2", "Đảm bảo có models/trained/best.pt", GREEN if exists else RED),
            ("Bước 3", "Chạy lại training/export_model.py", GREEN),
        ],
        meaning="Xuất models/trained/best.pt sang định dạng ONNX để deploy.",
        commands=[
            r".\.venv\Scripts\python run_train.py",
            r".\.venv\Scripts\python training\export_model.py",
        ],
    )


def main() -> None:
    try:
        model_path = resolve_export_model_path()
    except FileNotFoundError as exc:
        _print_export_ready_help(exc)
        raise SystemExit(1)
    _require_yolo()(str(model_path)).export(format="onnx")
    print("Export model xong.")


if __name__ == "__main__":
    main()
