from __future__ import annotations

from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from training.common import GREEN, RED, YELLOW, count_files, print_help_screen, require_yolo
    from training.model_paths import resolve_data_config_path, resolve_model_source, resolve_trained_model_path
except ModuleNotFoundError:
    from common import GREEN, RED, YELLOW, count_files, print_help_screen, require_yolo
    from model_paths import resolve_data_config_path, resolve_model_source, resolve_trained_model_path


YOLO = None
ULTRALYTICS_IMPORT_ERROR = None
PROCESSED_VAL_DIR = Path("dataset/object_detection/processed/images/val")


def _require_yolo():
    global YOLO, ULTRALYTICS_IMPORT_ERROR
    YOLO, ULTRALYTICS_IMPORT_ERROR = require_yolo(YOLO, ULTRALYTICS_IMPORT_ERROR)
    return YOLO


def resolve_validation_model_path():
    model_path = resolve_trained_model_path(required=False, fallback="yolo11n.pt")
    if model_path == Path(model_path.name):
        model_path = Path("models/pretrained") / model_path.name
    return resolve_model_source(model_path)


def _ensure_validation_dataset_ready() -> None:
    if not PROCESSED_VAL_DIR.exists() or not any(PROCESSED_VAL_DIR.iterdir()):
        raise FileNotFoundError(
            "Chưa có ảnh trong dataset/object_detection/processed/images/val. Hãy bỏ dữ liệu vào dataset/object_detection/raw và chạy training/split_dataset.py trước."
        )


def _print_validation_ready_help(error: FileNotFoundError) -> None:
    val_count = count_files(PROCESSED_VAL_DIR)
    print_help_screen(
        title="YOLO VALIDATION :: DỮ LIỆU CHƯA SẴN SÀNG",
        reason=str(error),
        checks=[("Val images", f"{PROCESSED_VAL_DIR} ({val_count} file)", GREEN if val_count > 0 else RED)],
        steps=[
            ("Bước 1", "Bỏ ảnh và label vào dataset/object_detection/raw", YELLOW),
            ("Bước 2", "Chạy training/validate_dataset.py", YELLOW),
            ("Bước 3", "Chạy training/split_dataset.py", YELLOW),
            ("Bước 4", "Chạy lại training/validate_model.py", GREEN),
        ],
        meaning="Dùng tập val trong dataset/object_detection/processed để đo kết quả model.",
        commands=[
            r".\.venv\Scripts\python training\validate_dataset.py",
            r".\.venv\Scripts\python training\split_dataset.py",
            r".\.venv\Scripts\python training\validate_model.py",
        ],
    )


def main() -> None:
    try:
        _ensure_validation_dataset_ready()
    except FileNotFoundError as exc:
        _print_validation_ready_help(exc)
        raise SystemExit(1)
    metrics = _require_yolo()(str(resolve_validation_model_path())).val(
        data=str(resolve_data_config_path()),
        project=str(Path("runs/val").resolve()),
        name="validation",
    )
    print(metrics)


if __name__ == "__main__":
    main()
