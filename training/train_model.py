from __future__ import annotations

import shutil
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from training.common import GREEN, RED, YELLOW, count_files, print_help_screen, require_yolo
    from training.model_paths import TRAINED_BEST_MODEL_PATH, resolve_data_config_path, resolve_model_source
except ModuleNotFoundError:
    from common import GREEN, RED, YELLOW, count_files, print_help_screen, require_yolo
    from model_paths import TRAINED_BEST_MODEL_PATH, resolve_data_config_path, resolve_model_source

from utils.file_utils import ensure_project_directories, load_yaml
from utils.logger import get_logger
from utils.terminal_encoding import ensure_utf8_console

logger = get_logger(__name__)
YOLO = None
ULTRALYTICS_IMPORT_ERROR = None
RAW_IMAGES_DIR = Path("dataset/raw/images")
RAW_LABELS_DIR = Path("dataset/raw/labels")
PROCESSED_TRAIN_DIR = Path("dataset/processed/images/train")
PROCESSED_VAL_DIR = Path("dataset/processed/images/val")


def _require_yolo():
    global YOLO, ULTRALYTICS_IMPORT_ERROR
    YOLO, ULTRALYTICS_IMPORT_ERROR = require_yolo(YOLO, ULTRALYTICS_IMPORT_ERROR)
    return YOLO


def _training_kwargs(config: dict) -> dict:
    kwargs = {key: value for key, value in config.items() if key != "fallback_model"}
    if "project" in kwargs:
        kwargs["project"] = str(Path(kwargs["project"]).resolve())
    if "data" in kwargs:
        kwargs["data"] = str(resolve_data_config_path())
    return kwargs


def _apply_fallback_training_config(config: dict) -> dict:
    fallback_config = dict(config)
    fallback_config.update(
        model=fallback_config.get("fallback_model", "yolo11n.pt"),
        imgsz=min(int(fallback_config["imgsz"]), 416),
        batch=min(int(fallback_config["batch"]), 4),
    )
    return fallback_config


def _copy_best_weight(run_dir: Path) -> None:
    best_weight = run_dir / "weights" / "best.pt"
    if best_weight.exists():
        TRAINED_BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weight, TRAINED_BEST_MODEL_PATH)
        logger.info("Copied best.pt to %s", TRAINED_BEST_MODEL_PATH)


def _print_dataset_ready_help(error: FileNotFoundError) -> None:
    raw_images_count = count_files(RAW_IMAGES_DIR)
    raw_labels_count = count_files(RAW_LABELS_DIR)
    raw_images_color = GREEN if raw_images_count > 0 else RED
    raw_labels_color = GREEN if raw_labels_count > 0 else RED
    print_help_screen(
        title="OncoVision TRAINING :: DỮ LIỆU CHƯA SẴN SÀNG",
        reason=str(error),
        checks=[
            ("Raw images", f"{RAW_IMAGES_DIR} ({raw_images_count} file)", raw_images_color),
            ("Raw labels", f"{RAW_LABELS_DIR} ({raw_labels_count} file)", raw_labels_color),
        ],
        steps=[
            ("Bước 1", f"Bỏ ảnh vào {RAW_IMAGES_DIR}" if raw_images_count == 0 else f"Đã có {raw_images_count} ảnh trong {RAW_IMAGES_DIR}", raw_images_color),
            ("Bước 2", f"Bỏ label vào {RAW_LABELS_DIR}" if raw_labels_count == 0 else f"Đã có {raw_labels_count} label trong {RAW_LABELS_DIR}", raw_labels_color),
            ("Bước 3", "Chạy training/validate_dataset.py", YELLOW),
            ("Bước 4", "Chạy training/split_dataset.py", YELLOW),
            ("Bước 5", "Chạy lại run_train.py", GREEN),
        ],
        meaning="Đọc dataset đã split trong dataset/processed và bắt đầu huấn luyện.",
        commands=[
            r".\.venv\Scripts\python training\validate_dataset.py",
            r".\.venv\Scripts\python training\split_dataset.py",
            r".\.venv\Scripts\python run_train.py",
        ],
    )


def main() -> None:
    ensure_utf8_console()
    ensure_project_directories()
    try:
        if not PROCESSED_TRAIN_DIR.exists() or not any(PROCESSED_TRAIN_DIR.iterdir()) or not PROCESSED_VAL_DIR.exists() or not any(PROCESSED_VAL_DIR.iterdir()):
            raise FileNotFoundError(
                "Chưa có dữ liệu processed hợp lệ. Hãy chạy training/split_dataset.py trước khi train."
            )
    except FileNotFoundError as exc:
        _print_dataset_ready_help(exc)
        raise SystemExit(1)
    config = load_yaml("training/train_config.yaml")
    yolo_cls = _require_yolo()
    training_kwargs = _training_kwargs(config)
    model_name = config["model"]
    try:
        results = yolo_cls(str(resolve_model_source(model_name))).train(**training_kwargs)
    except Exception as exc:
        logger.warning("Primary training config failed: %s", exc)
        fallback_config = _apply_fallback_training_config(config)
        results = yolo_cls(str(resolve_model_source(fallback_config["model"]))).train(**_training_kwargs(fallback_config))
    save_dir = Path(getattr(results, "save_dir", training_kwargs["project"]))
    _copy_best_weight(save_dir)
    logger.info("Training completed. Output: %s", save_dir)


if __name__ == "__main__":
    main()
