from __future__ import annotations

import shutil
from pathlib import Path
import hashlib

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from training.auto_label_raw import auto_label_raw_images
    from training.common import GREEN, RED, YELLOW, count_files, print_help_screen, require_yolo
    from training.model_paths import TRAINED_BEST_MODEL_PATH, resolve_data_config_path, resolve_model_source
    from training.split_dataset import _copy_split, _reset_processed_dirs, _split_items, audit_raw_dataset
except ModuleNotFoundError:
    from auto_label_raw import auto_label_raw_images
    from common import GREEN, RED, YELLOW, count_files, print_help_screen, require_yolo
    from model_paths import TRAINED_BEST_MODEL_PATH, resolve_data_config_path, resolve_model_source
    from split_dataset import _copy_split, _reset_processed_dirs, _split_items, audit_raw_dataset

from utils.file_utils import ensure_project_directories, load_yaml
from utils.logger import get_logger
from utils.terminal_encoding import ensure_utf8_console

logger = get_logger(__name__)
YOLO = None
ULTRALYTICS_IMPORT_ERROR = None
PROCESSED_TRAIN_DIR = Path("dataset/object_detection/processed/images/train")
PROCESSED_VAL_DIR = Path("dataset/object_detection/processed/images/val")
RAW_IMAGES_DIR = Path("dataset/object_detection/raw/images")
RAW_LABELS_DIR = Path("dataset/object_detection/raw/labels")
AUTO_PREPARE_STATE_PATH = Path("training/.auto_prepare_state")


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


def _dataset_signature() -> str:
    digest = hashlib.sha256()
    for root in (RAW_IMAGES_DIR, RAW_LABELS_DIR):
        digest.update(str(root).encode("utf-8"))
        if not root.exists():
            digest.update(b"<missing>")
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            digest.update(relative.encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _load_auto_prepare_state() -> str | None:
    try:
        return AUTO_PREPARE_STATE_PATH.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


def _save_auto_prepare_state(signature: str) -> None:
    AUTO_PREPARE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTO_PREPARE_STATE_PATH.write_text(signature, encoding="utf-8")


def _auto_label_device() -> str:
    try:
        import torch  # type: ignore

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _ensure_training_dataset_ready() -> None:
    required_dirs = (
        (PROCESSED_TRAIN_DIR, "dataset/object_detection/processed/images/train"),
        (PROCESSED_VAL_DIR, "dataset/object_detection/processed/images/val"),
    )
    for directory, label in required_dirs:
        if not directory.exists() or not any(directory.iterdir()):
            raise FileNotFoundError(
                f"Chưa có ảnh trong {label}. Hãy bỏ dữ liệu vào dataset/object_detection/raw và chạy training/split_dataset.py trước."
            )


def _auto_prepare_training_dataset() -> dict[str, object]:
    ensure_project_directories()
    signature = _dataset_signature()
    previous_signature = _load_auto_prepare_state()
    report = {
        "raw_images": 0,
        "auto_labeled": 0,
        "eligible": 0,
        "no_detection": [],
        "skipped_rebuild": False,
        "device": None,
    }
    audit = audit_raw_dataset()
    report["raw_images"] = audit.raw_image_count
    if not audit.raw_image_count:
        if previous_signature != signature:
            _save_auto_prepare_state(signature)
        return report
    if previous_signature == signature and not audit.missing_labels and all(directory.exists() and any(directory.iterdir()) for directory in (PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR)):
        report["eligible"] = len(audit.eligible)
        report["skipped_rebuild"] = True
        return report
    if audit.missing_labels:
        auto_label_device = _auto_label_device()
        auto_report = auto_label_raw_images(overwrite=False, conf=0.25, device=auto_label_device)
        report["auto_labeled"] = int(auto_report.get("generated", 0))
        report["no_detection"] = list(auto_report.get("no_detection", []))
        report["device"] = auto_label_device
        audit = audit_raw_dataset()
    eligible = audit.eligible
    report["eligible"] = len(eligible)
    if not eligible:
        _save_auto_prepare_state(_dataset_signature())
        return report
    _reset_processed_dirs()
    for split_name, items in _split_items(eligible).items():
        _copy_split(split_name, items)
    _save_auto_prepare_state(_dataset_signature())
    return report


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
        meaning="Đọc dataset đã split trong dataset/object_detection/processed và bắt đầu huấn luyện.",
        commands=[
            r".\.venv\Scripts\python training\validate_dataset.py",
            r".\.venv\Scripts\python training\split_dataset.py",
            r".\.venv\Scripts\python run_train.py",
        ],
    )


def main() -> None:
    ensure_utf8_console()
    ensure_project_directories()
    _auto_prepare_training_dataset()
    try:
        _ensure_training_dataset_ready()
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
