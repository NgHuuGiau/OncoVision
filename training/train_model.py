from __future__ import annotations

import importlib
import shutil
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from training.model_paths import TRAINED_BEST_MODEL_PATH, resolve_data_config_path, resolve_model_source
except ModuleNotFoundError:
    from model_paths import TRAINED_BEST_MODEL_PATH, resolve_data_config_path, resolve_model_source

YOLO = None
ULTRALYTICS_IMPORT_ERROR = None

from utils.file_utils import ensure_project_directories, load_yaml
from utils.logger import get_logger


logger = get_logger(__name__)
PROCESSED_TRAIN_DIR = Path("dataset/processed/images/train")
PROCESSED_VAL_DIR = Path("dataset/processed/images/val")
RAW_IMAGES_DIR = Path("dataset/raw/images")
RAW_LABELS_DIR = Path("dataset/raw/labels")

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
CARD_WIDTH = 88


def _require_yolo():
    global YOLO, ULTRALYTICS_IMPORT_ERROR
    if YOLO is None and ULTRALYTICS_IMPORT_ERROR is None:
        try:
            YOLO = importlib.import_module("ultralytics").YOLO
        except Exception as exc:  # pragma: no cover
            ULTRALYTICS_IMPORT_ERROR = exc
    if YOLO is None:
        raise RuntimeError(f"Khong khoi tao duoc ultralytics/YOLO: {ULTRALYTICS_IMPORT_ERROR}")
    return YOLO


def _training_kwargs(config: dict) -> dict:
    kwargs = {key: value for key, value in config.items() if key != "fallback_model"}
    if "project" in kwargs:
        kwargs["project"] = str(Path(kwargs["project"]).resolve())
    if "data" in kwargs:
        kwargs["data"] = str(resolve_data_config_path())
    return kwargs


def _ensure_training_dataset_ready() -> None:
    if not PROCESSED_TRAIN_DIR.exists() or not any(PROCESSED_TRAIN_DIR.iterdir()):
        raise FileNotFoundError(
            "Chua co anh trong dataset/processed/images/train. "
            "Hay bo du lieu vao dataset/raw va chay training/split_dataset.py truoc."
        )
    if not PROCESSED_VAL_DIR.exists() or not any(PROCESSED_VAL_DIR.iterdir()):
        raise FileNotFoundError(
            "Chua co anh trong dataset/processed/images/val. "
            "Hay bo du lieu vao dataset/raw va chay training/split_dataset.py truoc."
        )


def _copy_best_weight(run_dir: Path) -> None:
    best_weight = run_dir / "weights" / "best.pt"
    target = TRAINED_BEST_MODEL_PATH
    if best_weight.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weight, target)
        logger.info("Copied best.pt to %s", target)


def _line(text: str = "", color: str = "") -> str:
    return f"{color}{text}{RESET}" if color else text


def _pad(text: str, width: int = CARD_WIDTH) -> str:
    return text[:width].ljust(width)


def _rule(char: str = "=") -> str:
    return char * CARD_WIDTH


def _section(title: str, color: str) -> str:
    return _line(_pad(f"[ {title} ]"), BOLD + color)


def _row(label: str, value: str = "", color: str = "", *, bounded: bool = True) -> str:
    content = f"{label:<18} {value}".rstrip()
    return _line(_pad(content) if bounded else content, color)


def _print_dataset_ready_help(error: FileNotFoundError) -> None:
    raw_images_count = len(list(RAW_IMAGES_DIR.glob("*"))) if RAW_IMAGES_DIR.exists() else 0
    raw_labels_count = len(list(RAW_LABELS_DIR.glob("*"))) if RAW_LABELS_DIR.exists() else 0
    raw_images_color = GREEN if raw_images_count > 0 else RED
    raw_labels_color = GREEN if raw_labels_count > 0 else RED
    print(_line(_rule("="), CYAN))
    print(_line(_pad("YOLO TRAINING :: DU LIEU CHUA SAN SANG"), BOLD + CYAN))
    print(_line(_rule("="), CYAN))
    print(_section("LY DO", RED))
    print(_row("Trang thai", str(error), RED, bounded=False))
    print(_line(_rule("-"), CYAN))
    print(_section("KIEM TRA NHANH", YELLOW))
    print(_row("Raw images", f"{RAW_IMAGES_DIR} ({raw_images_count} file)", raw_images_color, bounded=False))
    print(_row("Raw labels", f"{RAW_LABELS_DIR} ({raw_labels_count} file)", raw_labels_color, bounded=False))
    print(_line(_rule("-"), CYAN))
    print(_section("CAC BUOC CAN LAM", GREEN))
    print(_row("Buoc 1", f"Bo anh vao {RAW_IMAGES_DIR}" if raw_images_count == 0 else f"Da co {raw_images_count} anh trong {RAW_IMAGES_DIR}", raw_images_color, bounded=False))
    print(_row("Buoc 2", f"Bo label vao {RAW_LABELS_DIR}" if raw_labels_count == 0 else f"Da co {raw_labels_count} label trong {RAW_LABELS_DIR}", raw_labels_color, bounded=False))
    print(_row("Buoc 3", "Chay training/validate_dataset.py", YELLOW))
    print(_row("Buoc 4", "Chay training/split_dataset.py", YELLOW))
    print(_row("Buoc 5", "Chay lai run_train.py", GREEN))
    print(_line(_rule("-"), CYAN))
    print(_section("LENH NHANH", CYAN))
    print(_row("Lenh 1", r".\.venv\Scripts\python training\validate_dataset.py", CYAN, bounded=False))
    print(_row("Lenh 2", r".\.venv\Scripts\python training\split_dataset.py", CYAN, bounded=False))
    print(_row("Lenh 3", r".\.venv\Scripts\python run_train.py", CYAN, bounded=False))
    print(_line(_rule("="), CYAN))


def main() -> None:
    ensure_project_directories()
    try:
        _ensure_training_dataset_ready()
    except FileNotFoundError as exc:
        _print_dataset_ready_help(exc)
        raise SystemExit(1)
    config = load_yaml("training/train_config.yaml")
    model_name = config["model"]
    yolo_cls = _require_yolo()
    try:
        model = yolo_cls(str(resolve_model_source(model_name)))
        results = model.train(**_training_kwargs(config))
    except Exception as exc:
        logger.warning("Primary training config failed: %s", exc)
        fallback_model = config.get("fallback_model", "yolo11n.pt")
        config["model"] = fallback_model
        config["imgsz"] = min(int(config["imgsz"]), 416)
        config["batch"] = min(int(config["batch"]), 4)
        model = yolo_cls(str(resolve_model_source(fallback_model)))
        results = model.train(**_training_kwargs(config))

    save_dir = Path(getattr(results, "save_dir", config["project"]))
    _copy_best_weight(save_dir)
    logger.info("Training completed. Output: %s", save_dir)


if __name__ == "__main__":
    main()
