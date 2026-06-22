from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from training.model_paths import resolve_model_source
from training.train_model import (
    PROCESSED_TRAIN_DIR,
    PROCESSED_VAL_DIR,
    RAW_IMAGES_DIR,
    RAW_LABELS_DIR,
    main as run_training_main,
)
from utils.file_utils import ensure_project_directories, load_yaml


TRAIN_CONFIG_PATH = Path("training/train_config.yaml")


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file())


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Huấn luyện model YOLO custom từ dataset local.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Chỉ kiểm tra nhanh config, model và dữ liệu; không chạy train.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def run_train_preflight(print_fn=print) -> int:
    ensure_project_directories()
    config = load_yaml(TRAIN_CONFIG_PATH)

    raw_images = _count_files(RAW_IMAGES_DIR)
    raw_labels = _count_files(RAW_LABELS_DIR)
    train_images = _count_files(PROCESSED_TRAIN_DIR)
    val_images = _count_files(PROCESSED_VAL_DIR)
    primary_model = resolve_model_source(config["model"])
    fallback_model = resolve_model_source(config.get("fallback_model", "yolo11n.pt"))

    modules_ok = {
        "ultralytics": _module_available("ultralytics"),
        "torch": _module_available("torch"),
    }
    missing_models: list[str] = []
    if not primary_model.exists():
        missing_models.append(str(primary_model))
    if not fallback_model.exists() and fallback_model != primary_model:
        missing_models.append(str(fallback_model))

    hard_failure = not all(modules_ok.values())
    processed_ready = train_images > 0 and val_images > 0
    raw_ready = raw_images > 0

    print_fn("YOLO training preflight")
    print_fn(f"- Config: {TRAIN_CONFIG_PATH}")
    print_fn(f"- Primary model: {primary_model} | exists={primary_model.exists()}")
    print_fn(f"- Fallback model: {fallback_model} | exists={fallback_model.exists()}")
    print_fn(
        "- Dataset counts: "
        f"raw_images={raw_images}, raw_labels={raw_labels}, train={train_images}, val={val_images}"
    )
    print_fn(
        "- Python deps: "
        f"ultralytics={modules_ok['ultralytics']}, torch={modules_ok['torch']}"
    )

    if hard_failure:
        print_fn("- Status: lỗi cấu hình hoặc thiếu dependency, chưa thể chạy run_train.py.")
        return 1

    if missing_models:
        print_fn(
            "- Warning: thiếu pretrained model "
            + ", ".join(missing_models)
            + ". Có thể tải lại bằng: .\\.venv\\Scripts\\python training\\download_models.py"
        )

    if processed_ready:
        print_fn("- Status: dữ liệu train/val đã sẵn sàng, có thể chạy train ngay.")
        return 0

    if raw_ready:
        print_fn("- Status: đã có raw dataset, run_train.py có thể thử auto-prepare trước khi train.")
        return 0

    print_fn("- Status: entrypoint chạy được nhưng hiện chưa có dataset để train custom.")
    print_fn("- Gợi ý: thêm dữ liệu vào dataset/raw rồi chạy lại run_train.py.")
    return 0


def main() -> int:
    args = parse_args()
    if getattr(args, "check_only", False):
        return run_train_preflight()

    result = run_training_main()
    return 0 if result is None else int(result)


__all__ = ["main", "run_train_preflight", "run_training_main", "parse_args", "build_parser"]


if __name__ == "__main__":
    raise SystemExit(main())
