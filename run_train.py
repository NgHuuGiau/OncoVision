from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from medical.training import audit_medical_raw_dataset, medical_training_paths, run_full_medical_training_pipeline
from utils.entrypoint_common import run_entrypoint
from utils.file_utils import ensure_project_directories
from utils.terminal_encoding import ensure_utf8_console


_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_TRAIN_LOG_PATH = Path("output/medical/train_log.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Huấn luyện model medical 7 ung thư từ dataset/medical.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Chỉ kiểm tra nhanh dataset medical và cấu hình train; không chạy train.",
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Chạy training ở chế độ detached, ghi log ra output/medical/train_log.txt.",
    )
    return parser


def _count_split_items(split_map: dict[str, list[object]]) -> int:
    return sum(len(items) for items in split_map.values())


def launch_detached() -> int:
    ensure_project_directories()
    _TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TRAIN_LOG_PATH, "w", encoding="utf-8") as log_file:
        subprocess.Popen(
            [sys.executable, "run_train.py"],
            creationflags=_DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).resolve().parent,
        )
    print("Đã khởi động training ở chế độ detached.")
    print(f"Log: {_TRAIN_LOG_PATH}")
    print(f"Theo dõi: Get-Content -Tail 20 -Wait {_TRAIN_LOG_PATH}")
    return 0


def run_train_preflight(print_fn=print) -> int:
    ensure_project_directories()
    paths = medical_training_paths()
    audit = audit_medical_raw_dataset(paths)
    train_count = _count_split_items(audit["train_images"])
    val_count = _count_split_items(audit["val_images"])
    test_count = _count_split_items(audit["test_images"])
    missing_classes = audit["missing_classes"]
    ready = train_count > 0 and val_count > 0 and not missing_classes

    print_fn("Medical 7-cancer training preflight")
    print_fn(f"- Dataset root: {paths.dataset_root}")
    print_fn(f"- Model target: {paths.trained_model_path}")
    print_fn(f"- Classes: {len(paths.class_names)}")
    print_fn(f"- Counts: train={train_count}, val={val_count}, test={test_count}")
    print_fn(f"- Missing classes: {len(missing_classes)}")
    if missing_classes:
        print_fn("- Class list: " + ", ".join(missing_classes))
    print_fn(f"- Status: {'sẵn sàng train' if ready else 'chưa đủ dữ liệu medical'}")
    return 0 if ready else 1


def main() -> int:
    ensure_utf8_console()
    args = build_parser().parse_args()
    if getattr(args, "detached", False):
        return launch_detached()
    if getattr(args, "check_only", False):
        return run_train_preflight()

    start = time.perf_counter()
    result = run_full_medical_training_pipeline()
    metrics = result["validation_metrics"]
    print("Medical 7-cancer training complete")
    print(f"- Trained model: {result['trained_model_path']}")
    print(f"- Train/val/test: {result['train_count']}/{result['val_count']}/{result['test_count']}")
    print(f"- Prepare: {result['prepare_seconds']:.2f}s")
    print(f"- Train: {result['train_seconds']:.2f}s")
    print(f"- Validate: {result['validate_seconds']:.2f}s")
    print(f"- Total: {time.perf_counter() - start:.2f}s")
    print(f"- Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"- Validation model: {metrics['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_entrypoint(main))
