from __future__ import annotations

import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
DEFAULT_SEED = 42


def iter_matching_files(root: Path, *, suffixes: set[str]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def is_valid_yolo_label_line(line: str) -> bool:
    parts = line.strip().split()
    if not parts:
        return True
    if len(parts) != 5:
        return False
    try:
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return False
    if class_id < 0:
        return False
    return all(0.0 <= value <= 1.0 for value in coords)


def read_yolo_label_status(path: Path) -> tuple[bool, bool]:
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return True, True
    return all(is_valid_yolo_label_line(line) for line in non_empty), False


def reset_processed_dirs(
    *,
    processed_images_dir: Path,
    processed_labels_dir: Path,
    splits: tuple[tuple[str, float], ...] = DEFAULT_SPLITS,
) -> None:
    for root in (processed_images_dir, processed_labels_dir):
        if root.exists():
            shutil.rmtree(root)
    for split_name, _ratio in splits:
        (processed_images_dir / split_name).mkdir(parents=True, exist_ok=True)
        (processed_labels_dir / split_name).mkdir(parents=True, exist_ok=True)


def split_items(
    items: list[tuple[Path, Path]],
    *,
    splits: tuple[tuple[str, float], ...] = DEFAULT_SPLITS,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[tuple[Path, Path]]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    split_ratios = dict(splits)
    train_end = int(total * split_ratios["train"])
    val_end = train_end + int(total * split_ratios["val"])
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def copy_split(
    *,
    split_name: str,
    items: list[tuple[Path, Path]],
    processed_images_dir: Path,
    processed_labels_dir: Path,
) -> None:
    for image_path, label_path in items:
        shutil.copy2(image_path, processed_images_dir / split_name / image_path.name)
        shutil.copy2(label_path, processed_labels_dir / split_name / label_path.name)
