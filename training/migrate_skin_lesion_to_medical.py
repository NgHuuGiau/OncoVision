from __future__ import annotations

import shutil
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from medical.dataset import create_default_skin_cancer_dataset_config, ensure_medical_dataset_structure


SOURCE_ROOT = Path("dataset/medical/medical_skin_lesion")
TARGET_ROOT = Path("dataset/medical/skin_lesion")


def _copy_tree(src: Path, dst: Path) -> int:
    count = 0
    if not src.exists():
        return count
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if path.is_file():
            target = dst / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            count += 1
    return count


def migrate_skin_lesion_dataset() -> dict[str, int]:
    ensure_medical_dataset_structure(create_default_skin_cancer_dataset_config(TARGET_ROOT))
    counts = {
        "raw_images": _copy_tree(SOURCE_ROOT / "raw" / "images", TARGET_ROOT / "raw" / "images"),
        "raw_labels": _copy_tree(SOURCE_ROOT / "raw" / "labels", TARGET_ROOT / "raw" / "labels"),
        "processed_images_train": _copy_tree(SOURCE_ROOT / "processed" / "images" / "train", TARGET_ROOT / "processed" / "images" / "train"),
        "processed_images_val": _copy_tree(SOURCE_ROOT / "processed" / "images" / "val", TARGET_ROOT / "processed" / "images" / "val"),
        "processed_images_test": _copy_tree(SOURCE_ROOT / "processed" / "images" / "test", TARGET_ROOT / "processed" / "images" / "test"),
        "processed_labels_train": _copy_tree(SOURCE_ROOT / "processed" / "labels" / "train", TARGET_ROOT / "processed" / "labels" / "train"),
        "processed_labels_val": _copy_tree(SOURCE_ROOT / "processed" / "labels" / "val", TARGET_ROOT / "processed" / "labels" / "val"),
        "processed_labels_test": _copy_tree(SOURCE_ROOT / "processed" / "labels" / "test", TARGET_ROOT / "processed" / "labels" / "test"),
        "metadata": _copy_tree(SOURCE_ROOT / "metadata", TARGET_ROOT / "metadata"),
        "reports": _copy_tree(SOURCE_ROOT / "reports", TARGET_ROOT / "reports"),
    }
    return counts


def main() -> None:
    counts = migrate_skin_lesion_dataset()
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"Target: {TARGET_ROOT}")
    print(f"Source: {SOURCE_ROOT}")


if __name__ == "__main__":
    main()
