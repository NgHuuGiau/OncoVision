from __future__ import annotations

try:
    from training._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from training.split_dataset import audit_raw_dataset
from utils.file_utils import ensure_project_directories


def main() -> int:
    ensure_project_directories()
    audit = audit_raw_dataset()

    print("=== DATASET RAW AUDIT ===")
    print(f"Tong anh raw       : {audit.raw_image_count}")
    print(f"Anh hop le         : {len(audit.eligible_images)}")
    print(f"Anh thieu label    : {len(audit.missing_labels)}")
    print(f"Label rong         : {len(audit.empty_labels)}")
    print(f"Label loi          : {len(audit.invalid_labels)}")
    print(f"Label mo coi       : {len(audit.orphan_labels)}")

    for image_path in audit.missing_labels[:10]:
        print(f"[MISSING] {image_path.name}")
    for label_path, issue in audit.invalid_labels[:10]:
        print(f"[INVALID] {label_path.name} -> {issue}")
    for label_path in audit.orphan_labels[:10]:
        print(f"[ORPHAN]  {label_path.name}")

    if audit.raw_image_count == 0:
        print("Chua co du lieu raw.")
        return 1
    if audit.missing_labels or audit.invalid_labels:
        print("Dataset chua sach. Hay sua label truoc khi train.")
        return 1
    print("Dataset hop le de train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
