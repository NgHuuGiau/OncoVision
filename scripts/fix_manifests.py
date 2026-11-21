"""
Sua volume.json cap goc (organ root) loi thanh manifest hop le.
Sao chep tu volume.json o thuong muc modality (ct/, mri/, ultrasound…) cua cung nhom.
Backup file cu sang .bak truoc khi ghi de.
Chay: python scripts/fix_manifests.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from medical.cancer_model_registry import CANCERS_DIR


def _is_valid_manifest(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    task = str(data.get("task", ""))
    if task not in {"volume_classification", "volume_segmentation", "volume_multitask"}:
        return False
    shape = data.get("input_shape")
    if not (isinstance(shape, list) and len(shape) == 3):
        return False
    if not all(isinstance(v, int) and 0 < v <= 512 for v in shape):
        return False
    labels = [str(v).strip() for v in data.get("class_labels", []) if str(v).strip()]
    if task != "volume_segmentation" and not labels:
        return False
    threshold = float(data.get("mask_threshold", 0.5))
    return 0.0 < threshold < 1.0


def _find_valid_manifest(organ_dir: Path) -> Path | None:
    """Tim volume.json hop le trong cac thuong muc con (modality)."""
    for sub in sorted(organ_dir.iterdir()):
        if not sub.is_dir():
            continue
        for name in ("volume.json", "volume_model.json", f"{organ_dir.name}_volume_model.json"):
            cand = sub / name
            if cand.is_file() and _is_valid_manifest(cand):
                return cand
    return None


def main() -> int:
    fixed = 0
    skipped = 0

    for organ_dir in sorted(CANCERS_DIR.iterdir()):
        if not organ_dir.is_dir():
            continue
        root_manifest = organ_dir / "volume.json"
        if not root_manifest.is_file():
            skipped += 1
            print(f"[{organ_dir.name}] volume.json cap goc: KHONG CO, bo qua")
            continue

        if _is_valid_manifest(root_manifest):
            skipped += 1
            print(f"[{organ_dir.name}] volume.json cap goc: DANG HOP LE, bo qua")
            continue

        # File cu loi -> backup
        bak = organ_dir / "volume.json.invalid.bak"
        shutil.move(str(root_manifest), str(bak))
        print(f"[{organ_dir.name}] volume.json cap goc: LOI -> backup {bak.name}")

        # Tim manifest hop le tu modality subdir
        valid = _find_valid_manifest(organ_dir)
        if valid is None:
            print("  -> KHONG TIM THAY volume.json hop le o thuong muc con, de trong")
            continue

        # Doc, sua target thanh organ key, ghi lai
        data = json.loads(valid.read_text(encoding="utf-8"))
        data["target"] = organ_dir.name.lower()
        root_manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        fixed += 1
        print(f"  -> Ghi {root_manifest.name} tu {valid.relative_to(_project_root)} (target={organ_dir.name})")

    print("-" * 60)
    print(f"Sua: {fixed} file, Bo qua: {skipped} file.")
    print("File cu da duoc backup ra *.invalid.bak.")
    print("Chay lai python scripts/fix_all.py de xac minh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
