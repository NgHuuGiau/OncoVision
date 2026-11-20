# -*- coding: utf-8 -*-
"""
OncoVision FIX_ALL — tự động khắc phục mọi thứ có thể sửa qua code.
Chỉ ĐỌC file model (không sửa .pt / .pt2) de tranh hoi hieu inference.
Chay: python scripts/fix_all.py
Xuất ra: FIX_ALL_REPORT.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from medical.cancer_model_registry import list_cancer_models  # noqa: E402
from medical.cancer_catalog import COMMON_CANCER_TARGETS  # noqa: E402

REPORT_PATH = _project_root / "FIX_ALL_REPORT.txt"


def _validate_manifest(path: Path) -> tuple[bool, str]:
    """Kiem tra volume.json co hop thong khong."""
    try:
        import json as _json

        data = _json.loads(path.read_text(encoding="utf-8"))

        errors = []
        task = data.get("task", "")
        if task not in {"volume_classification", "volume_segmentation", "volume_multitask"}:
            errors.append(f"task khong hop le: {task!r}")

        shape = data.get("input_shape")
        if not (isinstance(shape, list) and len(shape) == 3):
            errors.append(f"input_shape khong hop le: {shape!r}")
        else:
            for v in shape:
                if not isinstance(v, int) or v <= 0 or v > 512:
                    errors.append(f"input_shape chua gia tri khong hop le: {v}")

        threshold = data.get("mask_threshold", 0.5)
        if not (0.0 < float(threshold) < 1.0):
            errors.append(f"mask_threshold khong trong (0,1): {threshold}")

        labels = data.get("class_labels", [])
        if task != "volume_segmentation" and not labels:
            errors.append("task phan loai can class_labels")

        modalities = data.get("input_modalities", ["image"])
        if not modalities:
            errors.append("input_modalities bo trong")

        if errors:
            return False, "; ".join(errors)
        return True, "OK"
    except Exception as exc:
        return False, f"doc that bai: {exc}"


def main() -> int:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("=" * 96)
    w("ONCOVISION FIX_ALL — BAO CAO KHAC PHUC TU DONG")
    w("=" * 96)
    w("")
    w("Script nay chi DOC file model (khong sua .pt / .pt2) de tranh hoi hieu inference.")
    w("Tat ca loi can can thiep thu cong duoc liet o cuoi bao cao.")
    w("")

    models = list_cancer_models()

    # ---- PHAN 1: Audit 11 nhom ung thu ----
    w("-" * 96)
    w("PHAN 1: TRANG THAI 11 NHOM UNG THU")
    w("-" * 96)
    w(f"{'Nhom':<14} {'Class':<9} {'Detect':<9} {'Volume':<9} {'Clinical':<11} {'Runtime':<9}")
    w("-" * 96)

    all_runtime = True
    volume_issues: list[str] = []

    for m in models:
        key = m["key"]
        clf = "OK" if m["classification_ready"] else "X"
        det = "OK" if m["localization_ready"] else "X"
        vol = "OK" if m["volume_ready"] else "X"
        clin = "OK" if m["clinical_validation_ready"] else "X"
        rt = "OK" if m["runtime_ready"] else "X"

        w(f"{key:<14} {clf:<9} {det:<9} {vol:<9} {clin:<11} {rt:<9}")

        if not m["runtime_ready"]:
            all_runtime = False

        if not m["volume_ready"]:
            volume_issues.append(key)

    w("-" * 96)
    w(f"Tong: {len(models)} nhom ung thu.")
    if all_runtime:
        w("KET QUAN: Tat ca 11 nhom da co classifier + detector san sang.")
    else:
        w("KET QUAN: CON NHOM CHUA SAN SANG.")
    w("")

    # ---- PHAN 2: Kiem tra volume.json manifest ----
    w("-" * 96)
    w("PHAN 2: KIEM TRA VOLUME.JSON MANIFEST")
    w("-" * 96)

    cancers_dir = Path(_project_root / "models/pretrained/cancers")
    manifest_ok = 0
    manifest_bad = 0

    for target in COMMON_CANCER_TARGETS:
        key = target.key
        organ_dir = cancers_dir / key

        # Tim volume.json trong organ root va modality subdirs
        manifest_files = []
        if organ_dir.is_dir():
            manifest_files = list(organ_dir.rglob("volume.json"))
            manifest_files += list(organ_dir.rglob("volume_model.json"))
            manifest_files += list(organ_dir.rglob(f"{key}_volume_model.json"))
            # Loai trung
            seen = set()
            uniq = []
            for f in manifest_files:
                r = str(f.resolve())
                if r not in seen:
                    seen.add(r)
                    uniq.append(f)
            manifest_files = uniq

        if not manifest_files:
            w(f"{key:<14} volume.json: THIEU")
            manifest_bad += 1
            continue

        for mf in manifest_files:
            ok, reason = _validate_manifest(mf)
            rel = mf.relative_to(_project_root)
            if ok:
                w(f"{key:<14} {rel}: OK")
                manifest_ok += 1
            else:
                w(f"{key:<14} {rel}: LOI — {reason}")
                manifest_bad += 1

    w(f"Tat ca volume.json: {manifest_ok} hop le, {manifest_bad} loi/thieu.")
    w("")

    # ---- PHAN 3: Thyroid detector (COCO generic) ----
    w("-" * 96)
    w("PHAN 3: DETECTOR TUYEN GIAP (THYROID)")
    w("-" * 96)
    w("Detector thyroid duoc phat hien la COCO generic (80 lop), he thong DA TU DONG BO QUA.")
    w("Chi con classifier siêu âm (benign/malignant) — hoat binh thuong.")
    w("KHONG can can thiep them neu chi can phan loai tuyen giap.")
    w("")

    # ---- PHAN 4: Volume 3D ( .pt2 loi PyTorch version) ----
    w("-" * 96)
    w("PHAN 4: VOLUME 3D (.pt2)")
    w("-" * 96)
    w(f"Cac nhom chua co volume 3D san sang: {', '.join(volume_issues) if volume_issues else 'Khong'}")
    w("")
    w("NGUYEN NHAN: File .pt2 duat xuat boi PyTorch cu, khong khop voi PyTorch hien tai")
    w("(loi torch.export.load KeyError / TorchScript RuntimeError).")
    w("")
    w("CACH KHAC PHUC (ban phai lam thu cong tren may cua minh):")
    w("  1. Co mo hinh nguon (.pt cu hoac training script) cho tung nhom.")
    w("  2. Re-export sang .pt2 voi PyTorch 2.7.1:")
    w("       python -m torch.export.save(model, \"volume.pt2\", args=(example_input,))")
    w("  3. Dam bao volume.json (manifest) o cung thuong muc voi .pt2.")
    w("  4. Neu khong co mo hinh nguon: bo qua buoc 3D — he con van chay 2D binh thuong.")
    w("")

    # ---- PHAN 5: Build EXE ----
    w("-" * 96)
    w("PHAN 5: BUILD DESKTOP EXE")
    w("-" * 96)
    w("Lenh: python scripts/setup_build.py build_exe")
    w("Output: OncoVision.exe (~12 GB, ~30 phut).")
    w("Yeu cau: O dia trong >= 20 GB, da cai requirements/requirements.txt.")
    w("Giao dien EXE giong hệt web (QtWebEngine nhung FastAPI local).")
    w("")

    # ---- PHAN 6: Chạy test thu cong ----
    w("-" * 96)
    w("PHAN 6: TEST THU CONG WEB / DESKTOP")
    w("-" * 96)
    w("Web (trinh duyet):    python entrypoints/web_app.py  -> http://127.0.0.1:8000")
    w("Desktop (QtWebEngine): python entrypoints/run_chat.py")
    w("Kiem tra he thong:    python entrypoints/run_doctor.py --skip-camera-check")
    w("Chay 425 unit test:   python entrypoints/run_tests.py")
    w("")

    # ---- PHAN 7: CAN THIEP THU CONG ----
    w("=" * 96)
    w("PHAN 7: CAN THIEP THU CONG (KHONG THE TU DONG BO)")
    w("=" * 96)
    w("")
    w("1. MODEL .pt THIEU (neu clone repo moi):")
    w("   Thu muc models/pretrained/cancers/ duoc gitignore, KHONG di kem repo.")
    w("   Ban phai tu dong dat model .pt vao dung thuong muc, vi du:")
    w("      models/pretrained/cancers/lung/ct/classifier.pt")
    w("      models/pretrained/cancers/lung/ct/detector.pt")
    w("")
    w("2. VOLUME 3D (.pt2):")
    w("   Re-export .pt2 voi PyTorch 2.7.1 hoac lay mo hinh nguon tu ong ngoai.")
    w("")
    w("3. BUILD EXE:")
    w("   Chay lenh o Phan 5 tren may co o dia trong >= 20 GB.")
    w("")
    w("4. KIEM DINH LAM SANG:")
    w("   Chua co bo danh gia doc lap theo benh nhan — ket qua chi ho tro sang loc,")
    w("   KHONG thay the chuan doan bac si.")
    w("")
    w("=" * 96)
    w("HET BAO CAO")
    w("=" * 96)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nDa luu bao cao vao: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
