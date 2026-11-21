"""
Audit trạng thái model của 11 nhóm ung thư trong models/pretrained/cancers/.
Chỉ ĐỌC (không sửa file model) de tranh hoi hieu inference.
Chay: python scripts/audit_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from medical.cancer_model_registry import list_cancer_models


def main() -> int:
    models = list_cancer_models()
    print("=" * 96)
    print("AUDIT MODEL UNG THU - OncoVision")
    print("=" * 96)
    print(
        f"{'Nhom':<14} {'Class':<9} {'Detect':<9} {'Volume':<9} {'Clinical':<11} {'Runtime':<9} {'Van de'}"
    )
    print("-" * 96)

    all_ready = True
    for m in models:
        key = m["key"]
        clf = "OK" if m["classification_ready"] else "X"
        det = "OK" if m["localization_ready"] else "X"
        vol = "OK" if m["volume_ready"] else "X"
        clin = "OK" if m["clinical_validation_ready"] else "X"
        rt = "OK" if m["runtime_ready"] else "X"

        problems: list[str] = []
        if not m["artifact_ready"]:
            problems.append("thua file model")
        note = m.get("note", "")
        if not m["runtime_ready"]:
            all_ready = False
            if "classifier.pt" in note and "khong chay duoc" in note:
                problems.append("classifier loi runtime")
            if "detector.pt" in note and "khong chay duoc" in note:
                problems.append("detector loi runtime")
            if not problems:
                problems.append("chua chay duoc runtime")

        prob_str = "; ".join(problems) if problems else "-"
        print(f"{key:<14} {clf:<9} {det:<9} {vol:<9} {clin:<11} {rt:<9} {prob_str}")

    print("-" * 96)
    print(f"Tong: {len(models)} nhom ung thu.")
    if all_ready:
        print("KET QUAN: Tat ca 11 nhom da co runtime san sang.")
    else:
        print("KET QUAN: CON NHOM CHUA SAN SANG - xem cot 'Van de' tren.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
