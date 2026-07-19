"""Danh gia thuc te modality classifier tren du lieu y khoa that.

Khong can medmnist. Dung model models/pretrained/modality_classifier.pt (resnet18,
8 lop: ct/mri/xray/ultrasound/mammogram/endoscopy/pet_ct/eus) du doan tren tap
anh test cua 7 folder ung thu, bao cao phan bo modality du doan va top-1 accuracy
theo mapping folder -> modality duoc cho la dung nhat.

Mapping folder -> modality la uoc luong (do du lieu goc khong gan nhan modality);
muc tieu la xem model phan biet duoc nhom anh nao, khong phai do chinh xac tuyet doi.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from medical.modality_classifier import load_modality_classifier, predict_modality_from_image

_FOLDER_TO_MODALITY = {
    "Ung thư cổ tử cung": "endoscopy",
    "Ung thư dạ dày": "endoscopy",
    "Ung thư gan": "ultrasound",
    "Ung thư phổi": "xray",
    "Ung thư tuyến tiền liệt": "ultrasound",
    "Ung thư vú": "mammogram",
    "Ung thư đại trực tràng": "endoscopy",
}

_TOPK = 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pretrained/modality_classifier.pt")
    parser.add_argument("--max-per-folder", type=int, default=400)
    parser.add_argument("--root", default="dataset/medical")
    args = parser.parse_args()

    wrapper = load_modality_classifier(args.model, device="cpu")
    labels = list(wrapper.class_labels)

    confusion = collections.Counter()
    top1_correct = 0
    top3_correct = 0
    total = 0
    per_folder = {}

    for folder, modality in _FOLDER_TO_MODALITY.items():
        test_dir = Path(args.root) / folder / "processed" / "images" / "test"
        if not test_dir.exists():
            continue
        images = [p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        images = images[: args.max_per_folder]
        folder_correct_top1 = 0
        folder_correct_top3 = 0
        dist = collections.Counter()
        for img in images:
            try:
                preds = predict_modality_from_image(wrapper, str(img), top_k=_TOPK)
            except Exception:
                continue
            pred_labels = [p["label"] for p in preds]
            dist[pred_labels[0]] += 1
            total += 1
            if pred_labels[0] == modality:
                top1_correct += 1
                folder_correct_top1 += 1
            if modality in pred_labels:
                top3_correct += 1
                folder_correct_top3 += 1
            confusion[(modality, pred_labels[0])] += 1
        per_folder[folder] = {
            "expected": modality,
            "n": len(images),
            "top1_acc": folder_correct_top1 / max(1, len(images)),
            "top3_acc": folder_correct_top3 / max(1, len(images)),
            "distribution": dict(dist),
        }

    print("=" * 60)
    print("DANH GIA MODALITY CLASSIFIER (du lieu that, tap test)")
    print("=" * 60)
    print(f"Tong mau: {total}")
    print(f"Top-1 accuracy (theo mapping uoc luong): {top1_correct / max(1, total):.3f}")
    print(f"Top-3 accuracy: {top3_correct / max(1, total):.3f}")
    print("\nPer-folder:")
    for folder, info in per_folder.items():
        print(f"  {folder} (expect {info['expected']}, n={info['n']}): "
              f"top1={info['top1_acc']:.3f} top3={info['top3_acc']:.3f}")
        top = sorted(info["distribution"].items(), key=lambda kv: -kv[1])[:3]
        print(f"      top preds: {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
