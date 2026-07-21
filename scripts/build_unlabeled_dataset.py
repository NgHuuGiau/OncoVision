"""Tải ảnh y khoa thật (MedMNIST test split) vào dataset/medical/unlabeled/.

Mục đích: cung cấp ảnh chưa dán nhãn thật để active-learning (run_medical.py
active-learning) gợi ý ảnh nào cần dán nhãn thêm. Ảnh lấy từ tập test của
MedMNIST (BSD license) - ảnh y khoa thực tế, đã chuẩn hóa.

Ảnh được resize 224x224 RGB và lưu với tên không nhãn để tránh nhầm lẫn
với dữ liệu đã dán nhãn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from medmnist import BloodMNIST, ChestMNIST, OrganMNIST3D, PathMNIST

OUTPUT_DIR = Path("dataset/medical/unlabeled")
TARGET_SIZE = 224
PER_SOURCE = 40


def _to_pil(array: np.ndarray) -> Image.Image:
    array = np.asarray(array)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    elif array.ndim >= 3:
        array = array[array.shape[0] // 2]
    array = np.clip(array, 0, 255).astype("uint8")
    if array.ndim == 2:
        return Image.fromarray(array, "L").convert("RGB")
    return Image.fromarray(array, "RGB")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = [
        ("blood", BloodMNIST),
        ("path", PathMNIST),
        ("organ", OrganMNIST3D),
        ("chest", ChestMNIST),
    ]
    total = 0
    for tag, loader in sources:
        try:
            ds = loader(split="test", download=True)
        except Exception as exc:
            print(f"! bo qua {tag}: {exc}")
            continue
        count = 0
        for i in range(min(len(ds), PER_SOURCE)):
            img, _ = ds[i]
            if isinstance(img, np.ndarray):
                img = _to_pil(img)
            elif not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
            out = OUTPUT_DIR / f"unlabeled_{tag}_{i:04d}.jpg"
            img.save(out, quality=92)
            count += 1
        total += count
        print(f"+ {tag:<6} {count} ảnh -> {OUTPUT_DIR}")
    print(f"Xong. Tổng {total} ảnh chưa dán nhãn tại {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
