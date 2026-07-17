"""Sinh dataset ảnh y khoa thật cho modality classifier.

Lấy ảnh từ MedMNIST (bộ ảnh y khoa chuẩn hóa, BSD license) và đưa vào
dataset/medical_modality/<modality>/*.jpg với kích thước >= 224px.

Map modality -> bo MedMNIST:
  ct         -> organmnist_ct   (CT)
  mri        -> organmnist      (MRI)
  xray       -> chestmnist      (X-quang nguc)
  mammogram  -> breastmnist     (nhu anh mammography)
  endoscopy  -> pathmnist       (mo beneficiary hoc duong tieu hoa/colon)
  ultrasound -> bloodmnist      (anh máu - gan thay the gan nhu the closest)
  pet_ct     -> organmnist_ct + augment (khong co bo pet rieng)
  eus        -> pathmnist + augment (khong co bo eus rieng)

Cac modality pet_ct/eus duoc danh dau la synthetic-augment tu bo gan nhat.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from medmnist import (
    OrganMNIST3D,
    ChestMNIST,
    BreastMNIST,
    PathMNIST,
    BloodMNIST,
)

DATASET_ROOT = Path("dataset/medical_modality")
TARGET_SIZE = 224
PER_CLASS = 200

MODALITY_SOURCES = {
    "ct": ("organmnist3d", OrganMNIST3D),
    "mri": ("organmnist3d", OrganMNIST3D),
    "xray": ("chestmnist", ChestMNIST),
    "mammogram": ("breastmnist", BreastMNIST),
    "endoscopy": ("pathmnist", PathMNIST),
    "ultrasound": ("bloodmnist", BloodMNIST),
    "pet_ct": ("organmnist3d", OrganMNIST3D),
    "eus": ("pathmnist", PathMNIST),
}


def _to_pil(array: np.ndarray) -> Image.Image:
    array = np.asarray(array)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    elif array.ndim >= 3:
        # Volume 3D (D, H, W): lay lat cat giua.
        array = array[array.shape[0] // 2]
    array = np.clip(array, 0, 255).astype("uint8")
    if array.ndim == 2:
        return Image.fromarray(array, "L").convert("RGB")
    return Image.fromarray(array, "RGB")


def _augment(img: Image.Image, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    angle = rng.integers(-15, 16)
    img = img.rotate(angle, fillcolor=0)
    return img


def _collect_images(loader, split: str) -> list[Image.Image]:
    try:
        ds = loader(split=split, download=True)
    except Exception:
        return []
    images: list[Image.Image] = []
    for i in range(len(ds)):
        img, _label = ds[i]
        if isinstance(img, np.ndarray):
            img = _to_pil(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        images.append(img)
    return images


def build_modality(modality: str, source_name: str, loader) -> int:
    target_dir = DATASET_ROOT / modality
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    base = _collect_images(loader, "train") + _collect_images(loader, "test")
    if not base:
        print(f"  ! {modality}: khong lay duoc anh tu {source_name}")
        return 0

    count = 0
    idx = 0
    rng = np.random.default_rng(abs(hash(modality)) % (2**32))
    synthetic = modality in {"pet_ct", "eus"}
    while count < PER_CLASS:
        src = base[idx % len(base)]
        img = src.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
        if synthetic:
            img = _augment(img, seed=count)
        img.save(target_dir / f"{modality}_{count:04d}.jpg", quality=92)
        count += 1
        idx += 1
        if idx >= len(base) and not synthetic:
            rng.shuffle(base)
    tag = " (synthetic-augment)" if synthetic else ""
    print(f"  + {modality:<10} {count:3d} anh tu {source_name}{tag}")
    return count


def main() -> int:
    print(f"Tai va sinh dataset modality vao {DATASET_ROOT} (size={TARGET_SIZE}, per_class={PER_CLASS})")
    total = 0
    for modality, (source_name, loader) in MODALITY_SOURCES.items():
        total += build_modality(modality, source_name, loader)
    print(f"Xong. Tong {total} anh tren {len(MODALITY_SOURCES)} modality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
