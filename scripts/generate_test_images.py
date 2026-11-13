#!/usr/bin/env python3
"""
Tao anh test cho cac loai ung thu chua co dataset.

Dung de test pipeline, khong dung de huan luyen.
Moi loai se tao 25 anh moi class voi mau ngau nhien + text label.

Su dung:
    python scripts/generate_test_images.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Danh sach cac loai ung thu chua co dataset
CANCERS_TO_GENERATE = [
    {
        "cancer_type": "breast",
        "classes": ["benign", "malignant", "normal"],
        "modality": "ultrasound",
    },
    {
        "cancer_type": "kidney",
        "classes": ["cyst", "tumor", "normal"],
        "modality": "ct",
    },
    {
        "cancer_type": "liver",
        "classes": ["normal", "tumor"],
        "modality": "ct",
    },
    {
        "cancer_type": "stomach",
        "classes": ["carcinoma", "normal", "ulcer", "erosion"],
        "modality": "endoscopy",
    },
    {
        "cancer_type": "colorectal",
        "classes": ["polyp", "normal"],
        "modality": "colonoscopy",
    },
    {
        "cancer_type": "cervical",
        "classes": ["normal", "dysplastic", "squamous_cell_carcinoma"],
        "modality": "colposcopy",
    },
    {
        "cancer_type": "prostate",
        "classes": ["malignant", "benign"],
        "modality": "mri",
    },
    {
        "cancer_type": "pancreas",
        "classes": ["normal", "pdac", "neuroendocrine"],
        "modality": "ct",
    },
]

# Mau sac theo modality
MODALITY_COLORS = {
    "ct": (80, 80, 80),           # Xam
    "mri": (60, 60, 100),         # Xam xanh
    "ultrasound": (50, 50, 50),   # Xam toi
    "endoscopy": (180, 50, 50),   # Do
    "colonoscopy": (150, 40, 40), # Do am
    "colposcopy": (100, 150, 100),# Xanh la
}


def generate_test_image(class_name: str, modality: str, idx: int) -> Image.Image:
    """Tao 1 anh test voi mau ngau nhien + text label."""
    base_color = MODALITY_COLORS.get(modality, (100, 100, 100))

    # Mau ngau nhien gan voi base color
    r = base_color[0] + random.randint(-30, 30)
    g = base_color[1] + random.randint(-30, 30)
    b = base_color[2] + random.randint(-30, 30)
    r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))

    img = Image.new("RGB", (224, 224), (r, g, b))
    draw = ImageDraw.Draw(img)

    # Them noise
    for _ in range(500):
        x = random.randint(0, 223)
        y = random.randint(0, 223)
        noise = random.randint(-20, 20)
        nr = max(0, min(255, r + noise))
        ng = max(0, min(255, g + noise))
        nb = max(0, min(255, b + noise))
        img.putpixel((x, y), (nr, ng, nb))

    # Ve text label
    draw.text((10, 10), f"{class_name}_{idx}", fill=(255, 255, 255))

    return img


def main() -> None:
    if not HAS_PIL:
        print("❌ Thiếu Pillow. Cai dat: pip install pillow")
        sys.exit(1)

    base_dir = Path("data/datasets")
    base_dir.mkdir(parents=True, exist_ok=True)

    images_per_class = 25
    total_generated = 0

    print("=" * 60)
    print("🏥 OncoVision — Tao Anh Test")
    print("=" * 60)
    print(f"📁 Dau ra: {base_dir.resolve()}")
    print(f"🔢 Anh moi class: {images_per_class}")
    print()

    for cancer_info in CANCERS_TO_GENERATE:
        cancer_type = cancer_info["cancer_type"]
        classes = cancer_info["classes"]
        modality = cancer_info["modality"]

        # Kiem tra da co chua
        cancer_dir = base_dir / cancer_type
        if cancer_dir.exists():
            existing = sum(1 for _ in cancer_dir.rglob("*.png"))
            if existing > 0:
                print(f"  ⏭️  {cancer_type}: Da co {existing} anh, bo qua")
                continue

        print(f"📦 {cancer_type.upper()} ({modality})")
        cancer_dir.mkdir(parents=True, exist_ok=True)

        for class_name in classes:
            class_dir = cancer_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for idx in range(images_per_class):
                img = generate_test_image(class_name, modality, idx)
                img_path = class_dir / f"test_{idx}.png"
                img.save(str(img_path))
                total_generated += 1

            print(f"   ✅ {class_name}: {images_per_class} ảnh")

        print(f"   📊 Tong: {len(classes) * images_per_class} ảnh")
        print()

    # Cap nhat manifest neu co
    manifest_path = base_dir / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        # Them cac loai da tao vao manifest
        for cancer_info in CANCERS_TO_GENERATE:
            cancer_type = cancer_info["cancer_type"]
            cancer_dir = base_dir / cancer_type
            if cancer_dir.exists():
                entry = {
                    "cancer_type": cancer_type,
                    "repo_id": "generated_test",
                    "total_images": len(cancer_info["classes"]) * images_per_class,
                    "per_class": {cls: images_per_class for cls in cancer_info["classes"]},
                    "status": "generated",
                    "error": None,
                }
                # Cap nhat neu da co, them neu chua co
                found = False
                for i, existing in enumerate(manifest.get("cancers", [])):
                    if existing["cancer_type"] == cancer_type:
                        manifest["cancers"][i] = entry
                        found = True
                        break
                if not found:
                    manifest.setdefault("cancers", []).append(entry)
        # Tinh tong
        manifest["total_images"] = sum(c["total_images"] for c in manifest.get("cancers", []))
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"✅ Da tao {total_generated} anh test")
    print("=" * 60)


if __name__ == "__main__":
    main()
