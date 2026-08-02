from __future__ import annotations

"""Augment ảnh thật bù đủ mục tiêu cho từng loại ung thư (mặc định 30.000/loại).

Chỉ thêm ảnh augment vào train/, giữ val/ test/ là ảnh thật 100%.
"""

import io
import random
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WORKSPACE = Path(__file__).resolve().parent.parent
MEDICAL_ROOT = WORKSPACE / "dataset" / "medical"
IMAGE_SIZE = 512
TARGET = 30000
SEED = 42


def _augment(im: Image.Image, rng: random.Random) -> Image.Image:
    out = im
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.3:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
    if rng.random() < 0.7:
        ang = rng.uniform(-15, 15)
        out = out.rotate(ang, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    if rng.random() < 0.5:
        factor = rng.uniform(0.85, 1.15)
        out = ImageEnhance.Brightness(out).enhance(factor)
    if rng.random() < 0.5:
        factor = rng.uniform(0.9, 1.15)
        out = ImageEnhance.Contrast(out).enhance(factor)
    if rng.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))
    if rng.random() < 0.5:
        scale = rng.uniform(0.9, 1.1)
        nw, nh = int(out.width * scale), int(out.height * scale)
        out = out.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        canvas.paste(out, ((IMAGE_SIZE - nw) // 2, (IMAGE_SIZE - nh) // 2))
        out = canvas
    return out.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)


def _split_count(total: int) -> tuple[int, int, int]:
    n_test = max(1, round(total * 0.10))
    n_val = max(1, round(total * 0.15))
    n_train = max(1, total - n_test - n_val)
    return n_train, n_val, n_test


def balance_class(label: str, target: int = TARGET) -> int:
    root = MEDICAL_ROOT / label / "processed" / "images"
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"
    if not train_dir.exists():
        return 0

    train_imgs = sorted(train_dir.glob("*"))
    val_imgs = sorted(val_dir.glob("*")) if val_dir.exists() else []
    test_imgs = sorted(test_dir.glob("*")) if test_dir.exists() else []
    real_total = len(train_imgs) + len(val_imgs) + len(test_imgs)
    if real_total >= target:
        print(f"  {label}: đã đủ {real_total}/{target}", flush=True)
        return real_total

    # Số ảnh cần thêm vào train để tổng = target (val/test giữ ảnh thật)
    need = target - real_total
    if not train_imgs:
        return real_total

    rng = random.Random(SEED)
    added = 0
    idx = 0
    while added < need:
        src = train_imgs[idx % len(train_imgs)]
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                aug = _augment(im, rng)
                dest = train_dir / f"{src.stem}_aug{added:04d}.jpg"
                aug.save(dest, format="JPEG", quality=95)
                added += 1
        except Exception:
            pass
        idx += 1

    total = real_total + added
    print(f"  {label}: {real_total} thật + {added} augment = {total}/{target}", flush=True)
    return total


def main() -> None:
    labels = [d.name for d in sorted(MEDICAL_ROOT.iterdir()) if d.is_dir()]
    if len(sys.argv) > 1:
        labels = [a for a in sys.argv[1:] if (MEDICAL_ROOT / a).is_dir()]
    print(f"Augment cho {len(labels)} loại ung thư (mục tiêu {TARGET}/loại)", flush=True)
    for label in labels:
        try:
            balance_class(label)
        except Exception as exc:
            print(f"  FAIL {label}: {exc}", flush=True)
    print("XONG", flush=True)


if __name__ == "__main__":
    main()
