from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Giữ đồng bộ với MEDICAL_IMAGE_EXTENSIONS trong medical/classifier.py
# (không import trực tiếp để tránh kéo torch vào module nhẹ này).
_MEDICAL_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


@lru_cache(maxsize=512)
def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with os.scandir(path) as entries:
            return sum(1 for item in entries if item.is_file())
    except OSError:
        return 0


def count_medical_images(directory: Path) -> int:
    # ponytail: os.walk không sort — đếm 240k ảnh nhanh gấp ~4 lần rglob+sorted;
    # nếu cần chính xác thời gian thực thì cache theo (path, mtime).
    if not directory.exists():
        return 0
    total = 0
    try:
        for _root, _dirs, files in os.walk(directory):
            total += sum(1 for name in files if Path(name).suffix.lower() in _MEDICAL_IMAGE_SUFFIXES)
    except OSError:
        return total
    return total
