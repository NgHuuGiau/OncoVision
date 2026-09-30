from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=512)
def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with os.scandir(path) as entries:
            return sum(1 for item in entries if item.is_file())
    except OSError:
        return 0
