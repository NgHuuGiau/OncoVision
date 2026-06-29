from __future__ import annotations

from pathlib import Path


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file())


def count_project_files(*paths: Path) -> tuple[int, ...]:
    return tuple(count_files(path) for path in paths)
