from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CleanupSummary:
    removed_files: int
    removed_dirs: int
    freed_bytes: int
    removed_paths: list[str]


def cleanup_directories(
    directories: Iterable[Path],
    *,
    older_than_days: int | None = None,
) -> CleanupSummary:
    threshold = None if older_than_days is None else datetime.now() - timedelta(days=max(0, older_than_days))
    removed_files = 0
    removed_dirs = 0
    freed_bytes = 0
    removed_paths: list[str] = []

    for directory in directories:
        if not directory.exists():
            continue
        for path in iter_cleanup_candidates(directory):
            if threshold is not None and datetime.fromtimestamp(path.stat().st_mtime) >= threshold:
                continue
            if path.is_file():
                freed_bytes += path.stat().st_size
                path.unlink(missing_ok=True)
                removed_files += 1
                removed_paths.append(str(path))
                continue
            if path.is_dir():
                try:
                    path.rmdir()
                    removed_dirs += 1
                    removed_paths.append(str(path))
                except OSError:
                    continue

    return CleanupSummary(
        removed_files=removed_files,
        removed_dirs=removed_dirs,
        freed_bytes=freed_bytes,
        removed_paths=removed_paths,
    )


def iter_cleanup_candidates(directory: Path) -> list[Path]:
    children = list(directory.rglob("*"))
    children.sort(key=lambda item: len(item.parts), reverse=True)
    return children
