from __future__ import annotations

from functools import lru_cache
import os
from dataclasses import dataclass
from pathlib import Path

from training.tcia_downloader import verify_downloads


DEFAULT_TCIA_COLLECTIONS_FILE = Path("training/tcia_collections_5.json")


@dataclass(frozen=True)
class DatasetCounts:
    raw_images: int
    raw_labels: int
    train_images: int
    val_images: int
    test_images: int = 0

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)


@dataclass(frozen=True)
class CancerDownloadCounts:
    target_total: int
    downloaded_total: int
    remaining_to_target: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)


@lru_cache(maxsize=512)
def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with os.scandir(path) as entries:
            return sum(1 for item in entries if item.is_file())
    except OSError:
        return 0


@lru_cache(maxsize=128)
def count_all_files(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    try:
        with os.scandir(path) as entries:
            for item in entries:
                if item.is_file(follow_symlinks=False):
                    total += 1
                elif item.is_dir(follow_symlinks=False):
                    total += count_all_files(Path(item.path))
    except OSError:
        return 0
    return total


def skin_dataset_counts(dataset_root: Path) -> DatasetCounts:
    return DatasetCounts(
        raw_images=count_files(dataset_root / "raw" / "images"),
        raw_labels=count_files(dataset_root / "raw" / "labels"),
        train_images=count_files(dataset_root / "processed" / "images" / "train"),
        val_images=count_files(dataset_root / "processed" / "images" / "val"),
        test_images=count_files(dataset_root / "processed" / "images" / "test"),
    )


def cancer_download_counts(collections_file: str | Path = DEFAULT_TCIA_COLLECTIONS_FILE) -> CancerDownloadCounts:
    report = verify_downloads(collections_file)
    return CancerDownloadCounts(
        target_total=int(report["target_total"]),
        downloaded_total=int(report["downloaded_total"]),
        remaining_to_target=int(report["remaining_to_target"]),
    )


# Backward-compatible aliases for internal callers.
TciaCounts = CancerDownloadCounts
tcia_counts = cancer_download_counts
