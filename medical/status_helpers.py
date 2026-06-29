from __future__ import annotations

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
class TciaCounts:
    target_total: int
    downloaded_total: int
    remaining_to_target: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file())


def count_all_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def skin_dataset_counts(dataset_root: Path) -> DatasetCounts:
    return DatasetCounts(
        raw_images=count_files(dataset_root / "raw" / "images"),
        raw_labels=count_files(dataset_root / "raw" / "labels"),
        train_images=count_files(dataset_root / "processed" / "images" / "train"),
        val_images=count_files(dataset_root / "processed" / "images" / "val"),
        test_images=count_files(dataset_root / "processed" / "images" / "test"),
    )


def tcia_counts(collections_file: str | Path = DEFAULT_TCIA_COLLECTIONS_FILE) -> TciaCounts:
    report = verify_downloads(collections_file)
    return TciaCounts(
        target_total=int(report["target_total"]),
        downloaded_total=int(report["downloaded_total"]),
        remaining_to_target=int(report["remaining_to_target"]),
    )
