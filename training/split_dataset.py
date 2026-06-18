from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from training.dataset_ops import (
    DEFAULT_SEED,
    DEFAULT_SPLITS,
    IMAGE_EXTENSIONS,
    copy_split,
    is_valid_yolo_label_line,
    iter_matching_files,
    reset_processed_dirs,
    split_items,
)
from training.terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section
from utils.file_utils import ensure_project_directories


RAW_IMAGES_DIR = Path("dataset/raw/images")
RAW_LABELS_DIR = Path("dataset/raw/labels")
PROCESSED_IMAGES_DIR = Path("dataset/processed/images")
PROCESSED_LABELS_DIR = Path("dataset/processed/labels")
SPLITS = DEFAULT_SPLITS
SEED = DEFAULT_SEED
INVALID_YOLO_LABEL_REASON = "Định dạng YOLO không hợp lệ"


@dataclass
class RawDatasetAudit:
    images: list[Path]
    labels: list[Path]
    eligible_pairs: list[tuple[Path, Path]]
    missing_labels: list[Path]
    empty_labels: list[Path]
    invalid_labels: list[tuple[Path, str]]
    orphan_labels: list[Path]

    @property
    def raw_image_count(self) -> int:
        return len(self.images)

    @property
    def eligible(self) -> list[tuple[Path, Path]]:
        return self.eligible_pairs

    @property
    def eligible_images(self) -> list[Path]:
        return [image_path for image_path, _ in self.eligible_pairs]

    def __getitem__(self, key: str):
        return getattr(self, "eligible_pairs" if key == "eligible" else key)


def _label_is_valid(path: Path) -> tuple[bool, bool]:
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return True, True
    return all(is_valid_yolo_label_line(line) for line in non_empty), False


def audit_raw_dataset() -> RawDatasetAudit:
    ensure_project_directories()
    images = iter_matching_files(RAW_IMAGES_DIR, suffixes=IMAGE_EXTENSIONS)
    labels = iter_matching_files(RAW_LABELS_DIR, suffixes={".txt"})
    label_by_stem = {path.stem: path for path in labels}
    image_stems = {path.stem for path in images}

    eligible: list[tuple[Path, Path]] = []
    missing_labels: list[Path] = []
    empty_labels: list[Path] = []
    invalid_labels: list[tuple[Path, str]] = []

    for image_path in images:
        label_path = label_by_stem.get(image_path.stem)
        if label_path is None:
            missing_labels.append(image_path)
            continue
        is_valid, is_empty = _label_is_valid(label_path)
        if not is_valid:
            invalid_labels.append((label_path, INVALID_YOLO_LABEL_REASON))
            continue
        if is_empty:
            empty_labels.append(label_path)
        eligible.append((image_path, label_path))

    orphan_labels = [path for path in labels if path.stem not in image_stems]
    return RawDatasetAudit(
        images=images,
        labels=labels,
        eligible_pairs=eligible,
        missing_labels=missing_labels,
        empty_labels=empty_labels,
        invalid_labels=invalid_labels,
        orphan_labels=orphan_labels,
    )


def _reset_processed_dirs() -> None:
    reset_processed_dirs(
        processed_images_dir=PROCESSED_IMAGES_DIR,
        processed_labels_dir=PROCESSED_LABELS_DIR,
        splits=SPLITS,
    )


def _split_items(items: list[tuple[Path, Path]]) -> dict[str, list[tuple[Path, Path]]]:
    return split_items(items, splits=SPLITS, seed=SEED)


def _copy_split(split_name: str, items: list[tuple[Path, Path]]) -> None:
    copy_split(
        split_name=split_name,
        items=items,
        processed_images_dir=PROCESSED_IMAGES_DIR,
        processed_labels_dir=PROCESSED_LABELS_DIR,
    )


def main() -> None:
    ensure_project_directories()
    report = audit_raw_dataset()
    total_images = report.raw_image_count
    eligible = report.eligible

    if total_images == 0:
        for item in header("YOLO DATASET :: KHÔNG CÓ DỮ LIỆU ĐỂ SPLIT", color=RED):
            print(item)
        print(section("LÝ DO", RED))
        print(row("Lý do không chạy", "Không có ảnh trong dataset/raw/images", RED))
        print(line(rule("-"), CYAN))
        print(section("CẦN LÀM", GREEN))
        print(row("Bước 1", "Bỏ ảnh vào dataset/raw/images", YELLOW))
        print(row("Bước 2", "Bỏ label vào dataset/raw/labels", YELLOW))
        print(row("Bước 3", "Chạy training/validate_dataset.py", YELLOW))
        print(row("Bước 4", "Chạy lại training/split_dataset.py", GREEN))
        print(line(rule("-"), CYAN))
        print(section("Ý NGHĨA LỆNH", CYAN))
        print(row("Lệnh này", "Lấy dữ liệu raw hợp lệ và chia sang train / val / test.", YELLOW, bounded=False))
        print(line(rule("="), CYAN))
        return

    _reset_processed_dirs()
    split_map = _split_items(eligible)
    for split_name, items in split_map.items():
        _copy_split(split_name, items)

    for item in header("YOLO DATASET :: CHIA TRAIN / VAL / TEST"):
        print(item)
    print(section("TỔNG QUAN", GREEN))
    print(row("Tổng ảnh raw", str(total_images), GREEN))
    print(row("Ảnh hợp lệ", str(len(eligible)), GREEN if eligible else YELLOW))
    print(row("Ảnh thiếu label", str(len(report.missing_labels)), RED if report.missing_labels else GREEN))
    print(row("Label lỗi", str(len(report.invalid_labels)), RED if report.invalid_labels else GREEN))
    print(row("Label mồ côi", str(len(report.orphan_labels)), YELLOW if report.orphan_labels else GREEN))
    print(line(rule("-"), CYAN))
    print(section("KẾT QUẢ SPLIT", YELLOW))
    print(row("Train", f"{len(split_map['train'])} file", GREEN))
    print(row("Val", f"{len(split_map['val'])} file", YELLOW))
    print(row("Test", f"{len(split_map['test'])} file", YELLOW))
    print(line(rule("-"), CYAN))
    print(section("THƯ MỤC ĐÍCH", CYAN))
    print(row("Images", str(PROCESSED_IMAGES_DIR), CYAN, bounded=False))
    print(row("Labels", str(PROCESSED_LABELS_DIR), CYAN, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("LỆNH TIẾP", CYAN))
    print(command_row(1, r".\.venv\Scripts\python run_train.py"))
    print(command_row(2, r".\.venv\Scripts\python training\validate_model.py"))
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    main()
