from __future__ import annotations

import argparse
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from medical.dataset import create_default_medical_dataset_config, ensure_medical_dataset_structure
from medical.importers import import_isic_2016_part3b_to_yolo
from training.terminal_ui import CYAN, GREEN, YELLOW, command_row, header, line, row, rule, section
from utils.terminal_encoding import ensure_utf8_console


DEFAULT_SOURCE_ROOT = Path("dataset/medical/skin_lesion/downloads/ISIC2016_Part3B/ISBI2016_ISIC_Part3B_Training_Data")
DEFAULT_DIAGNOSIS_CSV = Path("dataset/medical/skin_lesion/downloads/ISBI2016_Part3B/Training_GroundTruth.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import full ISIC2016 Part3B skin lesion dataset into dataset/medical/skin_lesion")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--diagnosis-csv", default=str(DEFAULT_DIAGNOSIS_CSV))
    parser.add_argument("--dataset-root", default="dataset/medical/skin_lesion")
    parser.add_argument("--metadata-output", default=None)
    return parser.parse_args()


def main() -> None:
    ensure_utf8_console()
    args = _parse_args()
    config = create_default_medical_dataset_config(args.dataset_root)
    ensure_medical_dataset_structure(config)
    raw_images_dir = config.dataset_root / "raw" / "images"
    raw_labels_dir = config.dataset_root / "raw" / "labels"
    metadata_output = Path(args.metadata_output) if args.metadata_output else config.metadata_dir / "isic2016_import.csv"
    for item in header("YOLO DATASET :: IMPORT SKIN LESION"):
        print(item)
    print(section("NGUON", GREEN))
    print(row("Source root", args.source_root, GREEN, bounded=False))
    print(row("Diagnosis CSV", args.diagnosis_csv, YELLOW, bounded=False))
    print(row("Target raw images", str(raw_images_dir), GREEN, bounded=False))
    print(row("Target raw labels", str(raw_labels_dir), GREEN, bounded=False))
    print(line(rule("-"), CYAN))
    result = import_isic_2016_part3b_to_yolo(
        args.source_root,
        target_images_dir=raw_images_dir,
        target_labels_dir=raw_labels_dir,
        diagnosis_csv_path=args.diagnosis_csv,
        metadata_output_path=metadata_output,
    )
    print(section("KET QUA", GREEN))
    print(row("Imported", str(result["imported"]), GREEN, bounded=False))
    print(row("Skipped", str(result["skipped"]), YELLOW, bounded=False))
    print(row("Metadata", str(metadata_output), GREEN, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("BUOC SAU", CYAN))
    print(command_row(1, r".\.venv\Scripts\python training\split_dataset.py"))
    print(command_row(2, r".\.venv\Scripts\python run_medical.py status"))
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    main()
