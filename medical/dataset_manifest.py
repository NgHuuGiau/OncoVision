from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from medical.cancer_catalog import get_cancer_target
from medical.dataset import (
    is_supported_medical_upload_path,
    supported_medical_modalities_for_target,
)
from medical.validator import _canonical_modality

REQUIRED_FIELDS = ("patient_id", "image_path", "cancer_target", "modality", "truth_label", "site_id")
ALLOWED_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class DatasetAudit:
    row_count: int
    patient_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def _read_rows(csv_path: str | Path) -> tuple[Path, list[dict[str, str]], list[str]]:
    source = Path(csv_path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    return source, rows, fields


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_dataset_manifest(csv_path: str | Path) -> DatasetAudit:
    source, rows, fields = _read_rows(csv_path)
    errors: list[str] = []
    warnings: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in fields]
    if missing:
        return DatasetAudit(len(rows), 0, (f"Thiếu cột bắt buộc: {', '.join(missing)}",), ())

    patient_splits: dict[str, set[str]] = {}
    seen_images: set[Path] = set()
    root = source.parent
    for number, row in enumerate(rows, start=2):
        patient_id = row["patient_id"]
        if not patient_id:
            errors.append(f"Dòng {number}: patient_id trống.")
        target = get_cancer_target(row["cancer_target"])
        if target is None:
            errors.append(f"Dòng {number}: cancer_target không hợp lệ: {row['cancer_target']!r}.")
        modality = _canonical_modality(row["modality"])
        if modality is None:
            errors.append(f"Dòng {number}: modality không hợp lệ: {row['modality']!r}.")
        elif target is not None and modality not in supported_medical_modalities_for_target(target.key):
            errors.append(f"Dòng {number}: modality {modality!r} không phù hợp target {target.key!r}.")
        if not row["truth_label"]:
            errors.append(f"Dòng {number}: truth_label trống.")
        if not row["site_id"]:
            errors.append(f"Dòng {number}: site_id trống.")

        image = _resolve_path(root, row["image_path"])
        if image in seen_images:
            errors.append(f"Dòng {number}: image_path bị lặp: {row['image_path']}.")
        seen_images.add(image)
        if not image.exists():
            errors.append(f"Dòng {number}: không tìm thấy image_path: {row['image_path']}.")
        elif not is_supported_medical_upload_path(image):
            errors.append(f"Dòng {number}: định dạng ảnh không được hỗ trợ: {row['image_path']}.")

        mask_value = row.get("mask_path", "")
        if mask_value and not _resolve_path(root, mask_value).is_file():
            errors.append(f"Dòng {number}: không tìm thấy mask_path: {mask_value}.")
        bbox_value = row.get("bbox", "")
        if bbox_value:
            try:
                bbox = json.loads(bbox_value)
                if not isinstance(bbox, list) or len(bbox) != 4 or any(float(value) < 0 for value in bbox):
                    raise ValueError
            except (ValueError, TypeError, json.JSONDecodeError):
                errors.append(f"Dòng {number}: bbox phải là JSON [x1,y1,x2,y2] không âm.")

        split = row.get("split", "").lower()
        if split:
            if split not in ALLOWED_SPLITS:
                errors.append(f"Dòng {number}: split phải là train, validation hoặc test.")
            patient_splits.setdefault(patient_id, set()).add(split)
        if not row.get("protocol", ""):
            warnings.append(f"Dòng {number}: thiếu protocol chụp.")
        if not row.get("scanner", ""):
            warnings.append(f"Dòng {number}: thiếu thông tin máy chụp.")
        if not mask_value and not bbox_value:
            warnings.append(f"Dòng {number}: không có mask/bbox; chỉ dùng được cho classification.")

    for patient_id, splits in patient_splits.items():
        if len(splits) > 1:
            errors.append(f"Rò rỉ bệnh nhân {patient_id!r} giữa các split: {', '.join(sorted(splits))}.")
    return DatasetAudit(len(rows), len({row["patient_id"] for row in rows if row["patient_id"]}), tuple(errors), tuple(warnings))


def split_dataset_by_patient(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: str = "oncovision",
) -> Path:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Tỷ lệ train/validation phải dương và tổng phải nhỏ hơn 1.")
    source, rows, fields = _read_rows(csv_path)
    audit = validate_dataset_manifest(source)
    if not audit.valid:
        raise ValueError("Dataset không hợp lệ: " + "; ".join(audit.errors))
    patient_split: dict[str, str] = {}
    for patient_id in sorted({row["patient_id"] for row in rows}):
        digest = hashlib.sha256(f"{seed}:{patient_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        patient_split[patient_id] = (
            "train" if value < train_ratio else "validation" if value < train_ratio + validation_ratio else "test"
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_fields = [field for field in fields if field != "split"] + ["split"]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "split": patient_split[row["patient_id"]]})
    return destination


__all__ = ["DatasetAudit", "split_dataset_by_patient", "validate_dataset_manifest"]
