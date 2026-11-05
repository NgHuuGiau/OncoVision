from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from medical.cancer_catalog import get_cancer_target
from medical.validator import _canonical_modality


@dataclass(frozen=True)
class MedicalCaseManifest:
    patient_id: str
    cancer_target: str
    modality: str
    image: Path


def load_case_manifest(path: str | Path) -> MedicalCaseManifest:
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Manifest ca bệnh phải là một YAML mapping.")
    missing = [field for field in ("patient_id", "cancer_target", "modality", "image") if not data.get(field)]
    if missing:
        raise ValueError(f"Manifest ca bệnh thiếu: {', '.join(missing)}.")
    target_key = str(data["cancer_target"]).strip().lower()
    if get_cancer_target(target_key) is None:
        raise ValueError(f"cancer_target không hợp lệ: {target_key}.")
    modality = str(data["modality"]).strip()
    if _canonical_modality(modality) is None:
        raise ValueError(f"modality không hợp lệ: {modality}.")
    image = Path(str(data["image"]))
    if not image.is_absolute():
        image = source.parent / image
    if not image.exists():
        raise ValueError(f"Không tìm thấy ảnh/volume: {image}.")
    return MedicalCaseManifest(str(data["patient_id"]).strip(), target_key, modality, image)


__all__ = ["MedicalCaseManifest", "load_case_manifest"]
