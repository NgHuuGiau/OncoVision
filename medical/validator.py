from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image

from utils.file_utils import load_yaml
from medical.dataset import supported_medical_modalities_for_target
from medical.router import InputRoute, route_input

MEDICAL_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
DEFAULT_MIN_CONFIDENCE = 0.70

_DEFAULT_ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".dcm", ".nii", ".nii.gz"]

_MODALITY_LABEL_TO_CANONICAL = {
    "Mammogram": "mammogram",
    "Siêu âm vú": "ultrasound",
    "MRI vú": "mri",
    "X-quang ngực": "xray",
    "CT ngực": "ct",
    "MRI tuyến tiền liệt": "mri",
    "MRI trực tràng": "mri",
    "Nội soi đại tràng": "colonoscopy",
    "CT ngực-bụng-chậu": "ct",
    "EUS": "eus",
    "PET/CT": "pet_ct",
    "PET": "pet_ct",
    "Nội soi": "endoscopy",
    "Siêu âm": "ultrasound",
    "CT": "ct",
    "MRI": "mri",
}

_TARGET_LABEL_TO_CANONICAL = {
    "liver": "liver",
    "lung": "lung",
    "breast": "breast",
    "stomach": "stomach",
    "colorectal": "colorectal",
    "prostate": "prostate",
    "cervical": "cervix",
}

_MODALITY_TO_TARGET_KEY = {
    "Mammogram": "breast",
    "Siêu âm vú": "breast",
    "MRI vú": "breast",
    "X-quang ngực": "lung",
    "CT ngực": "lung",
    "MRI tuyến tiền liệt": "prostate",
    "MRI trực tràng": "colorectal",
    "Nội soi đại tràng": "colorectal",
    "CT ngực-bụng-chậu": "colorectal",
    "EUS": "stomach",
    "PET/CT": "lung",
    "PET": "lung",
    "Nội soi": "stomach",
    "Siêu âm": "liver",
    "CT": "liver",
    "MRI": "liver",
}

_DICOM_MODALITY_MAP = {
    "CT": "ct",
    "MR": "mri",
    "US": "ultrasound",
    "PT": "pet_ct",
    "MG": "mammogram",
    "CR": "xray",
    "DX": "xray",
    "XA": "xray",
}

_DEFAULT_MODALITY_TUNING: dict[str, float | str] = {
    "certainty_threshold": 0.55,
    "medium_threshold": 0.45,
    "quality_threshold": 0.45,
    "contrast_boost": 1.0,
    "normalize": "default",
}

SUPPORTED_MAPPING = {
    "liver": supported_medical_modalities_for_target("liver"),
    "lung": supported_medical_modalities_for_target("lung"),
    "breast": supported_medical_modalities_for_target("breast"),
    "stomach": supported_medical_modalities_for_target("stomach"),
    "colorectal": supported_medical_modalities_for_target("colorectal"),
    "prostate": supported_medical_modalities_for_target("prostate"),
    "cervix": supported_medical_modalities_for_target("cervical"),
}

DEFAULT_MEDICAL_SETTINGS_PATH = Path("config/medical_settings.yaml")


def _collect_image_text(image_path: str) -> str:
    from medical.dataset import _collect_dicom_text, resolve_medical_upload_path

    source = Path(image_path)
    parts = [source.name, *source.parts[-3:]]
    if source.is_dir():
        try:
            candidate = resolve_medical_upload_path(source)
        except Exception:
            candidate = None
        if candidate is not None and candidate.is_file():
            parts.append(_collect_dicom_text(candidate))
        return " ".join(parts)
    parts.append(_collect_dicom_text(source))
    return " ".join(parts)


def _score_hint_confidence(normalized_text: str, hints, target_label):
    from medical.dataset import _find_first_matching_hint

    if target_label is None:
        target_label = _find_first_matching_hint(normalized_text, hints)
    if target_label is None:
        return 0.0
    padded = f" {normalized_text} "
    terms = []
    for label, ts in hints:
        if label == target_label:
            terms = list(ts)
            break
    if not terms:
        return 0.0
    matches = sum(1 for term in terms if term in padded)
    if matches == 0:
        return 0.0
    return matches / len(terms)


def _fallback_modality_label(normalized_text: str) -> str | None:
    padded = f" {normalized_text} "
    patterns = (
        ("Mammogram", r"\bmammo(?:gram)?\b"),
        ("MRI", r"\bmri\b"),
        ("CT", r"\bct\b"),
        ("X-quang ngực", r"\bcxr\b|\bxray\b|\bchest radiograph\b"),
        ("Siêu âm", r"\bultrasound\b|\bsonography\b"),
        ("Nội soi đại tràng", r"\bcolonoscopy\b"),
        ("Nội soi", r"\bendoscopy\b|\begd\b|\bgastroscopy\b"),
        ("EUS", r"\beus\b"),
        ("PET/CT", r"\bpet\s*/?\s*ct\b|\bpetct\b"),
        ("PET", r"\bpet\b"),
    )
    for label, pattern in patterns:
        if re.search(pattern, padded):
            return label
    return None


def _canonical_modality(label):
    if label is None:
        return None
    canonical = _MODALITY_LABEL_TO_CANONICAL.get(label)
    if canonical is not None:
        return canonical
    # Fallback: derive canonical modality from the modality keyword in the label.
    # Covers specific labels from dataset._MODALITY_HINTS missing in the static map
    # (e.g. "CT gan", "MRI gan", "Siêu âm gan") so valid inputs are not rejected.
    lowered = label.lower()
    if lowered.startswith("pet/ct") or "pet ct" in lowered or lowered.startswith("pet"):
        return "pet_ct"
    if lowered.startswith("ct"):
        return "ct"
    if lowered.startswith("mri"):
        return "mri"
    if lowered.startswith("eus"):
        return "eus"
    if "noi soi dai trang" in lowered or "nội soi đại tràng" in lowered:
        return "colonoscopy"
    if "noi soi" in lowered or "nội soi" in lowered:
        return "endoscopy"
    if lowered.startswith("sieu am") or "siêu âm" in lowered:
        return "ultrasound"
    if lowered.startswith("x-quang") or "x quang" in lowered:
        return "xray"
    if lowered.startswith("mammogram") or "mammo" in lowered:
        return "mammogram"
    return None


def _canonical_body_region(label):
    if label is None:
        return None
    mapped = _MODALITY_TO_TARGET_KEY.get(label)
    if mapped is not None:
        return _TARGET_LABEL_TO_CANONICAL.get(mapped, mapped)
    from medical.cancer_catalog import COMMON_CANCER_TARGETS
    for target in COMMON_CANCER_TARGETS:
        if target.key == label:
            return _TARGET_LABEL_TO_CANONICAL.get(label, label)
        if target.label == label:
            return _TARGET_LABEL_TO_CANONICAL.get(target.key, target.key)
    return _TARGET_LABEL_TO_CANONICAL.get(label, label)


@lru_cache(maxsize=1)
def _load_modality_tuning_from_config() -> dict[str, dict[str, float | str]]:
    try:
        settings = load_yaml(DEFAULT_MEDICAL_SETTINGS_PATH).get("medical", {})
    except Exception:
        settings = {}
    tuning = settings.get("modality_tuning", {})
    return tuning if isinstance(tuning, dict) else {}


def _normalize_tuning_block(block: dict[str, Any] | None) -> dict[str, float | str]:
    merged: dict[str, float | str] = dict(_DEFAULT_MODALITY_TUNING)
    if isinstance(block, dict):
        for key, value in block.items():
            if key in merged and value is not None and isinstance(value, (int, float, str)):
                merged[key] = value
    return merged


def get_modality_tuning(modality: str | None, tuning_settings: dict[str, Any] | None = None) -> dict[str, float | str]:
    source = tuning_settings if tuning_settings else _load_modality_tuning_from_config()
    default_block = _normalize_tuning_block(source.get("default") if isinstance(source, dict) else None)
    modality_key = (modality or "").lower()
    if isinstance(source, dict) and modality_key in source:
        return _normalize_tuning_block({**default_block, **source[modality_key]})
    return default_block


def _infer_body_region_from_text(source, normalized_text: str, modality_label: str | None = None):
    from medical.dataset import _DICOM_BODY_PART_TO_TARGET, _find_first_matching_hint, _TARGET_HINTS

    target_key = _find_first_matching_hint(normalized_text, _TARGET_HINTS)
    if target_key is not None:
        return target_key, 0.9

    if source.suffix.lower() == ".dcm":
        try:
            import pydicom

            ds = pydicom.dcmread(str(source), stop_before_pixels=True, force=True)
            body_part = str(getattr(ds, "BodyPartExamined", "")).upper()
            if body_part:
                mapped = _DICOM_BODY_PART_TO_TARGET.get(body_part)
                if mapped is not None:
                    return mapped, 0.95
        except Exception:
            pass

    if modality_label is not None:
        from medical.dataset import _MODALITY_TO_TARGET_KEY as DATASET_MODALITY_TO_TARGET_KEY

        mapped = DATASET_MODALITY_TO_TARGET_KEY.get(modality_label)
        if mapped is not None and modality_label not in {"CT", "MRI", "PET", "Siêu âm", "Nội soi"}:
            return mapped, 0.75

    if modality_label is not None:
        mapped = _MODALITY_TO_TARGET_KEY.get(modality_label)
        if mapped is not None and modality_label not in {"CT", "MRI", "PET", "Siêu âm", "Nội soi"}:
            return mapped, 0.75

    return None, 0.0


@dataclass(frozen=True)
class ValidationResult:
    status: str
    error_code: str | None = None
    message: str | None = None
    modality: str | None = None
    body_region: str | None = None
    modality_confidence: float = 0.0
    body_region_confidence: float = 0.0
    quality_warnings: tuple[str, ...] = ()
    route: InputRoute | None = None


def _is_size_only_warning(warnings: list[str] | tuple[str, ...]) -> bool:
    return bool(warnings) and all("kích thước" in warning.lower() for warning in warnings)


def assess_image_quality(image_path: str | Path) -> tuple[list[str], float]:
    source = Path(image_path)
    if not source.exists():
        return ["Không thể đọc ảnh để đánh giá chất lượng đầu vào."], 0.0

    image = None
    if source.suffix.lower() == ".dcm":
        try:
            import pydicom

            ds = pydicom.dcmread(str(source), force=True)
            pixel_array = ds.pixel_array
            if pixel_array is None:
                raise ValueError("No pixel data")
            if pixel_array.ndim == 2:
                image = np.stack([pixel_array] * 3, axis=-1)
            else:
                image = np.asarray(pixel_array)
            if np.issubdtype(image.dtype, np.integer):
                image = np.clip(image, 0, np.iinfo(image.dtype).max)
                image = image.astype(np.float32)
                image = (image - image.min()) / max(image.max() - image.min(), 1.0) * 255.0
                image = image.astype(np.uint8)
            else:
                image = np.clip(image, 0, 255)
        except Exception:
            return ["Không thể đọc ảnh để đánh giá chất lượng đầu vào."], 0.0
    else:
        try:
            with Image.open(source) as img:
                image = np.array(img.convert("RGB"))
        except Exception:
            return ["Không thể đọc ảnh để đánh giá chất lượng đầu vào."], 0.0

    if image is None or image.size == 0:
        return ["Ảnh rỗng hoặc không đọc được."], 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    warnings: list[str] = []

    if min(height, width) < 256:
        warnings.append("kích thước ảnh quá nhỏ để phân tích chính xác")
    if blur_score < 60:
        warnings.append("ảnh quá mờ")
    if contrast < 20:
        warnings.append("độ tương phản thấp (contrast)")
    if brightness < 45:
        warnings.append("ảnh quá tối")
    if brightness > 220:
        warnings.append("ảnh quá sáng")

    quality_score = min(1.0, max(0.0, 0.35 + (blur_score / 300.0) * 0.3 + (contrast / 100.0) * 0.2 + (1.0 - abs(brightness - 128) / 128.0) * 0.15))
    return warnings, quality_score


def _validate_single_file(source, allowed, min_confidence):
    from medical.dataset import (
        _MODALITY_HINTS,
        _TARGET_HINTS,
        _MODALITY_TO_TARGET_KEY,
        _DICOM_MODALITY_MAP,
        _normalize_medical_text,
        _find_first_matching_hint,
    )

    raw_text = _collect_image_text(source)
    normalized = _normalize_medical_text(raw_text)

    modality_label = _find_first_matching_hint(normalized, _MODALITY_HINTS)
    if modality_label is None:
        modality_label = _fallback_modality_label(normalized)
    modality_confidence = _score_hint_confidence(normalized, _MODALITY_HINTS, modality_label)

    if modality_label is None and source.suffix.lower() == ".dcm":
        try:
            import pydicom
            ds = pydicom.dcmread(str(source), stop_before_pixels=True, force=True)
            dicom_modality = getattr(ds, "Modality", "").upper()
            mapped_modality = _DICOM_MODALITY_MAP.get(dicom_modality)
            if mapped_modality:
                modality_label = mapped_modality
                modality_confidence = 0.9
        except Exception:
            pass

    target_key_hint = _find_first_matching_hint(normalized, _TARGET_HINTS)
    if modality_label is None and target_key_hint == "cervical":
        return ValidationResult(
            status="error",
            error_code="NON_IMAGE_CERVICAL_INPUT",
            message="Pap, HPV, colposcopy va cac dau hieu lam sang tuong tu khong phai anh y khoa de dua qua cung pipeline nay.",
            body_region="cervical",
            route=route_input(None, "cervix"),
        )

    canonical_modality = _canonical_modality(modality_label)
    if canonical_modality is None:
        return ValidationResult(
            status="error",
            error_code="UNKNOWN_IMAGE_TYPE",
            message="Không xác định được ảnh. Vui lòng tải lên đúng loại ảnh y khoa được hỗ trợ.",
        )

    tuning = get_modality_tuning(canonical_modality)

    if modality_confidence < min_confidence:
        return ValidationResult(
            status="uncertain",
            error_code="LOW_CONFIDENCE",
            message="Không đủ độ tin cậy để nhận diện loại ảnh một cách chắc chắn. Hãy dùng ảnh rõ hơn hoặc có thêm thông tin mô tả.",
            modality=canonical_modality,
            modality_confidence=modality_confidence,
        )

    target_key, target_hint_confidence = _infer_body_region_from_text(source, normalized, modality_label)
    if target_key is None:
        target_key = _find_first_matching_hint(normalized, _TARGET_HINTS)

    canonical_body = _canonical_body_region(target_key)
    if canonical_body is None:
        return ValidationResult(
            status="error",
            error_code="UNKNOWN_BODY_REGION",
            message="Không xác định được vùng cơ thể trong ảnh.",
        )

    body_confidence = _score_hint_confidence(normalized, _TARGET_HINTS, target_key)
    if target_key is not None and target_hint_confidence > 0.0:
        body_confidence = max(body_confidence, target_hint_confidence)
    if target_key and modality_label and modality_label in _MODALITY_TO_TARGET_KEY and _MODALITY_TO_TARGET_KEY[modality_label] == target_key:
        body_confidence = max(body_confidence, 0.80)
    if body_confidence < min_confidence:
        return ValidationResult(
            status="uncertain",
            error_code="LOW_CONFIDENCE",
            message="Không đủ độ tin cậy để xác định vùng cơ thể. Hãy cung cấp ảnh có góc chụp rõ và vùng quan tâm được thấy đầy đủ.",
            modality=canonical_modality,
            body_region=canonical_body,
            modality_confidence=modality_confidence,
            body_region_confidence=body_confidence,
        )

    if source.name.lower().endswith(".nii.gz"):
        from medical.validator import SUPPORTED_MAPPING
        if canonical_body not in SUPPORTED_MAPPING:
            return ValidationResult(
                status="error",
                error_code="UNSUPPORTED_BODY_REGION",
                message="Vùng cơ thể này chưa được hệ thống hỗ trợ.",
            )
        if canonical_modality not in SUPPORTED_MAPPING[canonical_body]:
            return ValidationResult(
                status="error",
                error_code="UNSUPPORTED_IMAGE_FOR_CANCER_TYPE",
                message="Loại ảnh này không được hỗ trợ cho nhóm ung thư cần nhận diện.",
            )

    warnings, quality_score = assess_image_quality(source)
    quality_warnings = tuple(warnings)
    status = "success"
    if quality_warnings and quality_score < float(tuning["quality_threshold"]) and not _is_size_only_warning(quality_warnings):
        status = "uncertain"
    if status == "uncertain" and quality_warnings and (modality_confidence >= min_confidence or body_confidence >= min_confidence):
        status = "success"

    routed = route_input(canonical_modality, canonical_body)
    return ValidationResult(
        status=status,
        modality=canonical_modality,
        body_region=canonical_body,
        modality_confidence=modality_confidence,
        body_region_confidence=body_confidence,
        quality_warnings=quality_warnings,
        route=routed,
    )


def _validate_single_file_strict(source, allowed, min_confidence):
    from medical.dataset import (
        _DICOM_MODALITY_MAP,
        _MODALITY_HINTS,
        _MODALITY_TO_TARGET_KEY,
        _TARGET_HINTS,
        _find_first_matching_hint,
        _normalize_medical_text,
    )

    raw_text = _collect_image_text(source)
    normalized = _normalize_medical_text(raw_text)

    modality_label = _find_first_matching_hint(normalized, _MODALITY_HINTS)
    modality_confidence = _score_hint_confidence(normalized, _MODALITY_HINTS, modality_label)

    if modality_label is None and source.suffix.lower() == ".dcm":
        try:
            import pydicom

            ds = pydicom.dcmread(str(source), stop_before_pixels=True, force=True)
            dicom_modality = getattr(ds, "Modality", "").upper()
            mapped_modality = _DICOM_MODALITY_MAP.get(dicom_modality)
            if mapped_modality:
                modality_label = mapped_modality
                modality_confidence = 0.9
        except Exception:
            pass

    target_key_hint = _find_first_matching_hint(normalized, _TARGET_HINTS)
    if modality_label is None and target_key_hint == "cervical":
        return ValidationResult(
            status="error",
            error_code="NON_IMAGE_CERVICAL_INPUT",
            message="Pap, HPV, colposcopy va cac dau hieu lam sang tuong tu khong phai anh y khoa de dua qua cung pipeline nay.",
            body_region="cervical",
            route=route_input(None, "cervix"),
        )

    canonical_modality = _canonical_modality(modality_label)
    if canonical_modality is None:
        return ValidationResult(
            status="error",
            error_code="UNKNOWN_IMAGE_TYPE",
            message="Khong xac dinh duoc loai anh. Hay tai len dung loai anh y khoa duoc ho tro.",
        )

    tuning = get_modality_tuning(canonical_modality)
    if modality_confidence < min_confidence:
        return ValidationResult(
            status="uncertain",
            error_code="LOW_CONFIDENCE",
            message="Khong du do tin cay de nhan dien loai anh mot cach chac chan. Hay dung anh ro hon hoac co them thong tin mo ta.",
            modality=canonical_modality,
            modality_confidence=modality_confidence,
        )

    target_key, target_hint_confidence = _infer_body_region_from_text(source, normalized, modality_label)
    if target_key is None:
        target_key = _find_first_matching_hint(normalized, _TARGET_HINTS)

    canonical_body = _canonical_body_region(target_key)
    if canonical_body is None:
        return ValidationResult(
            status="error",
            error_code="UNKNOWN_BODY_REGION",
            message="Khong xac dinh duoc vung co the trong anh.",
        )

    if target_key == "cervical" and modality_label is None:
        return ValidationResult(
            status="error",
            error_code="NON_IMAGE_CERVICAL_INPUT",
            message="Pap, HPV, colposcopy va cac dau hieu lam sang tuong tu khong phai anh y khoa de dua qua cung pipeline nay.",
            body_region="cervical",
            route=route_input(None, "cervix"),
        )

    supported_modalities = SUPPORTED_MAPPING.get(canonical_body)
    if not supported_modalities:
        return ValidationResult(
            status="error",
            error_code="UNSUPPORTED_BODY_REGION",
            message="Vung co the nay chua duoc he thong ho tro.",
        )
    if canonical_modality not in supported_modalities:
        return ValidationResult(
            status="error",
            error_code="UNSUPPORTED_IMAGE_FOR_CANCER_TYPE",
            message="Loai anh nay khong duoc ho tro cho nhom ung thu can nhan dien.",
        )

    body_confidence = _score_hint_confidence(normalized, _TARGET_HINTS, target_key)
    if target_key is not None and target_hint_confidence > 0.0:
        body_confidence = max(body_confidence, target_hint_confidence)
    if target_key and modality_label and modality_label in _MODALITY_TO_TARGET_KEY and _MODALITY_TO_TARGET_KEY[modality_label] == target_key:
        body_confidence = max(body_confidence, 0.80)
    if body_confidence < min_confidence:
        return ValidationResult(
            status="uncertain",
            error_code="LOW_CONFIDENCE",
            message="Khong du do tin cay de xac dinh vung co the. Hay cung cap anh co goc chup ro va vung quan tam duoc thay day du.",
            modality=canonical_modality,
            body_region=canonical_body,
            modality_confidence=modality_confidence,
            body_region_confidence=body_confidence,
        )

    warnings, quality_score = assess_image_quality(source)
    quality_warnings = tuple(warnings)
    status = "success"
    if quality_warnings and quality_score < float(tuning["quality_threshold"]) and not _is_size_only_warning(quality_warnings):
        status = "uncertain"
    if status == "uncertain" and quality_warnings and (modality_confidence >= min_confidence or body_confidence >= min_confidence):
        status = "success"

    routed = route_input(canonical_modality, canonical_body)
    return ValidationResult(
        status=status,
        modality=canonical_modality,
        body_region=canonical_body,
        modality_confidence=modality_confidence,
        body_region_confidence=body_confidence,
        quality_warnings=quality_warnings,
        route=routed,
    )


def validate_image(image_path, allowed_extensions=None, min_confidence=DEFAULT_MIN_CONFIDENCE):
    allowed = frozenset(allowed_extensions or _DEFAULT_ALLOWED_EXTENSIONS)
    source = Path(image_path)

    suffix = source.suffix.lower()
    if suffix == ".gz" and source.name.lower().endswith(".nii.gz"):
        suffix = ".nii.gz"
    if suffix not in allowed:
        if source.is_dir():
            return _validate_directory(source, allowed, min_confidence)
        return ValidationResult(
            status="error",
            error_code="INVALID_FILE_FORMAT",
            message="Định dạng file không được hỗ trợ. Chỉ chấp nhận jpg, jpeg, png, dcm, nii, nii.gz.",
        )

    if not source.exists():
        return ValidationResult(
            status="error",
            error_code="IMAGE_READ_FAILED",
            message="Không tìm thấy file ảnh. Vui lòng tải lại ảnh hợp lệ.",
        )

    if source.is_dir():
        return _validate_directory(source, allowed, min_confidence)

    try:
        if source.suffix.lower() not in {".dcm", ".nii", ".nii.gz"}:
            with Image.open(source) as img:
                img.convert("RGB")
    except Exception:
        if source.suffix.lower() in {".dcm", ".nii", ".nii.gz"}:
            pass
        else:
            return ValidationResult(
                status="error",
                error_code="IMAGE_READ_FAILED",
                message="Không thể đọc ảnh. Vui lòng tải lại ảnh hợp lệ.",
            )

    return _validate_single_file_strict(source, allowed, min_confidence)


def _validate_directory(source, allowed, min_confidence):
    from medical.dataset import _medical_upload_suffix

    candidates = sorted(
        candidate
        for candidate in source.rglob("*")
        if candidate.is_file() and _medical_upload_suffix(candidate) in allowed
    )
    if not candidates:
        return ValidationResult(
            status="error",
            error_code="INVALID_FILE_FORMAT",
            message="Thư mục không chứa file ảnh hợp lệ. Chỉ chấp nhận jpg, jpeg, png, dcm, nii, nii.gz.",
        )

    dicom_candidates = [c for c in candidates if _medical_upload_suffix(c) == ".dcm"]
    if dicom_candidates:
        representative = dicom_candidates[len(dicom_candidates) // 2]
    else:
        representative = candidates[0]

    result = _validate_single_file(representative, allowed, min_confidence)
    if result.status == "uncertain" and result.quality_warnings and _is_size_only_warning(result.quality_warnings):
        return ValidationResult(
            status="success",
            error_code=result.error_code,
            message=result.message,
            modality=result.modality,
            body_region=result.body_region,
            modality_confidence=result.modality_confidence,
            body_region_confidence=result.body_region_confidence,
            quality_warnings=result.quality_warnings,
            route=result.route,
        )
    return result
