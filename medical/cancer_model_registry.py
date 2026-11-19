"""Registry nhẹ cho model theo từng nhóm ung thư."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from app.chat_ui.paths import PROJECT_ROOT
from medical.cancer_catalog import COMMON_CANCER_TARGETS

logger = logging.getLogger(__name__)


def _get_cancers_dir() -> Path:
    """Lay duong den models/pretrained/cancers/, hoat dong dung ca khi chay tu exe."""
    if getattr(sys, "frozen", False):
        base_dir = Path(os.path.dirname(sys.executable))
        internal_dir = base_dir / "_internal"
        if (internal_dir / "models" / "pretrained" / "cancers").is_dir():
            return internal_dir / "models" / "pretrained" / "cancers"
        if (base_dir / "models" / "pretrained" / "cancers").is_dir():
            return base_dir / "models" / "pretrained" / "cancers"
    return PROJECT_ROOT / "models/pretrained/cancers"


CANCERS_DIR = _get_cancers_dir()

# Labels COCO generic — nếu detector có >= 5 labels này, đánh dấu là không hợp lệ
_COCO_GENERIC_LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

# Các labels y khoa hợp lệ (minimal set)
_MEDICAL_LABEL_PATTERNS = {
    "tumor", "lesion", "mass", "nodule", "cyst", "calcification", "malignant", "benign",
    "carcinoma", "glioma", "meningioma", "pituitary", "abnormal", "dysplasia", "ulcer",
    "polyp", "adenoma", "fibroadenoma", "thyroid", "nodule_thyroid", "thyroid_nodule",
}


def _is_medical_detector(model_labels: list[str]) -> tuple[bool, str]:
    """Kiểm tra detector có labels y khoa hợp lệ hay không.

    Trả về (is_medical, reason).
    """
    if not model_labels:
        return False, "không có labels"

    # YOLO names có thể trả về dict {int: str} — chuyển sang list[str]
    if model_labels and isinstance(next(iter(model_labels), None), int):
        # model_labels là dict-like với keys là int
        model_labels = [str(label) for label in model_labels]
    labels_lower = [str(label).lower().strip() for label in model_labels]

    # Đếm số labels COCO generic
    coco_count = sum(1 for label in labels_lower if label in _COCO_GENERIC_LABELS)
    if coco_count >= 5:
        return False, f"dùng COCO generic ({coco_count}/{len(labels_lower)} labels không liên quan y khoa)"

    # Kiểm tra có ít nhất 1 label y khoa
    medical_count = 0
    for label in labels_lower:
        for pattern in _MEDICAL_LABEL_PATTERNS:
            if pattern in label:
                medical_count += 1
                break
    # Nếu không có label y khoa nào và có nhiều label (>10), có thể là generic
    if medical_count == 0 and len(labels_lower) > 10:
        return False, f"không có labels y khoa ({len(labels_lower)} labels tổng quát)"

    return True, ""

KEY_TO_DIR = {target.key: target.key for target in COMMON_CANCER_TARGETS}

_KIND_BY_SUFFIX = {
    ".pt2": "torch-export",
    ".pt": "torch-state",
    ".pth": "torch-state",
    ".onnx": "onnx",
    ".bin": "hf-transformers",
    ".safetensors": "safetensors",
    ".ckpt": "ckpt",
    ".keras": "keras",
}


def _detect_kind(files: list[Path]) -> str:
    kinds = [_KIND_BY_SUFFIX[p.suffix.lower()] for p in files if p.suffix.lower() in _KIND_BY_SUFFIX]
    for preferred in ("torch-export", "torch-state", "onnx", "hf-transformers", "safetensors", "ckpt", "keras"):
        if preferred in kinds:
            return preferred
    return "unknown"


def _first_file(directory: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return directory / names[0]


_PROBE_CACHE: dict[tuple[str, str, int, int], tuple[bool, str]] = {}

# Cache cho detector validation warnings — chỉ log 1 lần mỗi file
_DETECTOR_WARNING_LOGGED: set[str] = set()


def _short(exc: object) -> str:
    return " ".join(str(exc).split())[:200]


def probe_runtime_capability(path: str | Path, kind: str) -> tuple[bool, str]:
    """Kiểm tra file có nạp được runtime thật không; trả (ok, lý do).

    Dùng cùng cơ chế load của pipeline (pickle không an toàn) nên chỉ chạy với
    file người dùng tự tin tưởng đặt vào models/pretrained/.
    """
    source = Path(path)
    try:
        stat = source.stat()
    except OSError as exc:
        return False, f"không đọc được file ({exc})"
    key = (kind, str(source), stat.st_mtime_ns, stat.st_size)
    cached = _PROBE_CACHE.get(key)
    if cached is not None:
        return cached
    if kind == "classifier":
        try:
            from medical.cnn_classifier import load_cnn_classifier

            load_cnn_classifier(source)
        except Exception as exc:
            return False, f"không nạp được classifier ({_short(exc)})"
        result = (True, "")
    elif kind == "detector":
        try:
            from ultralytics import YOLO

            model = YOLO(str(source))
        except Exception as exc:
            result = (False, f"không nạp được YOLO ({type(exc).__name__}: {_short(exc)})")
        else:
            task = getattr(model, "task", None)
            if task not in {"detect", "segment"}:
                result = (False, f"task={task!r}, cần detect hoặc segment")
            else:
                # Kiểm tra labels có phải y khoa không
                model_labels = getattr(getattr(model, "names", None), "names", list(getattr(model, "names", {}) or {}))
                if isinstance(model_labels, dict):
                    model_labels = list(model_labels.values())
                is_medical, medical_reason = _is_medical_detector(model_labels)
                if not is_medical:
                    result = (False, f"detector không phải y khoa ({medical_reason})")
                    source_str = str(source)
                    if source_str not in _DETECTOR_WARNING_LOGGED:
                        logger.warning("Detector %s không hợp lệ: %s", source, medical_reason)
                        _DETECTOR_WARNING_LOGGED.add(source_str)
                    _PROBE_CACHE[key] = result
                    return result
                # Nạp được chưa đủ: state_dict lạ vẫn load được nhưng predict mới vỡ
                try:
                    import numpy as np

                    model.predict(np.zeros((320, 320, 3), dtype=np.uint8), imgsz=320, verbose=False)
                except Exception as exc:
                    result = (False, f"không predict được ({type(exc).__name__}: {_short(exc)})")
                else:
                    result = (True, "")
    elif kind == "volume":
        try:
            from medical.volume_inference import load_volume_model

            load_volume_model(source)
        except Exception as exc:
            result = (False, f"không nạp được model 3D ({type(exc).__name__}: {_short(exc)})")
        else:
            result = (True, "")
    else:
        result = (False, f"kind không hỗ trợ: {kind}")
    _PROBE_CACHE[key] = result
    return result


def _any_probe(paths: list[Path], kind: str) -> tuple[bool, str]:
    reason = ""
    for candidate in paths:
        ok, reason = probe_runtime_capability(candidate, kind)
        if ok:
            return True, ""
    return False, reason


def modality_key(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    aliases = {
        "mri_não": "mri", "ct_sọ_não": "ct", "x_quang_ngực": "xray",
        "siêu_âm": "ultrasound", "siêu_âm_vú": "ultrasound", "siêu_âm_gan": "ultrasound",
        "siêu_âm_thận": "ultrasound", "siêu_âm_tuyến_giáp": "ultrasound",
        "nội_soi": "endoscopy", "nội_soi_đại_tràng": "colonoscopy",
        "ảnh_soi_cổ_tử_cung": "colposcopy", "soi_cổ_tử_cung": "colposcopy",
        "pet_ct": "pet_ct",
        "colonoscopy": "endoscopy", "colon": "endoscopy",
        "histopathology": "histopathology", "histology": "histopathology",
    }
    if normalized in aliases:
        return aliases[normalized]
    for prefix, canonical in (("mri", "mri"), ("ct", "ct"), ("siêu_âm", "ultrasound"), ("pet_ct", "pet_ct")):
        if normalized.startswith(prefix):
            return canonical
    return normalized


def target_model_paths(key: str, base: Path | None = None, modality: str | None = None) -> dict[str, Path]:
    """Tra duong dan model, khoa theo modality khi modality da duoc biet.

    Neu modality khong duoc chi dinh, tra ve organ root directory.
    Neu base la None, tu dong phat hien (hoat dong dung khi chay tu exe).
    """
    if base is None:
        base = CANCERS_DIR
    normalized = key.strip().lower()
    target_directory = Path(base) / KEY_TO_DIR.get(normalized, normalized)
    modality_dir_key = modality_key(modality)
    modality_directory = target_directory / modality_dir_key if modality_dir_key else None
    
    # Neu modality duoc chi dinh, chi tim trong modality subdirectory do
    if modality_dir_key:
        # Chi tra ve modality directory neu no ton tai, nguoc lai tra ve modality directory (khong ton tai)
        if modality_directory.exists():
            directory = modality_directory
            return _build_model_paths(directory, normalized, target_directory, allow_fallback=False)
        else:
            # Modality khong ton tai - tra ve modality directory (khong ton tai), KHONG fallback ve root
            return _build_model_paths(modality_directory, normalized, target_directory, allow_fallback=False)
    
    # Neu modality khong chi dinh, tra ve organ root directory
    return _build_model_paths(target_directory, normalized, target_directory, allow_fallback=True)


def _build_model_paths(directory: Path, normalized: str, target_directory: Path, allow_fallback: bool = False) -> dict[str, Path]:
    """Xay dung dict cac duong dan model tu mot directory cho truoc."""
    classifier_path = directory / "classifier.pt"
    if not classifier_path.is_file() and allow_fallback:
        # Check root target directory for fallback (only when modality not explicitly specified)
        if (target_directory / "classifier.pt").is_file():
            classifier_path = target_directory / "classifier.pt"
        elif (target_directory / f"{normalized}_classifier.pt").is_file():
            classifier_path = target_directory / f"{normalized}_classifier.pt"
        else:
            # Check for wrapped models
            wrapped_files = list(directory.glob("*_wrapped.pt"))
            if not wrapped_files:
                wrapped_files = list(target_directory.glob("*_wrapped.pt"))
            if not wrapped_files:
                wrapped_files = list(target_directory.glob(f"{normalized}_*.pt"))
            if wrapped_files:
                classifier_path = wrapped_files[0]
    
    return {
        "directory": directory,
        "classifier": classifier_path,
        "detector": _first_file(
            directory, ("detector.pt", f"{normalized}_detector.pt", "detector.onnx", f"{normalized}_detector.onnx")
        ),
        "volume_model": _first_file(
            directory,
            (
                "volume.pt2",
                "volume_model.pt2",
                f"{normalized}_volume_model.pt2",
                "volume.pt",
                "volume_model.pt",
                f"{normalized}_volume_model.pt",
            ),
        ),
        "volume_manifest": _first_file(directory, ("volume.json", "volume_model.json", f"{normalized}_volume_model.json")),
        "clinical_validation": directory / "clinical_validation.json",
    }


def find_cancer_model(key: str, base: Path = CANCERS_DIR) -> dict:
    """Trả capability có mặt theo hợp đồng classifier/detector/volume."""
    normalized = key.strip().lower()
    paths = target_model_paths(normalized, base)
    d = paths["directory"]
    target_directory = Path(base) / KEY_TO_DIR.get(normalized, normalized)
    if normalized not in KEY_TO_DIR:
        return {"key": normalized, "dir": str(d), "kind": "missing", "files": [], "artifact_ready": False,
                "runtime_ready": False, "classification_ready": False, "localization_ready": False,
                "volume_ready": False, "clinical_validation_ready": False, "note": "Nhóm ung thư không có trong catalog."}
    if not d.is_dir():
        return {"key": normalized, "dir": str(d), "kind": "missing", "files": [], "artifact_ready": False,
                "runtime_ready": False, "classification_ready": False, "localization_ready": False,
                "volume_ready": False, "clinical_validation_ready": False, "note": "Chưa có thư mục model cho nhóm này."}
    # Collect all model files in the selected modality directory only
    files = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in _KIND_BY_SUFFIX]
    # Classifier candidates: check root, modality subdirs, and target_directory
    classifier_candidates = [p for p in files if p.name in {"classifier.pt", f"{normalized}_classifier.pt"}]
    # Also check for wrapped models (any *_wrapped.pt)
    if not classifier_candidates:
        classifier_candidates = [p for p in files if p.name.endswith("_wrapped.pt") and "classifier" in p.name]
    if not classifier_candidates:
        classifier_candidates = [p for p in files if p.name.endswith("_wrapped.pt")]
    # Also check for models with group name prefix (e.g., liver_swin.pt, thyroid_tn.pt)
    if not classifier_candidates:
        classifier_candidates = [p for p in files if p.name.startswith(f"{normalized}_") and p.suffix == ".pt"]
    detector_candidates = [
        p
        for p in files
        if p.name in {"detector.pt", f"{normalized}_detector.pt", "detector.onnx", f"{normalized}_detector.onnx"}
    ]
    classification_ready, classification_reason = _any_probe(classifier_candidates, "classifier")
    localization_ready, detector_reason = _any_probe(detector_candidates, "detector")
    volume_candidates = [
        path
        for path in files
        if path.name
        in {
            "volume.pt2",
            "volume_model.pt2",
            f"{normalized}_volume_model.pt2",
            "volume.pt",
            "volume_model.pt",
            f"{normalized}_volume_model.pt",
        }
        and any((path.parent / name).is_file() for name in ("volume.json", "volume_model.json", f"{normalized}_volume_model.json"))
    ]
    volume_ready, volume_reason = _any_probe(volume_candidates, "volume")
    artifact_ready = bool(files)
    kind = _detect_kind(files) if files else "empty"
    rel = sorted(str(p.relative_to(d)) for p in files)
    runtime_ready = classification_ready or localization_ready or volume_ready
    note = "Đã có artifact đúng hợp đồng và nạp được runtime." if runtime_ready else "Artifact chưa khớp hợp đồng classifier.pt hoặc volume.pt2/volume.pt + volume.json."
    problems = []
    if classifier_candidates and not classification_ready:
        problems.append(f"classifier.pt không chạy được runtime ({classification_reason})")
    if detector_candidates and not localization_ready:
        problems.append(f"detector.pt không chạy được runtime ({detector_reason})")
    if volume_candidates and not volume_ready:
        problems.append(f"{volume_candidates[0].name} không chạy được runtime ({volume_reason})")
    if problems:
        note = f"{note} Lưu ý: {'; '.join(problems)}."
    for candidate in classifier_candidates + detector_candidates:
        source = candidate.with_suffix(".source.json")
        if not source.is_file():
            continue
        try:
            repo = json.loads(source.read_text(encoding="utf-8")).get("repo", "?")
        except (OSError, json.JSONDecodeError):
            continue
        note = f"{note} Nguồn {candidate.name}: {repo}."
    return {
        "key": normalized,
        "dir": str(d),
        "kind": kind,
        "files": rel,
        "artifact_ready": artifact_ready,
        "runtime_ready": runtime_ready,
        "classification_ready": classification_ready,
        "localization_ready": localization_ready,
        "volume_ready": volume_ready,
        "clinical_validation_ready": any(path.name == "clinical_validation.json" for path in d.rglob("clinical_validation.json")),
        "modality_directories": sorted(path.name for path in d.iterdir() if path.is_dir()),
        "note": note,
    }


def list_cancer_models(base: Path = CANCERS_DIR) -> list[dict]:
    return [find_cancer_model(key, base) for key in KEY_TO_DIR]


if __name__ == "__main__":
    for m in list_cancer_models():
        print(f"- {m['key']}: {m['kind']} ({len(m['files'])} files)")
