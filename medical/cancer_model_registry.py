"""Registry nhẹ cho artifact model cancers đã tải sẵn (heterogeneous).

Quét `models/pretrained/cancers/<key>/` và trả về artifact + kind,
không import torch/transformers/nnunet ở module load.
# ponytail: registry chỉ tìm file, inference từng kind làm riêng khi cần
"""
from __future__ import annotations

from pathlib import Path

CANCERS_DIR = Path("models/pretrained/cancers")

KEY_TO_DIR = {
    "liver": "liver",
    "lung": "lung",
    "breast": "breast",
    "stomach": "stomach",
    "colorectal": "colorectal",
    "prostate": "prostate",
    "cervical": "cervix",
    "cervix": "cervix",
    "brain": "brain",
    "kidney": "kidney",
    "pancreas": "pancreas",
    "thyroid": "thyroid",
}

_KIND_BY_SUFFIX = {
    ".pt": "torch-state",
    ".pth": "torch-state",
    ".bin": "hf-transformers",
    ".safetensors": "safetensors",
    ".ckpt": "ckpt",
    ".keras": "keras",
}


def _detect_kind(files: list[Path]) -> str:
    kinds = [_KIND_BY_SUFFIX[p.suffix.lower()] for p in files if p.suffix.lower() in _KIND_BY_SUFFIX]
    for preferred in ("torch-state", "hf-transformers", "safetensors", "ckpt", "keras"):
        if preferred in kinds:
            return preferred
    return "unknown"


def find_cancer_model(key: str, base: Path = CANCERS_DIR) -> dict:
    """Trả về artifact có mặt; `runtime_ready` chỉ đúng khi đã có adapter suy luận."""
    norm = KEY_TO_DIR.get(key.strip().lower(), key.strip().lower())
    d = Path(base) / norm
    if not d.is_dir():
        return {"key": key, "dir": str(d), "kind": "missing", "files": [], "artifact_ready": False, "runtime_ready": False,
                "note": "Chưa có thư mục model cho nhóm này."}
    files = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in _KIND_BY_SUFFIX]
    if not files:
        return {"key": key, "dir": str(d), "kind": "empty", "files": [], "artifact_ready": False, "runtime_ready": False,
                "note": "Có thư mục nhưng chưa có weight (.pt/.bin/.safetensors/.ckpt/.keras)."}
    kind = _detect_kind(files)
    rel = sorted(str(p.relative_to(d)) for p in files)
    note = {
        "torch-state": "Nạp bằng torch.load + đúng class kiến trúc (cần config kèm theo).",
        "hf-transformers": "Nạp bằng transformers AutoModel + preprocessor_config.json.",
        "safetensors": "nnUNet/VISTA3D — cần đúng pipeline seg, không phải classifier thuần.",
        "ckpt": "Lightning ckpt — cần class model gốc để load.",
        "keras": "Nạp bằng keras.models.load_model.",
    }.get(kind, "Chưa rõ loader.")
    return {"key": key, "dir": str(d), "kind": kind, "files": rel, "artifact_ready": True, "runtime_ready": False, "note": note}


def list_cancer_models(base: Path = CANCERS_DIR) -> list[dict]:
    seen = set()
    out = []
    for key in list(KEY_TO_DIR):
        norm = KEY_TO_DIR[key]
        if norm in seen:
            continue
        seen.add(norm)
        out.append(find_cancer_model(norm, base))
    return out


if __name__ == "__main__":
    for m in list_cancer_models():
        print(f"- {m['key']}: {m['kind']} ({len(m['files'])} files)")
