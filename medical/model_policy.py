from __future__ import annotations

from pathlib import Path
from typing import Any

from core.model_catalog import YOLO11_MODELS_ASC
from training.model_paths import PRETRAINED_MODELS_DIR, resolve_model_source


def is_generic_pretrained_model(model_path: Path) -> bool:
    resolved_model = model_path.resolve(strict=False)
    pretrained_root = PRETRAINED_MODELS_DIR.resolve(strict=False)
    return resolved_model.name in YOLO11_MODELS_ASC and pretrained_root in resolved_model.parents


def resolve_medical_runtime_model_path(config: Any) -> Path:
    resolved_model_path = Path(resolve_model_source(config.model_path))
    if resolved_model_path.exists():
        if not config.allow_fallback_model and is_generic_pretrained_model(resolved_model_path):
            raise FileNotFoundError(
                "Model medical hien tai dang tro vao model YOLO tong quat. "
                "Hay train model chuyen dung va cap nhat config/medical_settings.yaml. "
                "Chi bat allow_fallback_model khi ban chap nhan day la che do nghien cuu."
            )
        return resolved_model_path

    if config.allow_fallback_model and config.fallback_model_path is not None:
        fallback_model_path = Path(resolve_model_source(config.fallback_model_path))
        if fallback_model_path.exists():
            return fallback_model_path

    raise FileNotFoundError(
        f"Chua tim thay model y duoc chuyen dung tai {resolved_model_path}. "
        "Hay huan luyen bang run_medical.py train-all hoac cap nhat config/medical_settings.yaml."
    )


def resolve_medical_base_model(settings: dict[str, Any]) -> Path:
    base_model = str(settings.get("base_model", settings.get("fallback_model", "yolo11n.pt")))
    return Path(resolve_model_source(base_model))
