from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_medical_runtime_model_path(config: Any) -> Path:
    resolved_model_path = Path(config.model_path)
    if resolved_model_path.exists():
        return resolved_model_path

    if config.allow_fallback_model and config.fallback_model_path is not None:
        fallback_model_path = Path(config.fallback_model_path)
        if fallback_model_path.exists():
            return fallback_model_path

    raise FileNotFoundError(
        f"Chua tim thay model medical tai {resolved_model_path}. "
        "Hay chay run_medical.py train-all hoac cap nhat config/medical_settings.yaml."
    )
