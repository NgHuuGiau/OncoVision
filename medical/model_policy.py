from __future__ import annotations

from pathlib import Path
from typing import Any


def iter_medical_runtime_model_paths(config: Any) -> tuple[Path, ...]:
    configured_model_path = Path(config.model_path)
    candidates = [configured_model_path, Path("medical") / configured_model_path.name]
    if config.allow_fallback_model and config.fallback_model_path is not None:
        candidates.append(Path(config.fallback_model_path))
    return tuple(dict.fromkeys(candidates))


def resolve_medical_runtime_model_path(config: Any) -> Path:
    candidates = iter_medical_runtime_model_paths(config)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Chua tim thay model medical. Da thu: "
        + ", ".join(str(path) for path in candidates)
        + ". Hay chay run_medical.py train-all hoac cap nhat config/medical_settings.yaml."
    )
