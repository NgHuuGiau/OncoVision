from __future__ import annotations

from pathlib import Path


TRAINED_BEST_MODEL_PATH = Path("models/trained/best.pt")


def resolve_trained_model_path(*, required: bool, fallback: str | None = None) -> Path:
    if TRAINED_BEST_MODEL_PATH.exists():
        return TRAINED_BEST_MODEL_PATH
    if fallback is not None:
        return Path(fallback)
    if required:
        raise FileNotFoundError(f"Khong tim thay {TRAINED_BEST_MODEL_PATH}")
    return TRAINED_BEST_MODEL_PATH
