from __future__ import annotations

from pathlib import Path

from core.model_catalog import is_allowed_model_reference, is_supported_pretrained_model
from utils.file_utils import load_yaml, save_yaml


TRAINED_BEST_MODEL_PATH = Path("models/trained/best.pt")
PRETRAINED_MODELS_DIR = Path("models/pretrained")
TRAINING_DATA_CONFIG_PATH = Path("training/data.yaml")
GENERATED_DATA_CONFIG_PATH = Path("training/.generated_data.yaml")


def resolve_model_source(model_name: str | Path) -> Path:
    model_path = Path(model_name)
    if not is_allowed_model_reference(model_path):
        raise ValueError(
            f"Unsupported model reference: {model_path}. "
            "Only YOLO11 pretrained models (yolo11n/s/m/l/x.pt) and models/trained/*.pt are allowed."
        )
    if model_path.exists():
        return model_path
    if is_supported_pretrained_model(model_path.name):
        pretrained_path = PRETRAINED_MODELS_DIR / model_path.name
        if pretrained_path.exists():
            return pretrained_path
    return model_path


def resolve_trained_model_path(*, required: bool, fallback: str | None = None) -> Path:
    if TRAINED_BEST_MODEL_PATH.exists():
        return TRAINED_BEST_MODEL_PATH
    if fallback is not None:
        return PRETRAINED_MODELS_DIR / Path(fallback).name
    if required:
        raise FileNotFoundError(f"Không tìm thấy {TRAINED_BEST_MODEL_PATH}")
    return TRAINED_BEST_MODEL_PATH


def resolve_data_config_path() -> Path:
    config = load_yaml(TRAINING_DATA_CONFIG_PATH)
    dataset_root = (TRAINING_DATA_CONFIG_PATH.parent / config["path"]).resolve()
    normalized = dict(config)
    normalized["path"] = str(dataset_root)
    save_yaml(GENERATED_DATA_CONFIG_PATH, normalized)
    return GENERATED_DATA_CONFIG_PATH
