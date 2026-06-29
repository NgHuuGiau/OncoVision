from __future__ import annotations

from pathlib import Path

YOLO11_MODELS_ASC = (
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
)

YOLO11_MODELS_DESC = tuple(reversed(YOLO11_MODELS_ASC))
DEFAULT_MODEL_FALLBACK = YOLO11_MODELS_ASC[0]

MODEL_QUALITY_SCORES = {
    "yolo11x.pt": 100,
    "yolo11l.pt": 92,
    "yolo11m.pt": 84,
    "yolo11s.pt": 74,
    "yolo11n.pt": 58,
}


def is_supported_pretrained_model(model_name: str | Path) -> bool:
    return Path(model_name).name in YOLO11_MODELS_ASC


def is_allowed_model_reference(model_name: str | Path) -> bool:
    model_path = Path(model_name)
    normalized = str(model_path).replace("\\", "/")
    if normalized == "models/trained/best.pt":
        return True
    if normalized.startswith("models/trained/"):
        return model_path.suffix.lower() == ".pt"
    if model_path.suffix.lower() == ".pt" and (
        model_path.is_absolute() or model_path.parent != Path(".")
    ):
        return True
    return is_supported_pretrained_model(model_path.name)


def build_model_backups() -> dict[str, tuple[str | None, ...]]:
    backups: dict[str, tuple[str | None, ...]] = {}
    for index, model_name in enumerate(YOLO11_MODELS_DESC):
        backups[model_name] = (None, *YOLO11_MODELS_DESC[index + 1 :])
    return backups
