from __future__ import annotations

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


def build_model_backups() -> dict[str, tuple[str | None, ...]]:
    backups: dict[str, tuple[str | None, ...]] = {}
    for index, model_name in enumerate(YOLO11_MODELS_DESC):
        backups[model_name] = (None, *YOLO11_MODELS_DESC[index + 1 :])
    return backups
