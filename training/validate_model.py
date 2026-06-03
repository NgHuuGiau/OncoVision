from __future__ import annotations

import importlib
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from training.model_paths import resolve_data_config_path, resolve_trained_model_path, resolve_model_source
except ModuleNotFoundError:
    from model_paths import resolve_data_config_path, resolve_trained_model_path, resolve_model_source

YOLO = None
ULTRALYTICS_IMPORT_ERROR = None


def _require_yolo():
    global YOLO, ULTRALYTICS_IMPORT_ERROR
    if YOLO is None and ULTRALYTICS_IMPORT_ERROR is None:
        try:
            YOLO = importlib.import_module("ultralytics").YOLO
        except Exception as exc:  # pragma: no cover
            ULTRALYTICS_IMPORT_ERROR = exc
    if YOLO is None:
        raise RuntimeError(f"Khong khoi tao duoc ultralytics/YOLO: {ULTRALYTICS_IMPORT_ERROR}")
    return YOLO


def resolve_validation_model_path():
    return resolve_model_source(resolve_trained_model_path(required=False, fallback="yolo11n.pt"))


def main() -> None:
    model_path = resolve_validation_model_path()
    model = _require_yolo()(str(model_path))
    metrics = model.val(
        data=str(resolve_data_config_path()),
        project=str(Path("runs/val").resolve()),
        name="validation",
    )
    print(metrics)


if __name__ == "__main__":
    main()
