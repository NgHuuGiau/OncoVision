from __future__ import annotations

from pathlib import Path

try:
    from training._bootstrap import ensure_project_root_on_path
    from training.model_artifacts import resolve_trained_model_path
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root_on_path
    from model_artifacts import resolve_trained_model_path

ensure_project_root_on_path()

from ultralytics import YOLO


def resolve_validation_model_path():
    return resolve_trained_model_path(required=False, fallback="yolo11n.pt")


def main() -> None:
    model_path = resolve_validation_model_path()
    model = YOLO(str(model_path))
    metrics = model.val(data="training/data.yaml", project="runs/val", name="validation")
    print(metrics)


if __name__ == "__main__":
    main()
