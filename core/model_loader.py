from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Tuple

YOLO = None
ULTRALYTICS_IMPORT_ERROR = None

from core.model_selector import RuntimeConfig
from utils.file_utils import load_yaml_cached
from utils.logger import get_logger


logger = get_logger(__name__)
MODEL_CONFIG_PATH = Path("config/model_config.yaml")


@dataclass
class LoadedModel:
    model: Any
    model_name: str
    source_path: str


def _require_yolo() -> Any:
    global YOLO, ULTRALYTICS_IMPORT_ERROR
    if YOLO is None and ULTRALYTICS_IMPORT_ERROR is None:
        try:
            YOLO = importlib.import_module("ultralytics").YOLO
        except Exception as exc:  # pragma: no cover
            ULTRALYTICS_IMPORT_ERROR = exc
    if YOLO is None:
        raise RuntimeError(f"Không khởi tạo được ultralytics/YOLO: {ULTRALYTICS_IMPORT_ERROR}")
    return YOLO


def _candidate_paths(model_name: str) -> list[str]:
    trained_path = Path("models/trained/best.pt")
    pretrained_path = Path("models/pretrained") / model_name
    local_root_path = Path(model_name)
    configured_priority = load_yaml_cached(str(MODEL_CONFIG_PATH)).get("priority_order", [])
    path_map = {
        "models/trained/best.pt": trained_path,
        f"models/pretrained/{model_name}": pretrained_path,
        model_name: local_root_path,
    }
    candidates: list[str] = []

    for configured_item in configured_priority:
        configured_str = str(configured_item)
        candidate_path = path_map.get(configured_str)
        if candidate_path is not None and candidate_path.exists():
            candidates.append(str(candidate_path))

    for candidate_path in path_map.values():
        if candidate_path.exists():
            candidates.append(str(candidate_path))
    return list(dict.fromkeys(candidates))


def load_yolo_model(runtime: RuntimeConfig) -> Tuple[LoadedModel, str]:
    yolo_cls = None
    errors = []
    for model_name in runtime.candidate_models:
        candidate_paths = _candidate_paths(model_name)
        if not candidate_paths:
            errors.append(f"{model_name}: không tìm thấy file model local")
            logger.warning("No local file found for model %s", model_name)
            continue
        if yolo_cls is None:
            yolo_cls = _require_yolo()
        for candidate in candidate_paths:
            try:
                logger.info("Trying model candidate: %s", candidate)
                model = yolo_cls(candidate)
                runtime.active_model_name = model_name
                return LoadedModel(model=model, model_name=model_name, source_path=candidate), runtime.resolved_device
            except Exception as exc:  # pragma: no cover
                errors.append(f"{candidate}: {exc}")
                logger.warning("Failed to load %s: %s", candidate, exc)
    raise RuntimeError(
        "Không thể load bất kỳ model local nào. "
        "Hãy kiểm tra models/pretrained hoặc models/trained.\n" + "\n".join(errors)
    )
