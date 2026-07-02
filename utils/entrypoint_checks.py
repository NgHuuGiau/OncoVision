from __future__ import annotations

import importlib.util
from pathlib import Path

from medical.pipeline import build_default_medical_analyzer_config, validate_medical_analyzer_config
from medical.system_status import get_medical_system_status
from utils.file_utils import ensure_project_directories, load_yaml, yaml_file_issues


SETTINGS_PATH = Path("config/settings.yaml")
MODEL_CONFIG_PATH = Path("config/model_config.yaml")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def count_files(path: Path, *, suffix: str | None = None) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and (suffix is None or item.suffix == suffix))


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def chat_preflight_status() -> tuple[dict[str, bool], int, bool, str, Path]:
    ensure_project_directories()
    capture_dir = ensure_directory(Path("output/chat_captures"))
    medical_status = get_medical_system_status()
    required_modules = {
        "PySide6": module_available("PySide6"),
        "cv2": module_available("cv2"),
        "numpy": module_available("numpy"),
        "ultralytics": module_available("ultralytics"),
        "torch": module_available("torch"),
    }
    icons_count = count_files(Path("assets/icons"), suffix=".svg")
    return required_modules, icons_count, medical_status.model_ready, medical_status.model_message, capture_dir


def medical_config_issues() -> list[str]:
    try:
        config = build_default_medical_analyzer_config()
    except Exception as exc:
        return [str(exc)]
    return validate_medical_analyzer_config(config)


def runtime_config_issues() -> list[str]:
    issues: list[str] = []
    issues.extend(
        yaml_file_issues(
            SETTINGS_PATH,
            required_keys=("models", "inference", "camera"),
            label="config/settings.yaml",
        )
    )
    issues.extend(
        yaml_file_issues(
            MODEL_CONFIG_PATH,
            required_keys=("preferred_models", "priority_order"),
            label="config/model_config.yaml",
        )
    )

    settings = load_yaml(SETTINGS_PATH)
    model_config = load_yaml(MODEL_CONFIG_PATH)
    if not isinstance(settings, dict) or not isinstance(model_config, dict):
        return issues

    models = settings.get("models")
    if not isinstance(models, dict) or not models:
        issues.append("config/settings.yaml thiếu hoặc sai mục `models`.")
    else:
        required_modes = {"high", "medium", "low"}
        missing_modes = sorted(required_modes - set(models))
        if missing_modes:
            issues.append("Thiếu các mode: " + ", ".join(missing_modes))
        for mode_name, mode_config in models.items():
            if not isinstance(mode_config, dict):
                issues.append(f"Mode `{mode_name}` phải là mapping.")
                continue
            if "model" not in mode_config:
                issues.append(f"Mode `{mode_name}` thiếu trường `model`.")
            if "imgsz" not in mode_config:
                issues.append(f"Mode `{mode_name}` thiếu trường `imgsz`.")

    preferred_models = model_config.get("preferred_models")
    if not isinstance(preferred_models, dict):
        issues.append("config/model_config.yaml thiếu mục `preferred_models`.")
    priority_order = model_config.get("priority_order")
    if not isinstance(priority_order, list) or not priority_order:
        issues.append("config/model_config.yaml thiếu mục `priority_order`.")
    return issues
