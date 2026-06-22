from __future__ import annotations

import importlib.util
from pathlib import Path

from medical.system_status import get_medical_system_status
from utils.file_utils import ensure_project_directories


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
