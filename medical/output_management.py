from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.chat_ui.paths import PROJECT_ROOT
from utils.file_utils import load_yaml


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def _medical_output_directories() -> list[Path]:
    settings = load_yaml(_project_path("config/medical_settings.yaml")).get("medical", {})
    return [
        _project_path(settings.get("reports_dir", "output/medical/reports")),
        _project_path(settings.get("processed_dir", "output/medical/normalized_images")),
        _project_path(settings.get("overlay_dir", "output/medical/processed_images")),
        _project_path(settings.get("output_root", "output/medical")) / "exports",
    ]
