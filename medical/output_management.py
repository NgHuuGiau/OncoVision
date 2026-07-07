from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from utils.file_utils import load_yaml


@lru_cache(maxsize=1)
def _medical_output_directories() -> list[Path]:
    settings = load_yaml("config/medical_settings.yaml").get("medical", {})
    return [
        Path(settings.get("reports_dir", "output/medical/reports")),
        Path(settings.get("processed_dir", "output/medical/normalized_images")),
        Path(settings.get("overlay_dir", "output/medical/processed_images")),
        Path(settings.get("output_root", "output/medical")) / "exports",
    ]
