from __future__ import annotations

from pathlib import Path

from app.chat_ui.paths import get_chat_capture_dir
from utils.cleanup_utils import CleanupSummary, cleanup_directories


def chat_output_directories(*, base_dir: str | Path | None = None) -> list[Path]:
    return [Path(base_dir) if base_dir is not None else get_chat_capture_dir()]


def cleanup_chat_outputs(*, older_than_days: int | None = None, base_dir: str | Path | None = None) -> CleanupSummary:
    return cleanup_directories(
        chat_output_directories(base_dir=base_dir),
        older_than_days=older_than_days,
    )
