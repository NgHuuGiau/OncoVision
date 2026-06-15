from __future__ import annotations

import time
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
CHAT_HISTORY_DB_PATH = OUTPUT_DIR / "chat_history.db"
CHAT_CAPTURES_DIR = OUTPUT_DIR / "chat_captures"


def get_chat_capture_dir(*, ensure_exists: bool = True) -> Path:
    if ensure_exists:
        CHAT_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    return CHAT_CAPTURES_DIR


def build_chat_capture_path(*, base_dir: str | Path | None = None) -> Path:
    target_dir = Path(base_dir) if base_dir is not None else get_chat_capture_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    token = uuid.uuid4().hex[:10]
    return target_dir / f"camera_capture_{timestamp}_{token}.png"
