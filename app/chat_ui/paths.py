from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path


def _get_project_root() -> Path:
    """Lay duong dan project root, hoat dong dung ca khi chay tu exe (cx_Freeze)."""
    if getattr(sys, "frozen", False):
        # Chay tu compiled executable: base_dir la thu muc chua exe
        base_dir = Path(os.path.dirname(sys.executable))
        # Models va modules nam trong _internal/
        internal_dir = base_dir / "_internal"
        if internal_dir.is_dir():
            return internal_dir
        return base_dir
    else:
        # Chay tu script: __file__ la app/chat_ui/paths.py
        return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _get_project_root()
OUTPUT_DIR = PROJECT_ROOT / "output"
ONCO_DB_PATH = OUTPUT_DIR / "onco.db"
CHAT_HISTORY_DB_PATH = ONCO_DB_PATH
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
