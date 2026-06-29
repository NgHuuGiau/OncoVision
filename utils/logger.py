from __future__ import annotations

import logging
from pathlib import Path

from utils.terminal_encoding import ensure_utf8_console


LOG_FILE = Path("output/logs/app.log")


def get_logger(name: str) -> logging.Logger:
    ensure_utf8_console()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    except OSError:
        file_handler = None
    if file_handler is not None:
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
