from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from utils.terminal_encoding import ensure_utf8_console

LOG_FILE = Path("output/logs/app.log")
LOG_JSON_FILE = Path("output/logs/app.json.log")


class JsonFormatter(logging.Formatter):
    """Structured JSON logging formatter cho production monitoring.

    Mỗi log line là một JSON object với các trường:
    timestamp, level, logger, message, module, function, line, exception
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data
        return json.dumps(log_entry, ensure_ascii=False, default=str)


def get_logger(name: str, *, json_format: bool = False) -> logging.Logger:
    """Tạo logger với optional JSON format cho production.

    Args:
        name: Tên logger.
        json_format: Nếu True, dùng JSON format cho file log (phù hợp cho log aggregation).
    """
    ensure_utf8_console()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)

    # Console handler — luôn dùng text format cho dễ đọc
    text_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(text_formatter)
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(stream_handler)

    # File handler — text format (mặc định) hoặc JSON format (production)
    file_formatter = JsonFormatter() if json_format else text_formatter

    try:
        target_file = LOG_JSON_FILE if json_format else LOG_FILE
        file_handler = RotatingFileHandler(
            target_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    except OSError:
        pass

    logger.propagate = False
    return logger


def setup_production_logging() -> None:
    """Setup structured logging cho production environment.

    Tự động kích hoạt khi ONCOVISION_LOG_FORMAT=json hoặc ONCOVISION_ENV=production.
    Nên gọi một lần khi khởi động ứng dụng.
    """
    log_format = os.environ.get("ONCOVISION_LOG_FORMAT", "").lower()
    env = os.environ.get("ONCOVISION_ENV", "").lower()

    if log_format == "json" or env in ("prod", "production"):
        # Set JSON formatter cho root logger
        root = logging.getLogger()
        if root.handlers:
            return
        root.setLevel(logging.INFO)
        json_fmt = JsonFormatter()

        LOG_JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            fh = RotatingFileHandler(
                LOG_JSON_FILE, maxBytes=10 * 1024 * 1024, backupCount=10, encoding="utf-8"
            )
            fh.setFormatter(json_fmt)
            fh.setLevel(logging.INFO)
            root.addHandler(fh)
        except OSError:
            pass

        # Console handler vẫn dùng text format
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        console.setLevel(logging.WARNING)
        root.addHandler(console)
