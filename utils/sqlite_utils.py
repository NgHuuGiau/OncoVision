from __future__ import annotations

import sqlite3
from pathlib import Path


DEFAULT_SQLITE_TIMEOUT_SECONDS = 5.0
DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 5000


def create_sqlite_connection(
    db_path: str | Path,
    *,
    timeout_seconds: float = DEFAULT_SQLITE_TIMEOUT_SECONDS,
    enable_foreign_keys: bool = False,
) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=timeout_seconds)
    if enable_foreign_keys:
        conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute(f"PRAGMA busy_timeout = {DEFAULT_SQLITE_BUSY_TIMEOUT_MS}")
    return conn
