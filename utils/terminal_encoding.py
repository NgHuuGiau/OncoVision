from __future__ import annotations

import ctypes
import os
import sys


def ensure_utf8_console() -> None:
    try:
        if os.name == "nt":
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleCP(65001)
            kernel32.SetConsoleOutputCP(65001)
            os.system("chcp 65001 > nul")
            os.environ.setdefault("PYTHONIOENCODING", "utf-8")
            os.environ.setdefault("PYTHONUTF8", "1")
        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(sys, stream_name, None)
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        return
