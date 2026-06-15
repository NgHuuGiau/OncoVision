from __future__ import annotations

import os
import sys


def ensure_utf8_console() -> None:
    try:
        if os.name == "nt":
            os.system("chcp 65001 > nul")
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name, None)
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        return
