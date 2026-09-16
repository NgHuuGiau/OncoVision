from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    return value


def write_inference_dashboard(output_dir: str | Path, payload: dict[str, Any]) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / "inference_dashboard.json"
    serialized = json.dumps(_normalize_for_json(payload), ensure_ascii=False, indent=2)
    try:
        report_path.write_text(serialized, encoding="utf-8")
        return report_path
    except (PermissionError, OSError):
        fallback_path = target_dir / f"inference_dashboard_{time.time_ns()}.json"
        fallback_path.write_text(serialized, encoding="utf-8")
        return fallback_path
