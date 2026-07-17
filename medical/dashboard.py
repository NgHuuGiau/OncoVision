from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


TRAINING_PROGRESS_PATH = Path("output/medical/reports/training_progress.json")


def write_training_progress(*, backend: str | None = None, tag: str | None = None, **fields: Any) -> None:
    """Ghi tien do huan luyen ra file JSON de theo doi tu ben ngoai.

    Ho tro ca hai backend:
    - cnn: theo epoch (truyen epoch, num_epochs, train_loss, val_loss, val_acc, lr, best_val_acc).
    - centroid: theo so anh (truyen processed, total).
    Chi giu lai cac truong khac None de ghi chuong trinh goi.
    """
    try:
        path = TRAINING_PROGRESS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {key: value for key, value in fields.items() if value is not None}
        history: list[dict[str, Any]] = []
        if path.exists():
            try:
                history = json.loads(path.read_text(encoding="utf-8")).get("history", [])
            except (ValueError, OSError):
                history = []
        history.append(record)
        payload = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "backend": backend,
            "tag": tag,
            "current": record,
            "history": history[-300:],
        }
        path.write_text(json.dumps(_normalize_for_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError):
        pass


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    return value


def _build_training_summary(payload: dict[str, Any]) -> dict[str, Any]:
    history = payload.get("history", {})
    accuracy = payload.get("accuracy")
    val_acc = history.get("val_acc") or []
    train_loss = history.get("train_loss") or []
    summary = {
        "accuracy": accuracy,
        "best_val_accuracy": max(val_acc) if val_acc else None,
        "final_train_loss": train_loss[-1] if train_loss else None,
        "final_val_accuracy": val_acc[-1] if val_acc else None,
        "epochs": max(len(val_acc), len(train_loss)),
        "confusion_matrix": payload.get("confusion_matrix"),
        "top_k_predictions": payload.get("top_k_predictions", []),
        "low_confidence_cases": payload.get("low_confidence_cases", []),
    }
    return summary


def write_training_dashboard(output_dir: str | Path, payload: dict[str, Any]) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    enriched = dict(payload)
    enriched["summary"] = _build_training_summary(payload)
    report_path = target_dir / "training_dashboard.json"
    serialized = json.dumps(_normalize_for_json(enriched), ensure_ascii=False, indent=2)
    try:
        report_path.write_text(serialized, encoding="utf-8")
        return report_path
    except (PermissionError, OSError):
        fallback_path = target_dir / f"training_dashboard_{time.time_ns()}.json"
        fallback_path.write_text(serialized, encoding="utf-8")
        return fallback_path


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
