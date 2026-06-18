from __future__ import annotations

import json
import zipfile
from pathlib import Path
from time import strftime
from typing import Any

from medical.compliance import build_medical_disclaimer


def write_case_report(output_dir: str | Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = strftime("%Y%m%d_%H%M%S")
    json_path = target_dir / f"case_report_{stamp}.json"
    md_path = target_dir / f"case_report_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(payload), encoding="utf-8")
    return json_path, md_path


def export_case_bundle(case_payload: dict[str, Any], export_dir: str | Path, *, include_files: list[str] | None = None) -> Path:
    target_dir = Path(export_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    case_id = case_payload.get("case_id", "unknown")
    stamp = strftime("%Y%m%d_%H%M%S")
    bundle_path = target_dir / f"medical_case_{case_id}_{stamp}.zip"
    include_files = include_files or []
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("case_summary.json", json.dumps(case_payload, ensure_ascii=False, indent=2))
        archive.writestr("case_summary.md", _markdown_report(case_payload))
        for path_str in include_files:
            path = Path(path_str)
            if path.exists() and path.is_file():
                archive.write(path, arcname=path.name)
    return bundle_path


def _markdown_report(payload: dict[str, Any]) -> str:
    detections = payload.get("detections", [])
    quality_warnings = payload.get("quality_warnings", [])
    detection_lines = "\n".join(
        f"- {item['label']} | conf={item['confidence']:.2f} | bbox={item['bbox']}" for item in detections
    ) or "- Khong co vung nghi ngo nao duoc ghi nhan."
    quality_lines = "\n".join(f"- {warning}" for warning in quality_warnings) or "- Khong co canh bao chat luong anh."
    return (
        "# Medical Imaging Case Report\n\n"
        f"- Case ID: {payload.get('case_id', '-')}\n"
        f"- Risk level: {payload.get('risk_level', '-')}\n"
        f"- Suspected malignant: {payload.get('suspected_malignant', False)}\n"
        f"- Model: {payload.get('model_name', '-')}\n"
        f"- Source image: {payload.get('source_image', '-')}\n"
        f"- Processed image: {payload.get('processed_image', '-')}\n\n"
        "## Findings\n"
        f"{detection_lines}\n\n"
        "## Image Quality\n"
        f"{quality_lines}\n\n"
        "## Recommendation\n"
        f"{payload.get('recommendation', '-')}\n\n"
        "## Legal Notice\n"
        f"{build_medical_disclaimer()}\n"
    )
