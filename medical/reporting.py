from __future__ import annotations

from datetime import datetime
import json
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from medical.compliance import build_medical_disclaimer
from medical.cancer_catalog import supported_cancer_labels


def build_artifact_stamp() -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"


def _write_case_report_files(json_path: Path, md_path: Path, payload: dict[str, Any]) -> None:
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(payload), encoding="utf-8")


def write_case_report(output_dir: str | Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = build_artifact_stamp()
    json_path = target_dir / f"case_report_{stamp}.json"
    md_path = target_dir / f"case_report_{stamp}.md"
    _write_case_report_files(json_path, md_path, payload)
    return json_path, md_path


def update_case_report_case_id(json_path: str | Path, md_path: str | Path, *, case_id: int) -> None:
    json_file = Path(json_path)
    md_file = Path(md_path)
    payload = json.loads(json_file.read_text(encoding="utf-8"))
    payload["case_id"] = case_id
    _write_case_report_files(json_file, md_file, payload)


def export_case_bundle(case_payload: dict[str, Any], export_dir: str | Path, *, include_files: list[str] | None = None) -> Path:
    target_dir = Path(export_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    case_id = case_payload.get("case_id", "unknown")
    stamp = build_artifact_stamp()
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
    supported_targets = ", ".join(supported_cancer_labels())
    return (
        "# Medical Imaging Case Report\n\n"
        f"- Case ID: {payload.get('case_id', '-')}\n"
        f"- Risk level: {payload.get('risk_level', '-')}\n"
        f"- Suspected malignant: {payload.get('suspected_malignant', False)}\n"
        f"- Model: {payload.get('model_name', '-')}\n"
        f"- Supported screening targets: {supported_targets}\n"
        f"- Source image: {payload.get('source_image', '-')}\n"
        f"- Normalized image: {payload.get('normalized_image', '-')}\n"
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
