from __future__ import annotations

from typing import Any


def build_detection_metadata(result, *, user_prompt: str = "") -> dict[str, Any]:
    metadata = {
        "normalized_image": str(result.normalized_image),
        "average_confidence": result.average_confidence,
        "model_name": result.model_name,
        "detection_count": len(result.detections),
        "detections": [
            {"label": item.label, "confidence": item.confidence, "bbox": list(item.bbox)} for item in result.detections
        ],
        "quality_warnings": result.quality_warnings,
    }
    if user_prompt:
        metadata["user_prompt"] = user_prompt
    return metadata


def build_case_export_payload(record) -> dict[str, Any]:
    return {
        "case_id": record.case_id,
        "patient_code": record.patient_code,
        "source_image": record.image_path,
        "processed_image": record.processed_image_path,
        "report_json_path": record.report_json_path,
        "report_md_path": record.report_md_path,
        "risk_level": record.risk_level,
        "suspected_malignant": record.suspected_malignant,
        "recommendation": record.recommendation,
        "quality_warnings": record.metadata.get("quality_warnings", []),
        "detections": record.metadata.get("detections", []),
        "model_name": record.metadata.get("model_name", "-"),
    }
