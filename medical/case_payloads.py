from __future__ import annotations

from typing import Any


def case_localization_available(metadata: dict[str, Any]) -> bool:
    if "localization_available" in metadata:
        return bool(metadata["localization_available"])
    return bool(metadata.get("volume_findings")) or any(
        isinstance(item, dict)
        and (
            (
                isinstance(item.get("bbox"), (list, tuple))
                and len(item["bbox"]) == 4
                and any(item["bbox"])
            )
            or bool(item.get("segmentation"))
        )
        for item in metadata.get("detections", [])
    )


def case_detections(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    has_localization = case_localization_available(metadata)
    return [
        {**item, "bbox": item.get("bbox") if has_localization else None}
        for item in metadata.get("detections", [])
        if isinstance(item, dict)
    ]


def build_detection_metadata(
    result,
    *,
    user_prompt: str = "",
    requested_target: str | None = None,
    requested_modality: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "normalized_image": str(result.normalized_image),
        "average_confidence": result.average_confidence,
        "model_name": result.model_name,
        "detection_count": len(result.detections),
        "clinical_assessment_available": False,
        "localization_available": bool(getattr(result, "volume_findings", []))
        or any(item.bbox or item.segmentation for item in result.detections),
        "detections": [
            {
                "label": item.label,
                "confidence": item.confidence,
                "bbox": list(item.bbox) if item.bbox else None,
                "segmentation": [list(point) for point in item.segmentation] if item.segmentation else None,
                "frame_index": item.frame_index,
                "timestamp_seconds": item.timestamp_seconds,
            }
            for item in result.detections
        ],
        "quality_warnings": result.quality_warnings,
        "modality": getattr(result, "modality", None),
        "body_region": getattr(result, "body_region", None),
        "input_dimension": getattr(result, "input_dimension", "2d"),
        "volume_findings": getattr(result, "volume_findings", []),
        "volume_shape": getattr(result, "volume_shape", None),
        "volume_spacing_xyz_mm": getattr(result, "volume_spacing_xyz_mm", None),
        "volume_coordinate_system": getattr(result, "volume_coordinate_system", None),
        "segmentation_mask_path": (
            str(result.segmentation_mask_path) if getattr(result, "segmentation_mask_path", None) else None
        ),
        "analyzed_frame_count": getattr(result, "analyzed_frame_count", 1),
        "representative_frame_index": getattr(result, "representative_frame_index", None),
        "volume_mask_path": (
            str(result.volume_mask_path) if getattr(result, "volume_mask_path", None) else None
        ),
    }
    if user_prompt:
        metadata["user_prompt"] = user_prompt
    if requested_target:
        metadata["requested_target"] = requested_target
    if requested_modality:
        metadata["requested_modality"] = requested_modality
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
        "review_status": record.review_status,
        "clinical_assessment_available": record.review_status in {"submitted", "approved"},
        "reviewed_by": record.reviewed_by,
        "reviewed_at": record.reviewed_at,
        "quality_warnings": record.metadata.get("quality_warnings", []),
        "detections": case_detections(record.metadata),
        "localization_available": case_localization_available(record.metadata),
        "input_dimension": record.metadata.get("input_dimension", "2d"),
        "volume_findings": record.metadata.get("volume_findings", []),
        "volume_shape": record.metadata.get("volume_shape"),
        "volume_spacing_xyz_mm": record.metadata.get("volume_spacing_xyz_mm"),
        "volume_coordinate_system": record.metadata.get("volume_coordinate_system"),
        "segmentation_mask_path": record.metadata.get("segmentation_mask_path"),
        "analyzed_frame_count": record.metadata.get("analyzed_frame_count", 1),
        "representative_frame_index": record.metadata.get("representative_frame_index"),
        "volume_mask_path": record.metadata.get("volume_mask_path"),
        "model_name": record.metadata.get("model_name", "-"),
    }
