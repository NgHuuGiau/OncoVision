from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from medical.cancer_catalog import supported_cancer_labels, supported_cancer_modalities
from medical.case_payloads import build_detection_metadata
from medical.compliance import MEDICAL_DISCLAIMER
from medical.pipeline import MedicalImageAnalyzer
from medical.reporting import update_case_report_case_id
from medical.storage import MedicalCaseDatabase

ProgressCallback = Callable[[str, float], None] | None


@dataclass(frozen=True)
class MedicalChatResponse:
    reply_text: str
    attachment_path: str | None
    attachment_kind: str | None
    metadata_json: str


class MedicalChatService:
    def __init__(
        self,
        analyzer: MedicalImageAnalyzer | None = None,
        case_db: MedicalCaseDatabase | None = None,
    ) -> None:
        self.analyzer = analyzer or MedicalImageAnalyzer()
        self.case_db = case_db or MedicalCaseDatabase()

    def check_ready(self) -> Path:
        return self.analyzer.ensure_ready()

    def analyze_attachment(self, *, image_path: str | Path, patient_code: str, user_prompt: str = "", progress_callback: ProgressCallback = None) -> MedicalChatResponse:
        try:
            result = self.analyzer.analyze_image(image_path, patient_code=patient_code, progress_callback=progress_callback)
        except ValueError as exc:
            message = str(exc)
            if ": " in message:
                error_code, error_message = message.split(": ", 1)
            else:
                error_code = "UNKNOWN_ERROR"
                error_message = message
            reply_text = (
                f"❌ **Lỗi phân tích** — Mã BN: {patient_code}\n\n"
                f"Mã lỗi: {error_code}\n"
                f"Chi tiết: {error_message}\n\n"
                f"Vui lòng tải lên ảnh y khoa hợp lệ (CT, MRI, PET/CT, X-quang, siêu âm, mammogram...).\n\n"
                f"*{MEDICAL_DISCLAIMER}*"
            )
            return MedicalChatResponse(
                reply_text=reply_text,
                attachment_path=None,
                attachment_kind=None,
                metadata_json="{}",
            )
        case_id = self.case_db.save_case(
            patient_code=result.patient_code,
            image_path=str(result.source_image),
            processed_image_path=str(result.processed_image),
            report_json_path=str(result.report_json_path),
            report_md_path=str(result.report_md_path),
            suspected_malignant=result.suspected_malignant,
            risk_level=result.risk_level,
            recommendation=result.recommendation,
            metadata=build_detection_metadata(result, user_prompt=user_prompt),
        )
        update_case_report_case_id(
            result.report_json_path,
            result.report_md_path,
            case_id=case_id,
        )
        gradcam_paths = list(result.gradcam_overlays) if result.gradcam_overlays else []
        dicom_info = result.dicom_info or {}
        metadata = {
            "medical_case_id": case_id,
            "source_image_path": str(result.source_image),
            "risk_level": result.risk_level,
            "suspected_malignant": result.suspected_malignant,
            "processed_image_path": str(result.processed_image),
            "gradcam_overlays": gradcam_paths,
            "dicom_info": dicom_info,
            "report_json_path": str(result.report_json_path),
            "report_md_path": str(result.report_md_path),
            "report_html_path": str(result.report_json_path.with_suffix(".html")),
            "recommendation": result.recommendation,
            "model_name": result.model_name,
            "average_confidence": result.average_confidence,
            "quality_warnings": result.quality_warnings,
            "supported_screening_targets": supported_cancer_labels(),
            "supported_modalities": supported_cancer_modalities(),
            "detections": [{"label": item.label, "confidence": item.confidence, "bbox": list(item.bbox)} for item in result.detections],
            "predicted_labels": [item.label for item in result.detections],
        }
        top_detections = result.detections[:3]
        if result.detections:
            detection_summary = "\n".join(
                f"• {item.label}: {item.confidence*100:.1f}%" for item in top_detections
            )
        else:
            detection_summary = "không ghi nhận vùng nghi ngờ rõ ràng"
        quality_text = (
            f"\n⚠️ Cảnh báo: {'; '.join(result.quality_warnings)}"
            if result.quality_warnings
            else ""
        )
        risk_icon = {"high": "🔴", "medium": "🟡", "low": "🟢", "uncertain": "⚪"}
        risk_text = f"{risk_icon.get(result.risk_level, '⚪')} Mức nguy cơ: {result.risk_level}"
        grad_summary = ""
        if result.gradcam_overlays:
            grad_summary = f"\n📊 Heatmap: {len(result.gradcam_overlays)} ảnh"
        elapsed = result.analysis_time_seconds
        time_str = f"{elapsed:.1f}s" if elapsed > 0 else ""
        reply_text = (
            f"📋 **Kết quả phân tích** — Mã BN: {patient_code}\n\n"
            f"{risk_text}\n"
            f"🔬 Phát hiện: {len(result.detections)} vùng\n"
            f"{detection_summary}\n"
            f"{grad_summary}"
            f"{quality_text}\n\n"
            f"💡 {result.recommendation}\n"
            + (f"⏱ {time_str}\n" if time_str else "")
            + f"\n*{result.disclaimer}*"
        )
        return MedicalChatResponse(
            reply_text=reply_text,
            attachment_path=str(result.processed_image),
            attachment_kind="image",
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )
