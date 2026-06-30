from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from medical.case_payloads import build_detection_metadata
from medical.cancer_catalog import supported_cancer_labels
from medical.pipeline import MedicalImageAnalyzer
from medical.reporting import update_case_report_case_id
from medical.storage import MedicalCaseDatabase


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

    def _build_case_metadata(self, result, *, user_prompt: str) -> dict[str, object]:
        return build_detection_metadata(result, user_prompt=user_prompt)

    def analyze_attachment(self, *, image_path: str | Path, patient_code: str, user_prompt: str = "") -> MedicalChatResponse:
        result = self.analyzer.analyze_image(image_path, patient_code=patient_code)
        case_id = self.case_db.save_case(
            patient_code=result.patient_code,
            image_path=str(result.source_image),
            processed_image_path=str(result.processed_image),
            report_json_path=str(result.report_json_path),
            report_md_path=str(result.report_md_path),
            suspected_malignant=result.suspected_malignant,
            risk_level=result.risk_level,
            recommendation=result.recommendation,
            metadata=self._build_case_metadata(result, user_prompt=user_prompt),
        )
        update_case_report_case_id(
            result.report_json_path,
            result.report_md_path,
            case_id=case_id,
        )
        metadata = {
            "medical_case_id": case_id,
            "source_image_path": str(result.source_image),
            "risk_level": result.risk_level,
            "suspected_malignant": result.suspected_malignant,
            "processed_image_path": str(result.processed_image),
            "report_json_path": str(result.report_json_path),
            "report_md_path": str(result.report_md_path),
            "recommendation": result.recommendation,
            "model_name": result.model_name,
            "average_confidence": result.average_confidence,
            "quality_warnings": result.quality_warnings,
            "supported_screening_targets": supported_cancer_labels(),
            "predicted_labels": [item.label for item in result.detections],
        }
        detection_summary = ", ".join(
            f"{item.label} {item.confidence:.2f}" for item in result.detections[:5]
        ) or "không ghi nhận vùng nghi ngờ rõ ràng"
        target_summary = ", ".join(supported_cancer_labels())
        quality_text = (
            " Cảnh báo chất lượng ảnh: " + "; ".join(result.quality_warnings)
            if result.quality_warnings
            else ""
        )
        reply_text = (
            f"Đã phân tích ảnh y khoa cho mã bệnh nhân {patient_code}. "
            f"Mức độ sàng lọc nguy cơ: {result.risk_level}. "
            f"Số vùng tổn thương ghi nhận: {len(result.detections)}. "
            f"Tóm tắt phát hiện: {detection_summary}. "
            f"Hệ thống hiện hỗ trợ danh mục sàng lọc: {target_summary}. "
            f"Khuyến nghị: {result.recommendation}."
            f"{quality_text} "
            f"Ảnh gốc: {result.source_image}. "
            f"Đã lưu ảnh đã xử lý tại: {result.processed_image}. "
            f"Đã lưu báo cáo JSON tại: {result.report_json_path}. "
            f"Đã lưu báo cáo Markdown tại: {result.report_md_path}. "
            f"Lưu ý: {result.disclaimer}"
        )
        return MedicalChatResponse(
            reply_text=reply_text,
            attachment_path=str(result.processed_image),
            attachment_kind="image",
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )
