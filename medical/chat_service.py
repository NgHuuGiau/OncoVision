from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from medical.pipeline import MedicalImageAnalyzer
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

    def analyze_attachment(self, *, image_path: str | Path, patient_code: str, user_prompt: str = "") -> MedicalChatResponse:
        self.check_ready()
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
            metadata={
                "normalized_image": str(result.normalized_image),
                "average_confidence": result.average_confidence,
                "model_name": result.model_name,
                "detection_count": len(result.detections),
                "detections": [
                    {"label": item.label, "confidence": item.confidence, "bbox": list(item.bbox)} for item in result.detections
                ],
                "user_prompt": user_prompt,
            },
        )
        metadata = {
            "medical_case_id": case_id,
            "risk_level": result.risk_level,
            "suspected_malignant": result.suspected_malignant,
            "report_json_path": str(result.report_json_path),
            "report_md_path": str(result.report_md_path),
            "recommendation": result.recommendation,
            "model_name": result.model_name,
            "average_confidence": result.average_confidence,
            "quality_warnings": result.quality_warnings,
        }
        detection_summary = ", ".join(
            f"{item.label} {item.confidence:.2f}" for item in result.detections[:5]
        ) or "khong ghi nhan vung nghi ngo ro rang"
        quality_text = (
            " Canh bao chat luong anh: " + "; ".join(result.quality_warnings)
            if result.quality_warnings
            else ""
        )
        reply_text = (
            f"Da phan tich anh y khoa cho ma benh nhan {patient_code}. "
            f"Muc do sang loc nguy co: {result.risk_level}. "
            f"So vung ton thuong ghi nhan: {len(result.detections)}. "
            f"Tom tat phat hien: {detection_summary}. "
            f"Khuyen nghi: {result.recommendation}."
            f"{quality_text} "
            f"Luu y: {result.disclaimer}"
        )
        return MedicalChatResponse(
            reply_text=reply_text,
            attachment_path=str(result.processed_image),
            attachment_kind="image",
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )
