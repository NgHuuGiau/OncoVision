from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from medical.chat_service import MedicalChatService
from medical.pipeline import MedicalAnalysisResult, DetectionFinding
from medical.storage import MedicalCaseDatabase


class _FakeAnalyzer:
    def __init__(self, result: MedicalAnalysisResult):
        self.result = result

    def ensure_ready(self):
        return Path("models/trained/fake.pt")

    def analyze_image(self, image_path, *, patient_code: str, case_id=None):
        return self.result


class MedicalChatServiceTests(unittest.TestCase):
    def test_analyze_attachment_returns_reply_and_persists_case(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report_json = Path(temp_dir) / "report.json"
            report_md = Path(temp_dir) / "report.md"
            overlay = Path(temp_dir) / "overlay.jpg"
            source = Path(temp_dir) / "source.jpg"
            normalized = Path(temp_dir) / "normalized.jpg"
            report_json.write_text(
                json.dumps(
                    {
                        "case_id": None,
                        "risk_level": "high",
                        "suspected_malignant": True,
                        "model_name": "best.pt",
                        "source_image": str(source),
                        "processed_image": str(overlay),
                        "recommendation": "Can kham chuyen khoa",
                        "quality_warnings": ["Anh hoi mo"],
                        "detections": [{"label": "lesion", "confidence": 0.91, "bbox": [1, 2, 3, 4]}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            report_md.write_text("pending", encoding="utf-8")
            for path in (overlay, source, normalized):
                path.write_text("x", encoding="utf-8")
            result = MedicalAnalysisResult(
                case_id=None,
                patient_code="BN777",
                source_image=source,
                normalized_image=normalized,
                processed_image=overlay,
                report_json_path=report_json,
                report_md_path=report_md,
                detections=[DetectionFinding(label="lesion", confidence=0.91, bbox=(1, 2, 3, 4))],
                risk_level="high",
                suspected_malignant=True,
                recommendation="Can kham chuyen khoa",
                disclaimer="Khong thay the bac si",
                average_confidence=0.91,
                model_name="best.pt",
                quality_warnings=["Anh hoi mo"],
            )
            db = MedicalCaseDatabase(Path(temp_dir) / "cases.db")
            service = MedicalChatService(analyzer=_FakeAnalyzer(result), case_db=db)

            response = service.analyze_attachment(image_path=source, patient_code="BN777", user_prompt="kiem tra")

            metadata = json.loads(response.metadata_json)
            self.assertIn("medical_case_id", metadata)
            self.assertEqual(metadata["risk_level"], "high")
            self.assertEqual(metadata["processed_image_path"], str(overlay))
            self.assertEqual(metadata["report_html_path"], str(report_json.with_suffix(".html")))
            self.assertEqual(response.attachment_path, str(overlay))
            self.assertIn("BN777", response.reply_text)
            self.assertIn("Anh hoi mo", response.reply_text)
            self.assertIn(str(report_json), response.reply_text)
            self.assertIn(str(report_md), response.reply_text)
            synced_payload = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(synced_payload["case_id"], metadata["medical_case_id"])
