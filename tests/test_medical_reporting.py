from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from medical.reporting import update_case_report_case_id, write_case_report


class MedicalReportingTests(unittest.TestCase):
    def test_write_case_report_uses_unique_filenames(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            payload = {
                "case_id": None,
                "risk_level": "low",
                "suspected_malignant": False,
                "model_name": "best.pt",
                "source_image": "source.jpg",
                "processed_image": "overlay.jpg",
                "recommendation": "Theo doi",
                "quality_warnings": [],
                "detections": [],
            }

            first_json, first_md = write_case_report(temp_dir, payload)
            second_json, second_md = write_case_report(temp_dir, payload)

        self.assertNotEqual(first_json.name, second_json.name)
        self.assertNotEqual(first_md.name, second_md.name)

    def test_update_case_report_case_id_rewrites_json_and_markdown(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            payload = {
                "case_id": None,
                "risk_level": "medium",
                "suspected_malignant": True,
                "model_name": "best.pt",
                "source_image": "source.jpg",
                "processed_image": "overlay.jpg",
                "recommendation": "Tai kham",
                "quality_warnings": ["Anh mo"],
                "detections": [{"label": "lesion", "confidence": 0.6, "bbox": [1, 2, 3, 4]}],
            }
            report_json, report_md = write_case_report(Path(temp_dir), payload)

            update_case_report_case_id(report_json, report_md, case_id=17)

            synced_payload = json.loads(report_json.read_text(encoding="utf-8"))
            markdown = report_md.read_text(encoding="utf-8")

        self.assertEqual(synced_payload["case_id"], 17)
        self.assertIn("Case ID: 17", markdown)
