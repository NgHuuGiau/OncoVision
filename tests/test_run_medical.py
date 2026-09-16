from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import patch

import run_medical


class RunMedicalTests(unittest.TestCase):
    def test_cli_exposes_analysis_and_case_management_only(self) -> None:
        parser = run_medical.build_parser()
        command_action = next(action for action in parser._actions if action.dest == "command")
        self.assertEqual(
            set(command_action.choices),
            {"analyze", "validate-image", "status", "report", "history", "show-case", "export-case", "delete-case", "cleanup-output"},
        )
        for command in ("train", "train-all", "train-modality", "split-dataset", "audit-dataset", "active-learning"):
            with self.subTest(command=command), self.assertRaises(SystemExit):
                parser.parse_args([command])

    @patch("run_medical._medical_output_directories", return_value=[Path("output/medical/reports")])
    @patch("run_medical.cleanup_directories")
    def test_cleanup_output_command_reports_summary(self, cleanup_mock, _medical_dirs_mock) -> None:
        cleanup_mock.return_value = type("CleanupSummary", (), {"removed_files": 5, "removed_dirs": 2, "freed_bytes": 4096})()
        with patch("sys.argv", ["run_medical.py", "cleanup-output", "--older-than-days", "7"]), patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout:
            code = run_medical.main()
        self.assertEqual(code, 0)
        self.assertIn("5 tệp", stdout.getvalue())

    @patch("run_medical.get_medical_system_status")
    def test_status_command_prints_inference_and_case_summary(self, status_mock) -> None:
        status_mock.return_value = type(
            "MedicalStatus",
            (),
            {
                "configured_model_path": Path("models/pretrained/medical.pt"),
                "resolved_model_path": None,
                "allow_fallback_model": False,
                "using_fallback_model": False,
                "model_ready": False,
                "model_message": "Thiếu model suy luận",
                "report_files": 0,
                "normalized_files": 0,
                "overlay_files": 0,
                "export_files": 0,
                "case_db_path": Path("output/onco.db"),
                "case_count": 0,
                "screening_targets": (("Ung thư não", True), ("Ung thư thận", False)),
                "analyzed_cancers": ("Ung thư não", "Ung thư thận"),
                "analyzed_modalities": ("MRI",),
            },
        )()
        with patch("sys.argv", ["run_medical.py", "status"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(run_medical.main(), 0)
        self.assertIn("Thiếu model suy luận", stdout.getvalue())
        self.assertIn("chờ model suy luận", stdout.getvalue())
        self.assertNotIn("train-all", stdout.getvalue().lower())
        self.assertNotIn("sẵn sàng train", stdout.getvalue().lower())

    @patch("run_medical.update_case_report_case_id")
    @patch("run_medical.MedicalCaseDatabase")
    @patch("run_medical.MedicalImageAnalyzer")
    def test_analyze_saves_case_and_syncs_case_id(self, analyzer_cls, case_db_cls, update_report) -> None:
        analyzer_cls.return_value.analyze_image.return_value = type(
            "MedicalResult",
            (),
            {
                "patient_code": "BN555",
                "source_image": Path("source.jpg"),
                "normalized_image": Path("normalized.jpg"),
                "processed_image": Path("processed.jpg"),
                "report_json_path": Path("report.json"),
                "report_md_path": Path("report.md"),
                "suspected_malignant": True,
                "risk_level": "high",
                "recommendation": "Cần khám chuyên khoa",
                "average_confidence": 0.9,
                "model_name": "medical.pt",
                "detections": [],
                "quality_warnings": [],
            },
        )()
        case_db_cls.return_value.save_case.return_value = 42
        with patch("sys.argv", ["run_medical.py", "analyze", "--image", "sample.jpg", "--patient-code", "BN555"]), patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout:
            self.assertEqual(run_medical.main(), 0)
        update_report.assert_called_once_with(Path("report.json"), Path("report.md"), case_id=42)
        self.assertIn("Mã ca bệnh: 42", stdout.getvalue())

    def test_show_case_command_prints_case_detail(self) -> None:
        record = type(
            "Case",
            (),
            {
                "case_id": 100,
                "patient_code": "BN100",
                "created_at": "2026-07-01 10:00:00",
                "risk_level": "low",
                "image_path": "source.jpg",
                "processed_image_path": "overlay.jpg",
                "report_json_path": "report.json",
                "report_md_path": "report.md",
                "metadata": {},
            },
        )()
        fake_db = type("CaseDB", (), {"get_case": lambda self, _case_id: record})()
        with patch("run_medical.MedicalCaseDatabase", return_value=fake_db), patch(
            "sys.argv", ["run_medical.py", "show-case", "--case-id", "100"]
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(run_medical.main(), 0)
        self.assertIn("BN100", stdout.getvalue())

    def test_delete_case_command_removes_record(self) -> None:
        fake_db = type("CaseDB", (), {"delete_case": lambda self, _case_id: True})()
        with patch("run_medical.MedicalCaseDatabase", return_value=fake_db), patch(
            "sys.argv", ["run_medical.py", "delete-case", "--case-id", "200"]
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(run_medical.main(), 0)
        self.assertIn("Đã xóa ca bệnh", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
