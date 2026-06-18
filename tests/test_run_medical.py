from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from medical.storage import MedicalCaseDatabase as RealMedicalCaseDatabase
import run_medical


class RunMedicalTests(unittest.TestCase):
    def test_init_dataset_command_returns_success(self) -> None:
        with patch("sys.argv", ["run_medical.py", "init-dataset", "--dataset-root", "D:/YOLO/tmp-ds"]), patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Đã tạo dataset", stdout.getvalue())

    def test_metrics_command_prints_json_metrics(self) -> None:
        with patch(
            "sys.argv",
            ["run_medical.py", "metrics", "--truths", json.dumps([True, False]), "--predictions", json.dumps([True, True])],
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn('"true_positive": 1', stdout.getvalue())

    @patch("run_medical.prepare_medical_training_dataset")
    def test_split_dataset_command_prints_counts(self, prepare_mock) -> None:
        prepare_mock.return_value = type("Summary", (), {"train_count": 7, "val_count": 2, "test_count": 1})()
        with patch("sys.argv", ["run_medical.py", "split-dataset"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Train: 7", stdout.getvalue())

    @patch("run_medical.train_medical_model", return_value=Path("models/trained/skin_cancer_screening_best.pt"))
    def test_train_command_reports_model_path(self, _train_mock) -> None:
        with patch("sys.argv", ["run_medical.py", "train"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Đã lưu model medical", stdout.getvalue())

    @patch("run_medical.run_full_medical_training_pipeline")
    def test_train_all_command_reports_combined_pipeline(self, pipeline_mock) -> None:
        pipeline_mock.return_value = {
            "train_count": 7,
            "val_count": 2,
            "test_count": 1,
            "trained_model_path": Path("models/trained/skin_cancer_screening_best.pt"),
            "validation_metrics": {"map50": 0.81},
        }
        with patch("sys.argv", ["run_medical.py", "train-all"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Model: models\\trained\\skin_cancer_screening_best.pt", stdout.getvalue().replace("/", "\\"))

    def test_show_case_command_prints_case_detail(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            db_path = Path(temp_dir) / "medical.db"
            db = RealMedicalCaseDatabase(db_path)
            case_id = db.save_case(
                patient_code="BN100",
                image_path="source.jpg",
                processed_image_path="overlay.jpg",
                report_json_path="report.json",
                report_md_path="report.md",
                suspected_malignant=False,
                risk_level="low",
                recommendation="Theo doi",
                metadata={"quality_warnings": []},
            )
            with patch("run_medical.MedicalCaseDatabase", side_effect=lambda: RealMedicalCaseDatabase(db_path)), patch(
                "sys.argv", ["run_medical.py", "show-case", "--case-id", str(case_id)]
            ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("BN100", stdout.getvalue())

    def test_delete_case_command_removes_record(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            db_path = Path(temp_dir) / "medical.db"
            db = RealMedicalCaseDatabase(db_path)
            case_id = db.save_case(
                patient_code="BN200",
                image_path="source.jpg",
                processed_image_path="overlay.jpg",
                report_json_path="report.json",
                report_md_path="report.md",
                suspected_malignant=False,
                risk_level="low",
                recommendation="Theo doi",
                metadata={},
            )
            with patch("run_medical.MedicalCaseDatabase", side_effect=lambda: RealMedicalCaseDatabase(db_path)), patch(
                "sys.argv", ["run_medical.py", "delete-case", "--case-id", str(case_id)]
            ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Đã xóa", stdout.getvalue())

    def test_delete_case_command_can_remove_files(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            root = Path(temp_dir)
            db_path = root / "medical.db"
            db = RealMedicalCaseDatabase(db_path)
            image = root / "source.jpg"
            processed = root / "overlay.jpg"
            report_json = root / "report.json"
            report_md = root / "report.md"
            for path in (image, processed, report_json, report_md):
                path.write_text("x", encoding="utf-8")
            case_id = db.save_case(
                patient_code="BN300",
                image_path=str(image),
                processed_image_path=str(processed),
                report_json_path=str(report_json),
                report_md_path=str(report_md),
                suspected_malignant=False,
                risk_level="low",
                recommendation="Theo doi",
                metadata={},
            )
            with patch("run_medical.MedicalCaseDatabase", side_effect=lambda: RealMedicalCaseDatabase(db_path)), patch(
                "sys.argv", ["run_medical.py", "delete-case", "--case-id", str(case_id), "--delete-files"]
            ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Đã xóa các file liên quan", stdout.getvalue())
