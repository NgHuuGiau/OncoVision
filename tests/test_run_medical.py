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
        self.assertIn("Da tao dataset", stdout.getvalue())

    def test_metrics_command_prints_json_metrics(self) -> None:
        with patch(
            "sys.argv",
            ["run_medical.py", "metrics", "--truths", json.dumps([True, False]), "--predictions", json.dumps([True, True])],
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn('"true_positive": 1', stdout.getvalue())

    @patch("run_medical.cleanup_medical_outputs")
    def test_cleanup_output_command_reports_summary(self, cleanup_mock) -> None:
        cleanup_mock.return_value = type(
            "CleanupSummary",
            (),
            {"removed_files": 5, "removed_dirs": 2, "freed_bytes": 4096},
        )()
        with patch("sys.argv", ["run_medical.py", "cleanup-output", "--older-than-days", "7"]), patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Da xoa file: 5", stdout.getvalue())

    @patch("run_medical.recommended_medical_commands", return_value=["python run_medical.py train-all"])
    @patch(
        "run_medical.tcia_counts",
        return_value=type(
            "TciaCounts",
            (),
            {
                "downloaded_total": 1234,
                "target_total": 25000,
                "remaining_to_target": 23766,
                "__getitem__": lambda self, key: getattr(self, key),
            },
        )(),
    )
    @patch("run_medical.get_medical_system_status")
    def test_status_command_prints_medical_summary(self, status_mock, _tcia_counts_mock, recommended_mock) -> None:
        status_mock.return_value = type(
            "MedicalStatus",
            (),
            {
                "configured_model_path": Path("models/trained/skin.pt"),
                "resolved_model_path": None,
                "allow_fallback_model": False,
                "model_ready": False,
                "model_message": "missing model",
                "dataset_root": Path("dataset/medical/skin_lesion"),
                "data_yaml_path": Path("dataset/medical/skin_lesion/data.yaml"),
                "dataset_initialized": True,
                "raw_images": 0,
                "raw_labels": 0,
                "train_images": 0,
                "val_images": 0,
                "test_images": 0,
                "case_count": 0,
                "report_files": 0,
                "normalized_files": 0,
                "overlay_files": 0,
                "export_files": 0,
                "analyzed_cancers": ("Ung thu da", "Ung thu vu"),
            },
        )()
        with patch("sys.argv", ["run_medical.py", "status"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Trạng thái hệ thống y", output)
        self.assertIn("missing model", output)
        self.assertIn("Hệ thống đang phân tích các ung thư:", output)
        self.assertIn("TCIA cancer image count: 1234/25000", output)
        self.assertIn("Total cancer raw images in repo: 2134", output)
        self.assertIn("python run_medical.py train-all", output)
        recommended_mock.assert_called_once()

    @patch("run_medical.prepare_medical_training_dataset")
    def test_split_dataset_command_prints_counts(self, prepare_mock) -> None:
        prepare_mock.return_value = type("Summary", (), {"train_count": 7, "val_count": 2, "test_count": 1})()
        with patch("sys.argv", ["run_medical.py", "split-dataset"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Train: 7", stdout.getvalue())

    @patch(
        "run_medical.verify_downloads",
        return_value={
            "target_total": 25000,
            "downloaded_total": 27689,
            "remaining_to_target": 0,
            "collections": [
                {
                    "collection_name": "TCGA-LUAD",
                    "downloaded_in_collection": 254,
                    "failed_count": 0,
                    "collection_root": "dataset/medical/tcia/TCGA-LUAD",
                }
            ],
        },
    )
    def test_verify_tcia_command_prints_collection_dir_and_failed_count(self, _verify_mock) -> None:
        with patch("sys.argv", ["run_medical.py", "verify-tcia"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Đã tải: 27689", output)
        self.assertIn("TCGA-LUAD: 254 anh | failed=0 | dir=dataset/medical/tcia/TCGA-LUAD", output)

    @patch("run_medical.build_cancer_overview")
    def test_cancer_command_prints_local_image_breakdown(self, overview_mock) -> None:
        overview_mock.return_value = {
            "summary": {
                "total_cancer_images": 27726,
                "skin_raw_images": 900,
                "tcia_total_images": 26826,
                "skin_image_dir": "dataset/medical/skin_lesion/raw/images",
                "tcia_root": "dataset/medical/tcia",
            },
            "cancers": [
                {
                    "label": "Ung thu da",
                    "local_image_count": 900,
                    "local_status": "co_anh_local",
                    "model_ready": True,
                    "local_sources": [
                        {
                            "collection_name": "skin_lesion",
                            "image_count": 900,
                            "collection_root": "dataset/medical/skin_lesion/raw/images",
                        }
                    ],
                    "sources": [],
                    "model_notes": "Da co model.",
                },
                {
                    "label": "Ung thu phoi",
                    "local_image_count": 4173,
                    "local_status": "co_anh_local",
                    "model_ready": False,
                    "local_sources": [
                        {
                            "collection_name": "NSCLC-Radiomics",
                            "image_count": 4173,
                            "collection_root": "dataset/medical/tcia/NSCLC-Radiomics",
                        }
                    ],
                    "sources": [{"source_name": "TCIA NSCLC-Radiomics", "status": "source_confirmed"}],
                    "model_notes": "Can model CT chuyen dung.",
                },
            ],
        }
        with patch("sys.argv", ["run_medical.py", "cancer"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Tong anh ung thu local: 27726", output)
        self.assertIn("Ung thu da", output)
        self.assertIn("NSCLC-Radiomics: 4173 anh", output)

    @patch("run_medical.train_medical_model", return_value=Path("models/trained/skin_cancer_screening_best.pt"))
    def test_train_command_reports_model_path(self, _train_mock) -> None:
        with patch("sys.argv", ["run_medical.py", "train"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Da luu model medical", stdout.getvalue())

    @patch("run_medical.update_case_report_case_id")
    @patch("run_medical.MedicalCaseDatabase")
    @patch("run_medical.MedicalImageAnalyzer")
    def test_analyze_command_syncs_case_id_into_report(
        self,
        analyzer_cls_mock,
        case_db_cls_mock,
        update_report_mock,
    ) -> None:
        analyzer = analyzer_cls_mock.return_value
        analyzer.analyze_image.return_value = type(
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
                "recommendation": "Can kham chuyen khoa",
                "detections": [],
                "quality_warnings": [],
                "average_confidence": 0.88,
                "model_name": "medical_best.pt",
                "disclaimer": "Khong thay the bac si",
            },
        )()
        case_db = case_db_cls_mock.return_value
        case_db.save_case.return_value = 42

        with patch("sys.argv", ["run_medical.py", "analyze", "--image", "sample.jpg", "--patient-code", "BN555"]), patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        update_report_mock.assert_called_once_with(Path("report.json"), Path("report.md"), case_id=42)
        self.assertIn("Ma ca benh: 42", stdout.getvalue())

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
        self.assertIn("Da xoa", stdout.getvalue())

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
        self.assertIn("Da xoa cac file lien quan", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
