from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import run_medical


class RunMedicalTests(unittest.TestCase):
    def test_init_dataset_command_returns_success(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            with patch("sys.argv", ["run_medical.py", "init-dataset", "--dataset-root", temp_dir]), patch(
                "sys.stdout", new_callable=io.StringIO
            ) as stdout:
                code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Da tao cau truc medical", stdout.getvalue())

    def test_metrics_command_prints_json_metrics(self) -> None:
        with patch(
            "sys.argv",
            ["run_medical.py", "metrics", "--truths", json.dumps([True, False]), "--predictions", json.dumps([True, True])],
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn('"true_positive": 1', stdout.getvalue())

    @patch("run_medical._medical_output_directories")
    @patch("run_medical.cleanup_directories")
    def test_cleanup_output_command_reports_summary(self, cleanup_mock, medical_dirs_mock) -> None:
        medical_dirs_mock.return_value = [Path("output/medical/reports")]
        cleanup_mock.return_value = type("CleanupSummary", (), {"removed_files": 5, "removed_dirs": 2, "freed_bytes": 4096})()
        with patch("sys.argv", ["run_medical.py", "cleanup-output", "--older-than-days", "7"]), patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        medical_dirs_mock.assert_called_once_with()
        cleanup_mock.assert_called_once_with([Path("output/medical/reports")], older_than_days=7)
        self.assertIn("Da xoa file: 5", stdout.getvalue())

    @patch("run_medical.recommended_medical_commands", return_value=["python run_medical.py train-all"])
    @patch("run_medical.get_medical_system_status")
    def test_status_command_prints_medical_summary(self, status_mock, recommended_mock) -> None:
        status_mock.return_value = type(
            "MedicalStatus",
            (),
            {
                "configured_model_path": Path("medical_7_cancers.pt"),
                "resolved_model_path": None,
                "allow_fallback_model": False,
                "model_ready": False,
                "model_message": "missing model",
                "dataset_root": Path("dataset/medical"),
                "data_yaml_path": Path("dataset/medical/data.yaml"),
                "dataset_initialized": True,
                "train_images": 1,
                "val_images": 1,
                "test_images": 1,
                "total_images": 3,
                "case_count": 0,
                "report_files": 0,
                "normalized_files": 0,
                "overlay_files": 0,
                "export_files": 0,
                "analyzed_cancers": ("Ung thư gan", "Ung thư vú"),
            },
        )()
        with patch("sys.argv", ["run_medical.py", "status"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Trang thai he thong medical", output)
        self.assertIn("missing model", output)
        self.assertIn("He thong dang phan tich cac ung thu:", output)
        self.assertIn("python run_medical.py train-all", output)
        recommended_mock.assert_called_once()

    @patch("run_medical.prepare_medical_training_dataset")
    def test_split_dataset_command_prints_counts(self, prepare_mock) -> None:
        prepare_mock.return_value = type("Summary", (), {"train_count": 7, "val_count": 2, "test_count": 1})()
        with patch("sys.argv", ["run_medical.py", "split-dataset"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Train: 7", stdout.getvalue())

    @patch("run_medical.build_cancer_overview")
    def test_cancer_command_prints_local_image_breakdown(self, overview_mock) -> None:
        overview_mock.return_value = {
            "summary": {
                "total_cancer_images": 27726,
                "dataset_root": "dataset/medical",
            },
            "cancers": [
                {
                    "label": "Ung thư gan",
                    "local_image_count": 900,
                    "local_status": "có_ảnh_local",
                    "model_ready": True,
                    "local_sources": [
                        {
                            "collection_name": "train",
                            "image_count": 900,
                            "collection_root": "dataset/medical/Ung thư gan/processed/images/train",
                        }
                    ],
                }
            ],
        }
        with patch("sys.argv", ["run_medical.py", "cancer"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Tong anh ung thu local: 27726", output)
        self.assertIn("Ung thư gan", output)
        self.assertIn("train: 900 anh", output)

    def test_ready_command_prints_training_readiness(self) -> None:
        with patch(
            "run_medical.get_medical_system_status",
            return_value=type(
                "MedicalStatus",
                (),
                {
                    "dataset_root": Path("dataset/medical"),
                    "train_images": 7,
                    "val_images": 7,
                    "test_images": 7,
                    "dataset_initialized": True,
                    "model_ready": True,
                    "report_files": 3,
                    "normalized_files": 3,
                    "overlay_files": 3,
                    "export_files": 1,
                    "case_count": 5,
                    "raw_dataset_ready": True,
                    "processed_dataset_ready": True,
                },
            )(),
        ), patch("run_medical.medical_training_paths", return_value=type("Paths", (), {"dataset_root": Path("dataset/medical")})()), patch(
            "run_medical._dataset_split_counts",
            return_value={"train": 7, "val": 7, "test": 7},
        ), patch("sys.argv", ["run_medical.py", "ready"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("ready_for_train_medical: True", output)
        self.assertNotIn("tcia", output.lower())

    @patch("run_medical.train_medical_model", return_value=Path("medical_7_cancers.pt"))
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
            "val_count": 7,
            "test_count": 7,
            "trained_model_path": Path("medical_7_cancers.pt"),
            "validation_metrics": {"accuracy": 0.81},
        }
        with patch("sys.argv", ["run_medical.py", "train-all"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Model: medical_7_cancers.pt", stdout.getvalue().replace("/", "\\"))

    def test_show_case_command_prints_case_detail(self) -> None:
        fake_record = type(
            "FakeMedicalCaseRecord",
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
                "metadata": {"quality_warnings": []},
            },
        )()
        fake_db = type("FakeMedicalCaseDatabase", (), {"get_case": lambda self, case_id: fake_record})()
        with patch("run_medical.MedicalCaseDatabase", return_value=fake_db), patch(
            "sys.argv", ["run_medical.py", "show-case", "--case-id", "100"]
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("BN100", stdout.getvalue())

    def test_delete_case_command_removes_record(self) -> None:
        fake_db = type("FakeMedicalCaseDatabase", (), {"delete_case": lambda self, case_id: True})()
        with patch("run_medical.MedicalCaseDatabase", return_value=fake_db), patch(
            "sys.argv", ["run_medical.py", "delete-case", "--case-id", "200"]
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Da xoa", stdout.getvalue())

    def test_delete_case_command_can_remove_files(self) -> None:
        fake_db = type(
            "FakeMedicalCaseDatabase",
            (),
            {
                "delete_case_with_files": lambda self, case_id: (
                    True,
                    ["source.jpg", "overlay.jpg", "report.json", "report.md"],
                ),
            },
        )()
        with patch("run_medical.MedicalCaseDatabase", return_value=fake_db), patch(
            "sys.argv", ["run_medical.py", "delete-case", "--case-id", "300", "--delete-files"]
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = run_medical.main()

        self.assertEqual(code, 0)
        self.assertIn("Da xoa cac file lien quan", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
