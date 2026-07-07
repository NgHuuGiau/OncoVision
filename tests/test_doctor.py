from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from medical.system_status import MedicalSystemStatus
import run_doctor


class DoctorTests(unittest.TestCase):
    def test_present_and_missing_models_reports_expected_values(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "yolo11n.pt").write_text("n", encoding="utf-8")
            (model_dir / "yolo11s.pt").write_text("s", encoding="utf-8")
            present, missing = run_doctor._present_and_missing_models(model_dir)
        self.assertEqual(present, ["yolo11n.pt", "yolo11s.pt"])
        self.assertEqual(missing, ["yolo11m.pt", "yolo11l.pt", "yolo11x.pt"])

    def test_count_files_ignores_missing_directory(self) -> None:
        self.assertEqual(run_doctor.count_files(Path("missing-dir-for-doctor-test")), 0)

    @patch("run_doctor._open_camera_capture")
    def test_probe_camera_reports_warn_when_camera_cannot_open(self, open_camera_mock) -> None:
        capture = Mock()
        capture.isOpened.return_value = False
        open_camera_mock.return_value = capture

        result = run_doctor._probe_camera(0)

        self.assertEqual(result.level, "WARN")
        self.assertIn("camera index 0", result.summary)

    @patch("run_doctor._open_camera_capture")
    def test_probe_camera_reports_pass_when_frame_is_read(self, open_camera_mock) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, Mock(shape=(720, 1280, 3)))
        open_camera_mock.return_value = capture

        result = run_doctor._probe_camera(1)

        self.assertEqual(result.level, "PASS")
        self.assertIn("index 1", result.summary)
        self.assertIn("1280x720", result.detail)

    @patch("builtins.print")
    @patch("run_doctor.recommended_medical_commands", return_value=["python run_medical.py train-all"])
    @patch("run_doctor.get_medical_system_status")
    @patch("run_doctor.optimized_runtime")
    @patch("run_doctor.detect_hardware")
    @patch("run_doctor.ensure_project_directories")
    @patch("run_doctor.parse_args")
    def test_main_reports_missing_models_and_raw_dataset_guidance(
        self,
        parse_args_mock,
        _ensure_dirs_mock,
        detect_hardware_mock,
        optimized_runtime_mock,
        medical_status_mock,
        _medical_commands_mock,
        print_mock,
    ) -> None:
        parse_args_mock.return_value = Mock(camera_index=0, skip_camera_check=True, fix=False)
        detect_hardware_mock.return_value = Mock(
            cpu_name="Intel Core i7",
            ram_gb=16.0,
            os_name="Windows 11",
            gpu_name="Kh\u00f4ng ph\u00e1t hi\u1ec7n GPU",
            gpu_count=0,
            vram_gb=0.0,
            torch_version="2.0",
            cuda_runtime_reason="CPU-only",
            cuda_available=False,
            gpu_hardware_available=False,
        )
        optimized_runtime_mock.side_effect = [
            Mock(primary_model_name="yolo11n.pt", resolved_device="cpu", imgsz=320),
            Mock(primary_model_name="yolo11n.pt", resolved_device="cpu", imgsz=320),
            Mock(primary_model_name="yolo11n.pt", resolved_device="cpu", imgsz=320),
        ]
        medical_status_mock.return_value = MedicalSystemStatus(
            configured_model_path=Path("models/trained/skin.pt"),
            resolved_model_path=None,
            allow_fallback_model=False,
            using_fallback_model=False,
            model_ready=False,
            model_message="missing medical model",
            dataset_root=Path("dataset/medical/skin_lesion"),
            data_yaml_path=Path("dataset/medical/skin_lesion/data.yaml"),
            raw_images=0,
            raw_labels=0,
            train_images=0,
            val_images=0,
            test_images=0,
            report_files=0,
            normalized_files=0,
            overlay_files=0,
            export_files=0,
            case_db_path=Path("output/medical/medical_cases.db"),
            case_count=0,
            screening_targets=(("Ung thu da", True), ("Ung thu vu", False)),
            analyzed_cancers=("Ung thu da", "Ung thu vu"),
        )

        with patch("run_doctor._present_and_missing_models", return_value=([], ["yolo11n.pt"])), patch(
            "run_doctor.count_files",
            side_effect=[0] * 20,
        ):
            run_doctor.main()

        output = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("download_models.py", output)
        self.assertIn("prepare_dataset.py", output)
        self.assertIn("model local", output)
        self.assertIn("run_doctor.py --fix", output)
        self.assertIn("run_medical.py train-all", output)

    @patch("builtins.print")
    @patch("run_doctor.medical_config_issues", return_value=["conf_threshold phai nam trong khoang (0, 1)."])
    @patch("run_doctor.runtime_config_issues", return_value=["config/settings.yaml thieu hoac sai muc `models`."])
    @patch("run_doctor.recommended_medical_commands", return_value=["python run_medical.py validate"])
    @patch("run_doctor.get_medical_system_status")
    @patch("run_doctor.detect_hardware")
    @patch("run_doctor.ensure_project_directories")
    @patch("run_doctor.parse_args")
    def test_main_reports_medical_config_issues(
        self,
        parse_args_mock,
        _ensure_dirs_mock,
        detect_hardware_mock,
        medical_status_mock,
        _medical_commands_mock,
        _runtime_config_issues_mock,
        _config_issues_mock,
        print_mock,
    ) -> None:
        parse_args_mock.return_value = Mock(camera_index=0, skip_camera_check=True, fix=False)
        detect_hardware_mock.return_value = Mock(
            cpu_name="Intel Core i7",
            ram_gb=16.0,
            os_name="Windows 11",
            gpu_name="Khong phat hien GPU",
            gpu_count=0,
            vram_gb=0.0,
            torch_version="2.0",
            cuda_runtime_reason="CPU-only",
            cuda_available=False,
            gpu_hardware_available=False,
        )
        medical_status_mock.return_value = MedicalSystemStatus(
            configured_model_path=Path("models/trained/skin.pt"),
            resolved_model_path=Path("models/trained/skin.pt"),
            allow_fallback_model=False,
            using_fallback_model=False,
            model_ready=True,
            model_message="ready",
            dataset_root=Path("dataset/medical/skin_lesion"),
            data_yaml_path=Path("dataset/medical/skin_lesion/data.yaml"),
            raw_images=1,
            raw_labels=1,
            train_images=1,
            val_images=1,
            test_images=1,
            report_files=0,
            normalized_files=0,
            overlay_files=0,
            export_files=0,
            case_db_path=Path("output/medical/medical_cases.db"),
            case_count=0,
            screening_targets=(("Ung thu da", True), ("Ung thu vu", False)),
            analyzed_cancers=("Ung thu da", "Ung thu vu"),
        )

        run_doctor.main()

        output = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("CẤU HÌNH", output)
        self.assertIn("Cần kiểm tra", output)
        self.assertIn("conf_threshold", output)
        self.assertIn("models", output)


if __name__ == "__main__":
    unittest.main()
