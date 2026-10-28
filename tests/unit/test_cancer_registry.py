import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

import medical.cancer_model_registry as registry
from medical.cancer_model_registry import (
    find_cancer_model,
    list_cancer_models,
    target_model_paths,
)


def _write_project_classifier(path: Path, num_classes: int = 2) -> None:
    from medical.cnn_classifier import MedicalCNNClassifier

    model = MedicalCNNClassifier(num_classes=num_classes, backbone="resnet18", pretrained=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_labels": ["a", "b"][:num_classes],
            "num_classes": num_classes,
            "backbone": "resnet18",
            "image_size": 32,
        },
        path,
    )


class CancerRegistryTests(unittest.TestCase):
    def setUp(self):
        registry._PROBE_CACHE.clear()

    def tearDown(self):
        registry._PROBE_CACHE.clear()

    def test_lists_all_eleven_cancer_groups(self):
        with TemporaryDirectory() as temp_dir:
            models = list_cancer_models(base=Path(temp_dir))
        self.assertEqual(len(models), 11)
        self.assertEqual({model["key"] for model in models}, {
            "liver", "lung", "breast", "stomach", "colorectal", "prostate",
            "cervical", "kidney", "pancreas", "thyroid", "brain",
        })

    def test_brain_artifact_detected_when_present(self):
        with TemporaryDirectory() as temp_dir:
            brain_dir = Path(temp_dir) / "brain"
            brain_dir.mkdir()
            _write_project_classifier(brain_dir / "brain_classifier.pt")
            m = find_cancer_model("brain", base=Path(temp_dir))
        self.assertTrue(m["artifact_ready"])
        self.assertTrue(m["runtime_ready"])
        self.assertTrue(m["classification_ready"])
        self.assertEqual(m["kind"], "torch-state")

    def test_classifier_missing_num_classes_is_not_runtime_ready(self):
        # Checkpoint chỉ có 2 key như tài liệu cũ: status từng báo sẵn sàng
        # nhưng analyze crash KeyError 'num_classes'. Probe phải chạy load thật.
        with TemporaryDirectory() as temp_dir:
            brain_dir = Path(temp_dir) / "brain"
            brain_dir.mkdir()
            torch.save({"model_state_dict": {}, "class_labels": ["a", "b"]}, brain_dir / "classifier.pt")
            m = find_cancer_model("brain", base=Path(temp_dir))
        self.assertFalse(m["classification_ready"])
        self.assertFalse(m["runtime_ready"])
        self.assertIn("num_classes", m["note"])

    def test_garbage_classifier_is_not_runtime_ready(self):
        with TemporaryDirectory() as temp_dir:
            brain_dir = Path(temp_dir) / "brain"
            brain_dir.mkdir()
            (brain_dir / "classifier.pt").write_bytes(b"weights")
            m = find_cancer_model("brain", base=Path(temp_dir))
        self.assertTrue(m["artifact_ready"])
        self.assertFalse(m["classification_ready"])
        self.assertFalse(m["runtime_ready"])
        self.assertIn("không chạy được runtime", m["note"])

    def test_detector_task_is_checked_with_ultralytics(self):
        with TemporaryDirectory() as temp_dir:
            brain_dir = Path(temp_dir) / "brain"
            brain_dir.mkdir()
            detector = brain_dir / "detector.pt"
            detector.write_bytes(b"weights")

            with patch("ultralytics.YOLO") as yolo:
                yolo.return_value.task = "classify"
                m = find_cancer_model("brain", base=Path(temp_dir))
            self.assertFalse(m["localization_ready"])
            self.assertIn("task=", m["note"])

            registry._PROBE_CACHE.clear()
            with patch("ultralytics.YOLO") as yolo:
                yolo.return_value.task = "detect"
                m = find_cancer_model("brain", base=Path(temp_dir))
            self.assertTrue(m["localization_ready"])
            self.assertTrue(m["runtime_ready"])

    def test_garbage_detector_is_not_runtime_ready(self):
        with TemporaryDirectory() as temp_dir:
            brain_dir = Path(temp_dir) / "brain"
            brain_dir.mkdir()
            (brain_dir / "detector.pt").write_bytes(b"weights")
            m = find_cancer_model("brain", base=Path(temp_dir))
        self.assertFalse(m["localization_ready"])
        self.assertFalse(m["runtime_ready"])

    def test_detector_that_loads_but_predict_fails_is_not_ready(self):
        with TemporaryDirectory() as temp_dir:
            brain_dir = Path(temp_dir) / "brain"
            brain_dir.mkdir()
            (brain_dir / "detector.pt").write_bytes(b"weights")
            with patch("ultralytics.YOLO") as yolo:
                yolo.return_value.task = "detect"
                yolo.return_value.predict.side_effect = RuntimeError("state_dict mismatch")
                m = find_cancer_model("brain", base=Path(temp_dir))
        self.assertFalse(m["localization_ready"])
        self.assertIn("không predict được", m["note"])

    def test_onnx_detector_is_discovered(self):
        with TemporaryDirectory() as temp_dir:
            breast_dir = Path(temp_dir) / "breast"
            breast_dir.mkdir()
            (breast_dir / "detector.onnx").write_bytes(b"onnx-bytes")
            with patch("ultralytics.YOLO") as yolo:
                yolo.return_value.task = "detect"
                m = find_cancer_model("breast", base=Path(temp_dir))
        self.assertTrue(m["localization_ready"])
        self.assertTrue(m["runtime_ready"])
        self.assertEqual(m["kind"], "onnx")

    def test_probe_cache_is_per_kind(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            path.write_bytes(b"weights")
            with patch("ultralytics.YOLO") as yolo:
                yolo.return_value.task = "detect"
                yolo.return_value.predict.side_effect = None
                ok_detector, _ = registry.probe_runtime_capability(path, "detector")
            # kind khác không được lấy nhầm kết quả cache của detector
            ok_volume, volume_reason = registry.probe_runtime_capability(path, "volume")
            self.assertTrue(ok_detector)
            self.assertFalse(ok_volume)
            self.assertIn("model 3D", volume_reason)

    def test_garbage_volume_is_not_runtime_ready(self):
        with TemporaryDirectory() as temp_dir:
            brain_dir = Path(temp_dir) / "brain"
            brain_dir.mkdir()
            (brain_dir / "volume.pt2").write_bytes(b"weights")
            (brain_dir / "volume.json").write_text("{}", encoding="utf-8")
            m = find_cancer_model("brain", base=Path(temp_dir))
        self.assertTrue(m["artifact_ready"])
        self.assertFalse(m["volume_ready"])
        self.assertFalse(m["runtime_ready"])
        self.assertIn("volume.pt2", m["note"])

    def test_torch_export_volume_is_runtime_ready(self):
        with TemporaryDirectory() as temp_dir:
            brain_dir = Path(temp_dir) / "brain"
            brain_dir.mkdir()
            exported = torch.export.export(torch.nn.Identity(), (torch.zeros(1),))
            torch.export.save(exported, str(brain_dir / "volume.pt2"))
            (brain_dir / "volume.json").write_text("{}", encoding="utf-8")
            m = find_cancer_model("brain", base=Path(temp_dir))
        self.assertTrue(m["volume_ready"])
        self.assertTrue(m["runtime_ready"])

    def test_source_sidecar_names_repo_in_note(self):
        with TemporaryDirectory() as temp_dir:
            brain_dir = Path(temp_dir) / "brain"
            brain_dir.mkdir()
            (brain_dir / "detector.pt").write_bytes(b"weights")
            (brain_dir / "detector.source.json").write_text(
                '{"repo": "owner/repo", "sha256": "abc"}', encoding="utf-8"
            )
            m = find_cancer_model("brain", base=Path(temp_dir))
        self.assertIn("Nguồn detector.pt: owner/repo", m["note"])

    def test_unknown_key_missing(self):
        m = find_cancer_model("unknown-xyz")
        self.assertFalse(m["artifact_ready"])

    def test_modality_subdirectory_is_preferred(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = root / "lung" / "ct"
            directory.mkdir(parents=True)
            (directory / "classifier.pt").write_bytes(b"weights")
            paths = target_model_paths("lung", root, "CT ngực")
            self.assertEqual(paths["classifier"], directory / "classifier.pt")

    def test_known_modality_does_not_fall_back_to_organ_model(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = root / "lung"
            directory.mkdir(parents=True)
            (directory / "classifier.pt").write_bytes(b"weights")
            paths = target_model_paths("lung", root, "X-quang ngực")
            self.assertEqual(paths["directory"], directory / "xray")
            self.assertFalse(paths["classifier"].is_file())


if __name__ == "__main__":
    unittest.main()
