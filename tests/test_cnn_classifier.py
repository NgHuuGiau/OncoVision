from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from medical.cnn_classifier import MedicalCNNClassifier, is_cnn_classifier_path
from medical.classifier import MedicalClassifierModel


class CNNDummyWrapper:
    def __init__(self, model, class_labels, device=None):
        self.model = model
        self.class_labels = class_labels
        self.device = device

    def predict(self, source, *, top_k=3):
        return [
            {"label": self.class_labels[0], "confidence": 0.9, "probabilities": {self.class_labels[0]: 0.9}}
        ]


class CNNClassifierTests(unittest.TestCase):
    def test_model_forward_pass(self) -> None:
        model = MedicalCNNClassifier(num_classes=7, backbone="resnet18", pretrained=False)
        import torch
        x = torch.randn(2, 3, 320, 320)
        out = model(x)
        self.assertEqual(out.shape, (2, 7))

    def test_different_backbones(self) -> None:
        for backbone in ["resnet50", "efficientnet_b0"]:
            model = MedicalCNNClassifier(num_classes=7, backbone=backbone, pretrained=False)
            import torch
            x = torch.randn(1, 3, 320, 320)
            out = model(x)
            self.assertEqual(out.shape, (1, 7))

    def test_invalid_backbone_raises(self) -> None:
        with self.assertRaises(ValueError):
            MedicalCNNClassifier(num_classes=7, backbone="unknown")

    def test_save_and_load_roundtrip(self) -> None:
        model = MedicalCNNClassifier(num_classes=3, backbone="resnet18", pretrained=False)
        class_labels = ("A", "B", "C")
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            from medical.cnn_classifier import MedicalCNNClassifierWrapper
            wrapper = MedicalCNNClassifierWrapper(model=model, class_labels=class_labels)
            wrapper.save(path)
            self.assertTrue(path.exists())
            loaded = MedicalCNNClassifierWrapper.load(path)
            self.assertEqual(loaded.class_labels, class_labels)

    def test_detect_cnn_checkpoint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cnn_model.pt"
            import torch
            torch.save({"model_state_dict": {}, "class_labels": ("A",), "backbone": "resnet50", "num_classes": 1}, path)
            self.assertTrue(is_cnn_classifier_path(path))

    def test_centroid_model_not_detected_as_cnn(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "centroid.pt"
            dummy = MedicalClassifierModel(class_labels=("A",), centroids=np.zeros((1, 3072), dtype=np.float32))
            import pickle
            with open(path, "wb") as f:
                pickle.dump(dummy, f)
            self.assertFalse(is_cnn_classifier_path(path))
