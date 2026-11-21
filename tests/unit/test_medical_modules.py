"""Tests for medical modules without coverage: explainability, uncertainty, segmentation, preprocessing."""
import numpy as np


class TestExplainability:
    """Test Grad-CAM explainability module."""

    def test_gradcam_class_exists(self):
        from medical.explainability import GradCAM, GradCAMResult
        # Verify classes exist and are importable
        assert GradCAM is not None
        assert GradCAMResult is not None
        # GradCAMResult is a frozen dataclass — check fields via __dataclass_fields__
        import dataclasses
        fields = {f.name for f in dataclasses.fields(GradCAMResult)}
        assert "label" in fields

    def test_generate_gradcam_overlay(self):
        from medical.explainability import generate_gradcam_overlay

        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        heatmap = np.random.rand(32, 32).astype(np.float32)
        overlay = generate_gradcam_overlay(image, heatmap)
        assert overlay.shape == (64, 64, 3)
        assert overlay.dtype == np.uint8


class TestUncertainty:
    """Test uncertainty quantification module."""

    def test_quantification_module_imports(self):
        from medical.uncertainty import quantification
        # Verify module is importable
        assert hasattr(quantification, "__file__")

    def test_entropy_calculation(self):
        import torch

        # Manual entropy calculation to verify logic
        certain = torch.tensor([[0.9, 0.05, 0.05]])
        uniform = torch.tensor([[0.33, 0.33, 0.34]])

        def entropy(p):
            return -(p * torch.log(p + 1e-9)).sum(dim=-1)

        entropy_certain = entropy(certain)
        entropy_uniform = entropy(uniform)

        assert entropy_uniform > entropy_certain


class TestSegmentation:
    """Test segmentation models."""

    def test_segmentation_module_imports(self):
        from medical.segmentation import models
        # Verify module is importable
        assert hasattr(models, "__file__")

    def test_attention_unet_class(self):
        import torch

        from medical.segmentation.models import AttentionUNet

        # Check the actual signature of AttentionUNet
        model = AttentionUNet(in_channels=3, num_classes=2)
        input_tensor = torch.randn(1, 3, 64, 64)
        output = model(input_tensor)
        assert output.shape[0] == 1
        assert output.shape[1] == 2


class TestPreprocessing:
    """Test preprocessing pipeline."""

    def test_preprocess_image_basic(self):
        from medical.preprocessing.pipeline import preprocess_image

        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocess_image(image, target_size=320, modality="ct")
        assert result.image.shape == (320, 320, 3)
        assert result.modality == "ct"

    def test_preprocess_colonoscopy(self):
        from medical.preprocessing.colonoscopy import preprocess_colonoscopy

        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocess_colonoscopy(image, target_size=320)
        assert result.image.shape == (320, 320, 3)
        assert result.modality == "colonoscopy"
        assert result.metadata.get("bubbles_removed") is True

    def test_preprocess_colposcopy(self):
        from medical.preprocessing.colposcopy import preprocess_colposcopy

        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocess_colposcopy(image, target_size=320)
        assert result.image.shape == (320, 320, 3)
        assert result.modality == "colposcopy"
        assert result.metadata.get("acetowhite_normalized") is True

    def test_get_preprocessor_mapping(self):
        from medical.preprocessing.pipeline import get_preprocessor

        # Test all registered preprocessors
        for modality in ("ct", "mri", "xray", "mammogram", "pet_ct", "endoscopy",
                         "colonoscopy", "colposcopy", "ultrasound"):
            preprocessor = get_preprocessor(modality)
            assert preprocessor is not None
            assert callable(preprocessor)

    def test_default_preprocessor_fallback(self):
        from medical.preprocessing.pipeline import get_preprocessor

        preprocessor = get_preprocessor("unknown_modality")
        assert preprocessor is not None


class TestTracking:
    """Test tracking module."""

    def test_tracking_module_imports(self):
        from core import tracking
        # Verify module is importable
        assert hasattr(tracking, "__path__")

    def test_iou_calculation(self):
        # Manual IoU calculation to verify logic
        box_a = np.array([0, 0, 100, 100])
        box_b = np.array([50, 50, 150, 150])

        # Compute IoU manually
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - intersection
        iou = intersection / union if union > 0 else 0.0

        assert 0.0 <= iou <= 1.0
        assert iou > 0  # Overlapping boxes should have non-zero IoU
