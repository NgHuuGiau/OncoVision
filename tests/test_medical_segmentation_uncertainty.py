from __future__ import annotations

import unittest

import numpy as np


class SegmentationModelsTests(unittest.TestCase):
    def test_sam_roi_extractor_fallback_otsu(self) -> None:
        from medical.segmentation import SAMROIExtractor, SegmentationResult

        extractor = SAMROIExtractor()
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        image[20:70, 20:70] = 255
        result = extractor.extract_roi(image)
        self.assertIsInstance(result, SegmentationResult)
        self.assertEqual(len(result.bbox), 4)
        self.assertGreaterEqual(result.confidence, 0.0)

    def test_crop_to_roi(self) -> None:
        from medical.segmentation import crop_to_roi

        image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        cropped = crop_to_roi(image, (10, 10, 40, 40), margin=5)
        self.assertGreater(cropped.size, 0)
        self.assertLessEqual(cropped.shape[0], image.shape[0])

    def test_crop_to_roi_invalid_bbox_returns_original(self) -> None:
        from medical.segmentation import crop_to_roi

        image = np.zeros((50, 50, 3), dtype=np.uint8)
        cropped = crop_to_roi(image, (40, 40, 10, 10), margin=0)
        self.assertEqual(cropped.shape, image.shape)


class UncertaintyModuleTests(unittest.TestCase):
    def test_compute_ece(self) -> None:
        from medical.uncertainty import compute_ece

        probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
        labels = np.array([0, 1, 0])
        ece = compute_ece(probs, labels, n_bins=5)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_mc_dropout_on_tiny_model(self) -> None:
        import torch.nn as nn

        from medical.uncertainty import MCDropoutUncertainty

        model = nn.Sequential(nn.Flatten(), nn.Linear(12, 8), nn.Dropout(0.5), nn.Linear(8, 3))
        estimator = MCDropoutUncertainty(model, num_samples=5, device="cpu")
        import torch

        x = torch.randn(1, 3, 2, 2)
        result = estimator.predict(x, class_labels=["a", "b", "c"])
        self.assertIn(result.predicted_label, {"a", "b", "c"})
        self.assertEqual(result.mean_probability.shape[0], 3)


class ExperimentalImportTests(unittest.TestCase):
    def test_experimental_multitask_importable(self) -> None:
        from experimental.multitask import MultiTaskLoss, MultiTaskMedicalModel, MultiTaskOutput

        self.assertTrue(callable(MultiTaskMedicalModel))
        self.assertTrue(callable(MultiTaskLoss))
        self.assertTrue(MultiTaskOutput is not None)

    def test_experimental_self_supervised_importable(self) -> None:
        from experimental.self_supervised import DINOv2Pretrainer, SelfSupervisedConfig

        self.assertTrue(callable(DINOv2Pretrainer))
        config = SelfSupervisedConfig()
        self.assertEqual(config.method, "dinov2")

    def test_experimental_ensemble_importable(self) -> None:
        import experimental.ensemble as ensemble_mod

        self.assertTrue(hasattr(ensemble_mod, "MedicalEnsemble"))

    def test_experimental_augmentation_importable(self) -> None:
        from experimental.augmentation import AugmentationConfig, MedicalAugmentationPipeline

        self.assertTrue(callable(MedicalAugmentationPipeline))
        self.assertTrue(AugmentationConfig is not None)


if __name__ == "__main__":
    unittest.main()
