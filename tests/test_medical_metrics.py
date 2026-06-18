from __future__ import annotations

import unittest

from medical.metrics import compute_medical_metrics


class MedicalMetricsTests(unittest.TestCase):
    def test_compute_medical_metrics_returns_expected_values(self) -> None:
        metrics = compute_medical_metrics(
            truths=[True, True, False, False],
            predictions=[True, False, True, False],
        )

        self.assertEqual(metrics.true_positive, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.true_negative, 1)
        self.assertAlmostEqual(metrics.sensitivity, 0.5)
        self.assertAlmostEqual(metrics.specificity, 0.5)

    def test_compute_medical_metrics_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            compute_medical_metrics([True], [True, False])
