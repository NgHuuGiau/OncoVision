from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from medical.clinical_validation import (
    build_independent_validation_report,
    clinical_validation_status,
    plan_binary_validation_sample_size,
)


class ClinicalValidationTests(unittest.TestCase):
    def test_sample_size_plan_accounts_for_positive_and_negative_cases(self) -> None:
        plan = plan_binary_validation_sample_size(
            sensitivity=0.90,
            specificity=0.85,
            prevalence=0.20,
            margin=0.05,
            confidence=0.95,
        )
        self.assertGreaterEqual(plan["planned_positive_cases"], plan["minimum_positive_cases"])
        self.assertGreaterEqual(plan["planned_negative_cases"], plan["minimum_negative_cases"])

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["patient_id", "truth", "prediction"])
            writer.writeheader()
            writer.writerows(rows)

    def test_report_is_bound_to_model_and_requires_external_approval(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model.pt"
            model.write_bytes(b"model-a")
            cohort = root / "cohort.csv"
            self._write_csv(
                cohort,
                [
                    {"patient_id": "P1", "truth": "tumor", "prediction": "tumor"},
                    {"patient_id": "P2", "truth": "no_tumor", "prediction": "tumor"},
                ],
            )
            report_path = build_independent_validation_report(
                cohort, model, root / "report.json", cohort_id="external-hospital-01"
            )
            generated = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("accuracy_95_ci", generated["confidence_intervals"])
            self.assertEqual(len(generated["cohort_csv_sha256"]), 64)

            ready, message = clinical_validation_status(report_path, model)
            self.assertFalse(ready)
            self.assertIn("chưa được", message)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["approved_by"] = "Independent Review Board"
            report["approved_at"] = "2026-10-11T00:00:00+00:00"
            report["dataset_provenance"] = "External hospital registry"
            report["ground_truth_method"] = "Consensus pathology/radiology review"
            report["sample_size_justification"] = "Prospective power calculation"
            report["site_count"] = 2
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(clinical_validation_status(report_path, model)[0])

            model.write_bytes(b"model-b")
            self.assertFalse(clinical_validation_status(report_path, model)[0])

    def test_duplicate_patient_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model.pt"
            model.write_bytes(b"model")
            cohort = root / "cohort.csv"
            self._write_csv(
                cohort,
                [
                    {"patient_id": "P1", "truth": "tumor", "prediction": "tumor"},
                    {"patient_id": "P1", "truth": "tumor", "prediction": "tumor"},
                ],
            )
            with self.assertRaisesRegex(ValueError, "chỉ được xuất hiện một lần"):
                build_independent_validation_report(cohort, model, root / "report.json", cohort_id="external")


if __name__ == "__main__":
    unittest.main()
