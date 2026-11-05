from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from medical.dataset_manifest import split_dataset_by_patient, validate_dataset_manifest


class DatasetManifestTests(unittest.TestCase):
    def _write(self, root: Path, rows: list[dict[str, str]]) -> Path:
        source = root / "dataset.csv"
        fields = ["patient_id", "image_path", "cancer_target", "modality", "truth_label", "mask_path", "site_id", "split"]
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return source

    def test_detects_patient_leakage_between_splits(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.png").write_bytes(b"x")
            (root / "b.png").write_bytes(b"x")
            source = self._write(root, [
                {"patient_id": "P1", "image_path": "a.png", "cancer_target": "brain", "modality": "mri", "truth_label": "tumor", "mask_path": "", "site_id": "A", "split": "train"},
                {"patient_id": "P1", "image_path": "b.png", "cancer_target": "brain", "modality": "mri", "truth_label": "tumor", "mask_path": "", "site_id": "A", "split": "test"},
            ])
            audit = validate_dataset_manifest(source)
            self.assertFalse(audit.valid)
            self.assertIn("Rò rỉ bệnh nhân", " ".join(audit.errors))

    def test_split_keeps_each_patient_in_one_partition(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for patient in range(20):
                for image_index in range(2):
                    name = f"p{patient}_{image_index}.png"
                    (root / name).write_bytes(b"x")
                    rows.append({"patient_id": f"P{patient}", "image_path": name, "cancer_target": "brain", "modality": "mri", "truth_label": "tumor", "mask_path": "", "site_id": "A", "split": ""})
            source = self._write(root, rows)
            output = split_dataset_by_patient(source, root / "split.csv")
            with output.open("r", encoding="utf-8", newline="") as stream:
                result = list(csv.DictReader(stream))
            patient_splits: dict[str, set[str]] = {}
            for row in result:
                patient_splits.setdefault(row["patient_id"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in patient_splits.values()))
            self.assertEqual({row["split"] for row in result}, {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
