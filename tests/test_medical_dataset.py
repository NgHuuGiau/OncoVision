from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from medical.dataset import create_default_skin_cancer_dataset_config, ensure_medical_dataset_structure, normalize_uploaded_image


class MedicalDatasetTests(unittest.TestCase):
    def test_ensure_medical_dataset_structure_creates_skin_cancer_layout(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            config = create_default_skin_cancer_dataset_config(Path(temp_dir) / "medical_ds")
            summary = ensure_medical_dataset_structure(config)

            self.assertTrue(summary.data_yaml_path.exists())
            self.assertTrue((config.processed_images_dir / "train").exists())
            self.assertTrue((config.processed_labels_dir / "test").exists())

    def test_normalize_uploaded_image_letterboxes_to_square_rgb(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            source = Path(temp_dir) / "input.png"
            Image.new("L", (300, 100), color=180).save(source)

            normalized = normalize_uploaded_image(source, Path(temp_dir) / "out", image_size=256)

            self.assertTrue(normalized.exists())
            with Image.open(normalized) as image:
                self.assertEqual(image.size, (256, 256))
                self.assertEqual(image.mode, "RGB")

    def test_normalize_uploaded_image_uses_unique_filenames(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            source = Path(temp_dir) / "input.png"
            Image.new("RGB", (64, 64), color=(10, 20, 30)).save(source)

            first = normalize_uploaded_image(source, Path(temp_dir) / "out", image_size=128)
            second = normalize_uploaded_image(source, Path(temp_dir) / "out", image_size=128)

            self.assertNotEqual(first.name, second.name)
