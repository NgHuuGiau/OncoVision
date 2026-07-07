from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from medical.importers import import_isic_2016_part3b_to_yolo


class MedicalImporterTests(TestCase):
    def test_importer_converts_mask_bbox_using_pixel_edges(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "source"
            images_dir = Path(temp_dir) / "images"
            labels_dir = Path(temp_dir) / "labels"
            metadata_path = Path(temp_dir) / "metadata.csv"
            source_root.mkdir(parents=True, exist_ok=True)

            image_path = source_root / "case1.jpg"
            mask_path = source_root / "case1_Segmentation.png"
            Image.new("RGB", (4, 4), color=(0, 0, 0)).save(image_path)
            mask = Image.new("L", (4, 4), color=0)
            mask.putpixel((1, 1), 255)
            mask.putpixel((2, 1), 255)
            mask.putpixel((1, 2), 255)
            mask.putpixel((2, 2), 255)
            mask.save(mask_path)

            result = import_isic_2016_part3b_to_yolo(
                source_root,
                target_images_dir=images_dir,
                target_labels_dir=labels_dir,
                metadata_output_path=metadata_path,
            )

            self.assertEqual(result, {"imported": 1, "skipped": 0})
            self.assertEqual((labels_dir / "case1.txt").read_text(encoding="utf-8"), "0 0.500000 0.500000 0.500000 0.500000\n")
