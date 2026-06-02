from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training import split_dataset
from utils.file_utils import ensure_project_directories


class SplitDatasetTests(unittest.TestCase):
    def test_audit_raw_dataset_flags_invalid_and_orphan_labels(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                ensure_project_directories()
                Path("dataset/raw/images/a.jpg").write_text("img", encoding="utf-8")
                Path("dataset/raw/images/b.jpg").write_text("img", encoding="utf-8")
                Path("dataset/raw/labels/a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
                Path("dataset/raw/labels/b.txt").write_text("0 10 0.5 0.2 0.2\n", encoding="utf-8")
                Path("dataset/raw/labels/c.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

                audit = split_dataset.audit_raw_dataset()

                self.assertEqual(audit.raw_image_count, 2)
                self.assertEqual([path.name for path in audit.eligible_images], ["a.jpg"])
                self.assertEqual(audit.invalid_labels[0][0].name, "b.txt")
                self.assertEqual([path.name for path in audit.orphan_labels], ["c.txt"])
            finally:
                os.chdir(previous_cwd)

    def test_main_resets_processed_dirs_and_copies_only_eligible_files(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                ensure_project_directories()
                Path("dataset/raw/images/a.jpg").write_text("img-a", encoding="utf-8")
                Path("dataset/raw/images/b.jpg").write_text("img-b", encoding="utf-8")
                Path("dataset/raw/images/c.jpg").write_text("img-c", encoding="utf-8")
                Path("dataset/raw/labels/a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
                Path("dataset/raw/labels/b.txt").write_text("\n", encoding="utf-8")
                Path("dataset/raw/labels/c.txt").write_text("invalid line\n", encoding="utf-8")
                Path("dataset/processed/images/train/stale.jpg").write_text("stale", encoding="utf-8")

                split_dataset.main()

                copied_images = sorted(path.name for path in Path("dataset/processed/images").rglob("*") if path.is_file())
                copied_labels = sorted(path.name for path in Path("dataset/processed/labels").rglob("*") if path.is_file())
                self.assertEqual(copied_images, ["a.jpg", "b.jpg"])
                self.assertEqual(copied_labels, ["a.txt", "b.txt"])
                self.assertFalse(Path("dataset/processed/images/train/stale.jpg").exists())
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
