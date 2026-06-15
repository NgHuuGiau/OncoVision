from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from utils.file_utils import ensure_project_directories, load_yaml, load_yaml_cached, save_yaml


class FileUtilsTests(unittest.TestCase):
    def test_save_yaml_and_load_yaml_roundtrip(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            yaml_path = Path(temp_dir) / "sample.yaml"
            payload = {
                "system": {"auto_detect_hardware": True},
                "camera": {"width": 1000, "height": 650},
            }
            save_yaml(yaml_path, payload)
            loaded = load_yaml(yaml_path)
            self.assertEqual(loaded, payload)

    def test_ensure_project_directories_creates_all_configured_paths(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            project_dirs = (
                Path(temp_dir) / "dataset/raw/images",
                Path(temp_dir) / "models/pretrained",
                Path(temp_dir) / "output/logs",
            )

            with patch("utils.file_utils.PROJECT_DIRS", project_dirs):
                ensure_project_directories()

            for directory in project_dirs:
                self.assertTrue(directory.exists(), msg=str(directory))
                self.assertTrue(directory.is_dir(), msg=str(directory))

    def test_load_yaml_cached_returns_cached_value_until_cache_is_cleared(self) -> None:
        load_yaml_cached.cache_clear()
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            yaml_path = Path(temp_dir) / "cached.yaml"

            save_yaml(yaml_path, {"mode": "initial"})
            first = load_yaml_cached(str(yaml_path))

            yaml_path.write_text("mode: overwritten\n", encoding="utf-8")
            second = load_yaml_cached(str(yaml_path))

            save_yaml(yaml_path, {"mode": "saved"})
            third = load_yaml_cached(str(yaml_path))

        self.assertEqual(first, {"mode": "initial"})
        self.assertEqual(second, {"mode": "initial"})
        self.assertEqual(third, {"mode": "saved"})
