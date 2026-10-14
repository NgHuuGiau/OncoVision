import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.fetch_model import main, resolve_destination


class FetchModelTests(unittest.TestCase):
    def test_destination_without_modality_is_organ_level(self):
        with TemporaryDirectory() as temp_dir:
            dest = resolve_destination(Path(temp_dir), "lung", "detector", None)
        self.assertEqual(dest, Path(temp_dir) / "lung" / "detector.pt")

    def test_destination_with_modality_uses_modality_directory(self):
        with TemporaryDirectory() as temp_dir:
            dest = resolve_destination(Path(temp_dir), "brain", "classifier", "MRI não")
        self.assertEqual(dest, Path(temp_dir) / "brain" / "mri" / "classifier.pt")

    def test_invalid_target_rejected(self):
        with self.assertRaises(ValueError):
            resolve_destination(Path("models/pretrained/cancers"), "unknown-xyz", "detector", None)

    def test_main_copies_valid_download_to_destination(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "cached_model.pt"
            source.write_bytes(b"weights")
            base = root / "cancers"
            with patch("scripts.fetch_model.download", return_value=(source, "model")), patch(
                "scripts.fetch_model.probe_runtime_capability", return_value=(True, "")
            ):
                code = main([
                    "owner/repo", "--target", "lung", "--kind", "detector", "--base", str(base),
                ])
            self.assertEqual(code, 0)
            self.assertTrue((base / "lung" / "detector.pt").is_file())
            sidecar = base / "lung" / "detector.source.json"
            self.assertTrue(sidecar.is_file())
            self.assertIn('"repo": "owner/repo"', sidecar.read_text(encoding="utf-8"))

    def test_main_aborts_when_probe_fails(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "cached_model.pt"
            source.write_bytes(b"weights")
            base = root / "cancers"
            with patch("scripts.fetch_model.download", return_value=(source, "model")), patch(
                "scripts.fetch_model.probe_runtime_capability",
                return_value=(False, "task='classify', cần detect hoặc segment"),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    main([
                        "owner/repo", "--target", "lung", "--kind", "detector", "--base", str(base),
                    ])
            self.assertIn("Không dùng được", str(ctx.exception))
            self.assertFalse((base / "lung" / "detector.pt").exists())


if __name__ == "__main__":
    unittest.main()
