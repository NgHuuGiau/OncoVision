import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from medical.cancer_model_registry import KEY_TO_DIR
from scripts.fetch_model import main, resolve_destination

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / "models" / "model_catalog.json"


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

    def test_onnx_source_keeps_onnx_destination(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "cached_model.onnx"
            source.write_bytes(b"onnx-bytes")
            base = root / "cancers"
            with patch("scripts.fetch_model.download", return_value=(source, "model")), patch(
                "scripts.fetch_model.probe_runtime_capability", return_value=(True, "")
            ):
                code = main([
                    "owner/repo", "--target", "breast", "--kind", "detector", "--base", str(base),
                ])
            self.assertEqual(code, 0)
            self.assertTrue((base / "breast" / "detector.onnx").is_file())
            self.assertFalse((base / "breast" / "detector.pt").exists())


    def test_catalog_entries_are_valid_and_unique(self):
        data = json.loads(CATALOG.read_text(encoding="utf-8"))
        seen = set()
        for entry in data["entries"]:
            self.assertIn(entry["target"], KEY_TO_DIR)
            self.assertIn(entry["kind"], ("classifier", "detector"))
            self.assertTrue(entry["repo"] and entry["file"] and entry["sha256"])
            key = (entry["target"], entry.get("modality"), entry["kind"])
            self.assertNotIn(key, seen, f"trùng đích model {key}")
            seen.add(key)

    def _write_catalog(self, path: Path, entry: dict) -> None:
        path.write_text(json.dumps({"version": 1, "entries": [entry]}), encoding="utf-8")

    def test_fetch_all_skips_existing_model_with_matching_sha(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "cancers"
            payload = b"weights"
            dest = base / "lung" / "detector.pt"
            dest.parent.mkdir(parents=True)
            dest.write_bytes(payload)
            catalog = root / "catalog.json"
            self._write_catalog(
                catalog,
                {
                    "target": "lung",
                    "kind": "detector",
                    "modality": None,
                    "repo": "owner/repo",
                    "repo_type": "model",
                    "file": "best.pt",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
            )
            with patch("scripts.fetch_model.download", side_effect=AssertionError("không được tải")):
                code = main(["--all", "--catalog", str(catalog), "--base", str(base)])
            self.assertEqual(code, 0)
            self.assertEqual(dest.read_bytes(), payload)

    def test_fetch_all_reports_failure_without_aborting(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "cancers"
            catalog = root / "catalog.json"
            self._write_catalog(
                catalog,
                {
                    "target": "lung",
                    "kind": "detector",
                    "modality": None,
                    "repo": "owner/repo",
                    "repo_type": "model",
                    "file": "best.pt",
                    "sha256": None,
                },
            )
            with patch("scripts.fetch_model.download", side_effect=SystemExit("mạng lỗi")):
                code = main(["--all", "--catalog", str(catalog), "--base", str(base)])
            self.assertEqual(code, 1)
            self.assertFalse((base / "lung" / "detector.pt").exists())


if __name__ == "__main__":
    unittest.main()
