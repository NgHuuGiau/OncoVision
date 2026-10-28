from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from medical.case_manifest import load_case_manifest


class CaseManifestTests(unittest.TestCase):
    def test_loads_relative_image_and_validates_target(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "scan.png"
            image.write_bytes(b"image")
            manifest = root / "case.yaml"
            manifest.write_text(
                "patient_id: P001\ncancer_target: thyroid\nmodality: ultrasound\nimage: scan.png\n",
                encoding="utf-8",
            )
            loaded = load_case_manifest(manifest)
            self.assertEqual(loaded.cancer_target, "thyroid")
            self.assertEqual(loaded.image, image)


if __name__ == "__main__":
    unittest.main()
