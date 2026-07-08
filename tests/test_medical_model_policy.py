from __future__ import annotations

import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from medical.model_policy import resolve_medical_runtime_model_path


@dataclass(frozen=True)
class _ModelConfig:
    model_path: Path
    fallback_model_path: Path | None = None
    allow_fallback_model: bool = False


class MedicalModelPolicyTests(unittest.TestCase):
    def test_resolve_prefers_configured_model_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            configured_model = root / "medical_7_cancers.pt"
            configured_model.write_bytes(b"weights")

            resolved = resolve_medical_runtime_model_path(_ModelConfig(model_path=configured_model))

        self.assertEqual(resolved, configured_model)

    def test_resolve_falls_back_to_medical_folder_copy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundled_dir = root / "medical"
            bundled_dir.mkdir(parents=True, exist_ok=True)
            bundled_model = bundled_dir / "medical_7_cancers.pt"
            bundled_model.write_bytes(b"weights")
            original_cwd = Path.cwd()
            os.chdir(root)
            try:
                resolved = resolve_medical_runtime_model_path(_ModelConfig(model_path=Path("medical_7_cancers.pt")))
            finally:
                os.chdir(original_cwd)

        self.assertEqual(resolved, Path("medical/medical_7_cancers.pt"))


if __name__ == "__main__":
    unittest.main()
