from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from medical.volume_inference import (
    TorchScriptVolumeRuntime,
    VolumeData,
    VolumeModelManifest,
    load_medical_volume,
    render_volume_overlay,
)


class _TinyVolumeModel(torch.nn.Module):
    def forward(self, value):
        mean = value.mean(dim=(2, 3, 4), keepdim=True)
        logits = torch.cat((mean * 0.0, mean * 0.0 + 5.0), dim=1)
        return logits, (value - 0.5) * 12.0


class VolumeInferenceTests(unittest.TestCase):
    def _write_runtime(self, root: Path) -> tuple[Path, Path]:
        model_path = root / "brain_volume_model.pt"
        manifest_path = root / "brain_volume_model.json"
        traced = torch.jit.trace(_TinyVolumeModel(), torch.zeros((1, 1, 8, 8, 8)))
        traced.save(str(model_path))
        manifest_path.write_text(
            json.dumps(
                {
                    "task": "volume_multitask",
                    "target": "brain",
                    "class_labels": ["no_tumor", "tumor"],
                    "input_shape": [8, 8, 8],
                    "input_modalities": ["image"],
                    "segmentation_labels": ["whole_tumor"],
                    "mask_threshold": 0.5,
                }
            ),
            encoding="utf-8",
        )
        return model_path, manifest_path

    def test_manifest_rejects_unbounded_shape(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.json"
            path.write_text(
                json.dumps({"task": "volume_segmentation", "input_shape": [1024, 1024, 1024]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "giới hạn bộ nhớ"):
                VolumeModelManifest.load(path)

    def test_runtime_returns_voxel_mm_coordinates_and_volume(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path, manifest_path = self._write_runtime(root)
            array = np.zeros((8, 8, 8), dtype=np.float32)
            array[2:6, 1:5, 3:7] = 1.0
            affine = np.diag([2.0, 3.0, 4.0, 1.0])
            volume = VolumeData(array=array, spacing_xyz=(2.0, 3.0, 4.0), affine=affine)

            result = TorchScriptVolumeRuntime(model_path, manifest_path, device="cpu").predict(volume)

            self.assertEqual(result.findings[0].bbox_voxel, (3, 1, 2, 7, 5, 6))
            self.assertEqual(result.findings[0].bbox_mm, (6.0, 3.0, 8.0, 14.0, 15.0, 24.0))
            self.assertEqual(result.findings[0].volume_mm3, 64 * 24.0)
            normalized, overlay = render_volume_overlay(volume, result, root / "normalized", root / "overlay", patient_code="BN 01")
            self.assertTrue(normalized.is_file())
            self.assertTrue(overlay.is_file())

    def test_load_nifti_preserves_xyz_affine(self) -> None:
        import nibabel as nib

        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "brain.nii.gz"
            affine = np.diag([2.0, 3.0, 4.0, 1.0])
            nib.save(nib.Nifti1Image(np.zeros((5, 6, 7), dtype=np.float32), affine), source)

            volume = load_medical_volume(source)

            self.assertEqual(volume.array.shape, (7, 6, 5))
            self.assertEqual(volume.spacing_xyz, (2.0, 3.0, 4.0))
            self.assertEqual(volume.coordinate_system, "RAS+")
            self.assertEqual(volume.voxel_to_world((1, 2, 3)), (2.0, 6.0, 12.0))

    def test_load_multimodal_nifti_requires_registered_channels(self) -> None:
        import nibabel as nib

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            affine = np.diag([1.0, 1.0, 1.0, 1.0])
            for index, modality in enumerate(("t1", "t1ce", "t2", "flair")):
                nib.save(
                    nib.Nifti1Image(np.full((5, 6, 7), index, dtype=np.float32), affine),
                    root / f"case_{modality}.nii.gz",
                )

            volume = load_medical_volume(root, input_modalities=("t1", "t1ce", "t2", "flair"))

            self.assertEqual(volume.array.shape, (4, 7, 6, 5))
            self.assertEqual(volume.modality, "MRI")


if __name__ == "__main__":
    unittest.main()
