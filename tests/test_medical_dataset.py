from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import nibabel as nib
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.filewriter import dcmwrite

from medical.dataset import (
    MedicalDatasetConfig,
    create_default_medical_dataset_config,
    ensure_medical_dataset_structure,
    infer_medical_upload_context,
    is_medical_volume_source,
    is_supported_medical_upload_path,
    load_medical_volume_slices,
    resolve_medical_upload_path,
    normalize_uploaded_image,
)


class MedicalDatasetTests(unittest.TestCase):
    def test_ensure_medical_dataset_structure_creates_skin_cancer_layout(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = create_default_medical_dataset_config(Path(temp_dir) / "medical_ds")
            summary = ensure_medical_dataset_structure(config)

            self.assertTrue(summary.data_yaml_path.exists())
            self.assertTrue((config.dataset_root / "Ung thư gan" / "processed" / "images" / "train").exists())
            self.assertTrue((config.dataset_root / "Ung thư cổ tử cung" / "processed" / "images" / "test").exists())

    def test_ensure_medical_dataset_structure_creates_medical_cancer_layout(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "medical_cancer_ds"
            config = MedicalDatasetConfig(
                disease_name="medical_cancer_screening",
                dataset_root=root,
                metadata_dir=root / "metadata",
                reports_dir=root / "reports",
                data_yaml_path=root / "data.yaml",
                image_size=640,
                class_names=("Ung thư gan",),
            )
            summary = ensure_medical_dataset_structure(config)

            self.assertTrue(summary.data_yaml_path.exists())
            self.assertTrue((config.dataset_root / "Ung thư gan" / "processed" / "images" / "val").exists())
            self.assertTrue(config.metadata_dir.exists())

    def test_normalize_uploaded_image_letterboxes_to_square_rgb(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.png"
            Image.new("L", (300, 100), color=180).save(source)

            normalized = normalize_uploaded_image(source, Path(temp_dir) / "out", image_size=256)

            self.assertTrue(normalized.exists())
            with Image.open(normalized) as image:
                self.assertEqual(image.size, (256, 256))
                self.assertEqual(image.mode, "RGB")

    def test_normalize_uploaded_image_uses_unique_filenames(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.png"
            Image.new("RGB", (64, 64), color=(10, 20, 30)).save(source)

            first = normalize_uploaded_image(source, Path(temp_dir) / "out", image_size=128)
            second = normalize_uploaded_image(source, Path(temp_dir) / "out", image_size=128)

            self.assertNotEqual(first.name, second.name)

    def test_normalize_uploaded_image_handles_alpha_and_extension_support(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.tiff"
            Image.new("RGBA", (100, 50), color=(255, 0, 0, 128)).save(source)

            normalized = normalize_uploaded_image(source, Path(temp_dir) / "out", image_size=128)

            self.assertTrue(is_supported_medical_upload_path(source))
            with Image.open(normalized) as image:
                self.assertEqual(image.size, (128, 128))
                self.assertEqual(image.mode, "RGB")

    def test_normalize_uploaded_image_accepts_dicom(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "input.dcm"
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

            dataset = FileDataset(str(source), {}, file_meta=file_meta, preamble=b"\0" * 128)
            dataset.Rows = 32
            dataset.Columns = 16
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 0
            dataset.PixelData = (np.arange(32 * 16, dtype=np.uint16).reshape(32, 16)).tobytes()
            dcmwrite(source, dataset, little_endian=True, implicit_vr=False)

            normalized = normalize_uploaded_image(source, Path(temp_dir) / "out", image_size=96)

            self.assertTrue(is_supported_medical_upload_path(source))
            with Image.open(normalized) as image:
                self.assertEqual(image.size, (96, 96))
                self.assertEqual(image.mode, "RGB")

    def test_resolve_medical_upload_path_picks_middle_dicom_slice_from_folder(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "series"
            root.mkdir()

            def make_dicom(path: Path, value: int) -> None:
                file_meta = FileMetaDataset()
                file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
                file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
                dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
                dataset.Rows = 16
                dataset.Columns = 16
                dataset.SamplesPerPixel = 1
                dataset.PhotometricInterpretation = "MONOCHROME2"
                dataset.BitsAllocated = 16
                dataset.BitsStored = 16
                dataset.HighBit = 15
                dataset.PixelRepresentation = 0
                dataset.PixelData = (np.full((16, 16), value, dtype=np.uint16)).tobytes()
                dcmwrite(path, dataset, little_endian=True, implicit_vr=False)

            first = root / "slice_001.dcm"
            middle = root / "slice_002.dcm"
            last = root / "slice_003.dcm"
            make_dicom(first, 10)
            make_dicom(middle, 20)
            make_dicom(last, 30)

            resolved = resolve_medical_upload_path(root)

            self.assertEqual(resolved.name, "slice_002.dcm")
            self.assertTrue(is_supported_medical_upload_path(root))

    def test_normalize_uploaded_image_accepts_dicom_series_folder(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "series"
            root.mkdir()

            def make_dicom(path: Path, value: int) -> None:
                file_meta = FileMetaDataset()
                file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
                file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
                dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
                dataset.Rows = 16
                dataset.Columns = 16
                dataset.SamplesPerPixel = 1
                dataset.PhotometricInterpretation = "MONOCHROME2"
                dataset.BitsAllocated = 16
                dataset.BitsStored = 16
                dataset.HighBit = 15
                dataset.PixelRepresentation = 0
                dataset.PixelData = np.full((16, 16), value, dtype=np.uint16).tobytes()
                dcmwrite(path, dataset, little_endian=True, implicit_vr=False)

            make_dicom(root / "slice_001.dcm", 10)
            make_dicom(root / "slice_002.dcm", 20)
            make_dicom(root / "slice_003.dcm", 30)

            normalized = normalize_uploaded_image(root, Path(temp_dir) / "out", image_size=128)

            self.assertTrue(is_supported_medical_upload_path(root))
            with Image.open(normalized) as image:
                self.assertEqual(image.size, (128, 128))
                self.assertEqual(image.mode, "RGB")
            self.assertEqual(len(load_medical_volume_slices(root)), 3)
            self.assertTrue(is_medical_volume_source(root))

    def test_infer_medical_upload_context_uses_dicom_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "chest_ct_series.dcm"
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

            dataset = FileDataset(str(source), {}, file_meta=file_meta, preamble=b"\0" * 128)
            dataset.Modality = "CT"
            dataset.SeriesDescription = "Chest CT lung screening"
            dataset.BodyPartExamined = "CHEST"
            dataset.Rows = 16
            dataset.Columns = 16
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 0
            dataset.PixelData = np.full((16, 16), 20, dtype=np.uint16).tobytes()
            dcmwrite(source, dataset, little_endian=True, implicit_vr=False)

            target_key, modality = infer_medical_upload_context(source)

            self.assertEqual(target_key, "lung")
            self.assertEqual(modality, "CT ngực")

    def test_normalize_uploaded_image_accepts_nifti_volume(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "volume.nii.gz"
            volume = np.zeros((24, 18, 5), dtype=np.float32)
            volume[:, :, 0] = 5
            volume[:, :, 2] = 50
            volume[:, :, 4] = 100
            nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), source)

            normalized = normalize_uploaded_image(source, Path(temp_dir) / "out", image_size=128)

            self.assertTrue(is_supported_medical_upload_path(source))
            with Image.open(normalized) as image:
                self.assertEqual(image.size, (128, 128))
                self.assertEqual(image.mode, "RGB")
            self.assertEqual(len(load_medical_volume_slices(source)), 5)
            self.assertTrue(is_medical_volume_source(source))
