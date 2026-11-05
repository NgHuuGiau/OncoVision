import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from medical.compliance import deidentify_dicom, deidentify_dicom_file


def _sample_dataset() -> Dataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = file_meta
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientName = "NGUYEN VAN A"
    ds.PatientID = "BN-123456"
    ds.PatientBirthDate = "19700101"
    ds.InstitutionName = "Benh vien K"
    ds.Modality = "US"
    ds.Rows = 8
    ds.Columns = 8
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = np.arange(64, dtype=np.uint8).tobytes()
    return ds


class DicomDeidentificationTests(unittest.TestCase):
    def test_pii_blanks_and_uids_stay_valid(self) -> None:
        ds = _sample_dataset()
        old_instance = str(ds.SOPInstanceUID)

        deidentify_dicom(ds)

        self.assertEqual(str(ds.PatientName), "ANONYMOUS")
        self.assertEqual(str(ds.PatientID), "ANONYMOUS")
        self.assertEqual(str(ds.PatientBirthDate), "00010101")
        # SOPClassUID là loại ảnh, không phải PII, và DICOM yêu cầu UID hợp lệ
        self.assertEqual(str(ds.SOPClassUID), SecondaryCaptureImageStorage)
        for name in ("SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID"):
            self.assertTrue(str(ds[name].value).strip(), name)
            self.assertTrue(pydicom.uid.UID(str(ds[name].value)).is_valid, name)
        self.assertNotEqual(str(ds.SOPInstanceUID), old_instance)
        self.assertEqual(str(ds.file_meta.MediaStorageSOPInstanceUID), str(ds.SOPInstanceUID))

    def test_file_roundtrip_keeps_pixels_and_hides_patient(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.dcm"
            target = Path(temp_dir) / "target.dcm"
            ds = _sample_dataset()
            ds.save_as(str(source), enforce_file_format=True)

            deidentify_dicom_file(source, target)

            reread = pydicom.dcmread(str(target), force=True)
            self.assertEqual(str(reread.PatientName), "ANONYMOUS")
            self.assertEqual(str(reread.Modality), "US")
            self.assertTrue(np.array_equal(
                np.frombuffer(reread.PixelData, dtype=np.uint8), np.arange(64, dtype=np.uint8)
            ))


if __name__ == "__main__":
    unittest.main()
