from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from entrypoints.web.shared import _has_allowed_magic


class UploadMagicTests(unittest.TestCase):
    def test_nifti_requires_real_header(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fake.nii"
            path.write_bytes(b"not a nifti file")

            self.assertFalse(_has_allowed_magic(path, "fake.nii"))

    def test_nifti_header_is_accepted(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "brain.nii"
            path.write_bytes(b"\x00" * 344 + b"n+1\x00" + b"\x00" * 252)

            self.assertTrue(_has_allowed_magic(path, "brain.nii"))

    def test_gzip_nifti_is_accepted(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "brain.nii.gz"
            path.write_bytes(b"\x1f\x8b\x08\x00" + b"\x00" * 100)

            self.assertTrue(_has_allowed_magic(path, "brain.nii.gz"))


if __name__ == "__main__":
    unittest.main()
