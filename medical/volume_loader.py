from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _entropy_slice_score(slice_2d: np.ndarray) -> float:
    if slice_2d.size == 0:
        return 0.0
    hist, _ = np.histogram(slice_2d.ravel(), bins=64, range=(0, 255))
    total = max(hist.sum(), 1)
    probs = hist / total
    entropy = -float(np.sum(probs * np.log2(probs + 1e-9)))
    return entropy


def _variance_slice_score(slice_2d: np.ndarray) -> float:
    if slice_2d.size == 0:
        return 0.0
    return float(slice_2d.var())


def select_key_slices(
    slices: Sequence[np.ndarray],
    *,
    max_slices: int = 8,
    method: str = "entropy",
) -> list[np.ndarray]:
    if max_slices <= 0 or len(slices) == 0:
        return []

    if len(slices) <= max_slices:
        return list(slices)

    scores: list[tuple[float, int]] = []
    scorer = _entropy_slice_score if method == "entropy" else _variance_slice_score
    for idx, slice_2d in enumerate(slices):
        score = scorer(slice_2d)
        scores.append((score, idx))

    scores.sort(reverse=True)
    selected_indexes = [idx for _, idx in scores[:max_slices]]
    selected_indexes.sort()
    return [slices[i] for i in selected_indexes]


class LazyVolumeReader:
    def __init__(self, path: str | Path, *, max_slices: int = 0, slice_selection: str = "entropy") -> None:
        self.path = Path(path)
        self.max_slices = max_slices
        self.slice_selection = slice_selection
        self._slices: list[np.ndarray] = []
        self._loaded = False

    def __len__(self) -> int:
        return len(self._ensure_loaded())

    def __getitem__(self, index: int) -> np.ndarray:
        return self._ensure_loaded()[index]

    @property
    def shape(self) -> tuple[int, int, int]:
        slices = self._ensure_loaded()
        if not slices:
            return (0, 0, 0)
        h, w = slices[0].shape[:2]
        return (len(slices), h, w)

    def _ensure_loaded(self) -> list[np.ndarray]:
        if not self._loaded:
            self._slices = _load_volume_slices(self.path)
            if self.max_slices > 0 and len(self._slices) > self.max_slices:
                self._slices = select_key_slices(self._slices, max_slices=self.max_slices, method=self.slice_selection)
            self._loaded = True
        return self._slices

    def to_volume(self) -> np.ndarray:
        slices = self._ensure_loaded()
        if not slices:
            return np.zeros((0, 0, 0), dtype=np.uint8)
        return np.stack(slices, axis=0)

    def iter_slices(self):
        for slice_2d in self._ensure_loaded():
            yield slice_2d


def _load_volume_slices(path: Path) -> list[np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".gz" and path.name.lower().endswith(".nii.gz"):
        return _load_nifti_slices(path)
    if suffix == ".nii":
        return _load_nifti_slices(path)
    if suffix in {".mha", ".mhd"}:
        return _load_mhd_slices(path)
    if suffix == ".dcm":
        return _load_dicom_slice(path)
    if path.is_dir():
        return _load_dicom_series_slices(path)
    return []


def _load_nifti_slices(path: Path) -> list[np.ndarray]:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Can doc NIfTI can thu vien nibabel.") from exc

    img = nib.load(str(path))
    data = img.get_fdata()
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3:
        raise ValueError(f"NIfTI khong hop le: kich thuoc {data.shape}")

    slices: list[np.ndarray] = []
    for i in range(data.shape[2]):
        slice_2d = data[:, :, i]
        min_val, max_val = slice_2d.min(), slice_2d.max()
        if max_val > min_val:
            slice_2d = (slice_2d - min_val) / (max_val - min_val) * 255.0
        slices.append(slice_2d.astype(np.uint8))
    return slices


def _load_mhd_slices(path: Path) -> list[np.ndarray]:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("Can doc MHA/MHD can thu vien SimpleITK.") from exc

    img = sitk.ReadImage(str(path))
    data = sitk.GetArrayFromImage(img)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3:
        raise ValueError(f"MHA/MHD khong hop le: kich thuoc {data.shape}")

    slices: list[np.ndarray] = []
    for i in range(data.shape[0]):
        slice_2d = data[i]
        min_val, max_val = slice_2d.min(), slice_2d.max()
        if max_val > min_val:
            slice_2d = (slice_2d - min_val) / (max_val - min_val) * 255.0
        slices.append(slice_2d.astype(np.uint8))
    return slices


def _load_dicom_slice(path: Path) -> list[np.ndarray]:
    import pydicom

    ds = pydicom.dcmread(str(path), force=True)
    pixel_array = ds.pixel_array
    if pixel_array is None:
        return []

    if pixel_array.ndim == 2:
        pixel_array = np.stack([pixel_array] * 3, axis=-1)
    elif pixel_array.ndim == 3 and pixel_array.shape[-1] == 1:
        pixel_array = np.repeat(pixel_array, 3, axis=-1)
    elif pixel_array.ndim == 3 and pixel_array.shape[-1] == 3:
        pass
    else:
        pixel_array = pixel_array[..., 0]
        pixel_array = np.stack([pixel_array] * 3, axis=-1)

    if np.issubdtype(pixel_array.dtype, np.integer):
        info = np.iinfo(pixel_array.dtype)
        pixel_array = np.clip(pixel_array, info.min, info.max)
        pixel_array = pixel_array.astype(np.float32)
        pixel_array = (pixel_array - pixel_array.min()) / max(pixel_array.max() - pixel_array.min(), 1e-6) * 255.0
    else:
        pixel_array = np.clip(pixel_array, 0, 255)

    return [pixel_array.astype(np.uint8)]


def _load_dicom_series_slices(folder: Path) -> list[np.ndarray]:
    import pydicom

    files = sorted(folder.rglob("*.dcm"))
    slices: list[np.ndarray] = []
    for dcm_path in files:
        try:
            ds = pydicom.dcmread(str(dcm_path), force=True)
            if not hasattr(ds, "pixel_array"):
                continue
            pixel_array = ds.pixel_array
            if pixel_array.ndim == 2:
                pixel_array = np.stack([pixel_array] * 3, axis=-1)
            elif pixel_array.ndim == 3 and pixel_array.shape[-1] == 1:
                pixel_array = np.repeat(pixel_array, 3, axis=-1)
            elif pixel_array.ndim == 3 and pixel_array.shape[-1] == 3:
                pass
            else:
                continue

            if np.issubdtype(pixel_array.dtype, np.integer):
                info = np.iinfo(pixel_array.dtype)
                pixel_array = np.clip(pixel_array, info.min, info.max)
                pixel_array = pixel_array.astype(np.float32)
                pixel_array = (pixel_array - pixel_array.min()) / max(pixel_array.max() - pixel_array.min(), 1e-6) * 255.0
            else:
                pixel_array = np.clip(pixel_array, 0, 255)

            slices.append(pixel_array.astype(np.uint8))
        except Exception:
            continue
    return slices
