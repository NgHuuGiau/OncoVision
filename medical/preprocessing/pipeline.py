"""Medical image preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from medical.preprocessing.base import PreprocessingResult, _resize_and_pad, _to_uint8_rgb
from medical.preprocessing.ct import preprocess_ct
from medical.preprocessing.mri import preprocess_mri
from medical.preprocessing.xray import preprocess_xray
from medical.preprocessing.mammogram import preprocess_mammogram
from medical.preprocessing.pet import preprocess_pet_ct
from medical.preprocessing.endoscopy import preprocess_endoscopy
from medical.preprocessing.ultrasound import preprocess_ultrasound


_MODALITY_PREPROCESSORS: dict[str, Any] = {
    "ct": preprocess_ct,
    "mri": preprocess_mri,
    "xray": preprocess_xray,
    "mammogram": preprocess_mammogram,
    "pet_ct": preprocess_pet_ct,
    "endoscopy": preprocess_endoscopy,
    "ultrasound": preprocess_ultrasound,
}


def get_preprocessor(modality: str | None) -> Any:
    modality_key = (modality or "default").lower()
    if modality_key in _MODALITY_PREPROCESSORS:
        return _MODALITY_PREPROCESSORS[modality_key]
    if modality_key.startswith("pet"):
        return _MODALITY_PREPROCESSORS["pet_ct"]
    return _default_preprocessor


def preprocess_image(
    image: np.ndarray | str | Path,
    target_size: int = 320,
    modality: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PreprocessingResult:
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if image_path.is_file():
            try:
                image = _load_image_array(image_path)
            except Exception:
                with Image.open(image_path) as img:
                    image = np.array(img.convert("RGB"))
        else:
            raise FileNotFoundError(f"Khong tim thay anh: {image}")
    if image is None or image.size == 0:
        raise ValueError("Anh dau vao rong hoac None.")
    preprocessor = get_preprocessor(modality)
    return preprocessor(image, target_size=target_size, metadata=metadata)


def _load_image_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".dcm":
        try:
            import pydicom
            from pydicom.pixels import apply_voi_lut
            ds = pydicom.dcmread(str(path), force=True)
            array = ds.pixel_array.astype(np.float32)
            try:
                array = apply_voi_lut(array, ds)
            except Exception:
                pass
            if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
                array = np.max(array) - array
            array = array / max(array.max(), 1e-6) * 255
            return np.clip(array, 0, 255).astype(np.uint8)
        except Exception:
            pass
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _default_preprocessor(image: np.ndarray, target_size: int = 320, metadata: dict[str, Any] | None = None) -> PreprocessingResult:
    metadata = metadata or {}
    result = _to_uint8_rgb(image)
    result = _resize_and_pad(result, target_size)
    return PreprocessingResult(image=result, metadata=metadata, modality="default")
