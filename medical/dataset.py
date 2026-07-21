from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import numpy as np
from PIL import Image, ImageOps

from medical.cancer_catalog import COMMON_CANCER_TARGETS
from medical.classifier import iter_medical_image_paths
from medical.reporting import build_artifact_stamp
from utils.file_utils import save_yaml


MEDICAL_DATASET_ROOT = Path("dataset/medical")
MEDICAL_CLASS_NAMES = tuple(target.label for target in COMMON_CANCER_TARGETS)
MEDICAL_UPLOAD_EXTENSIONS = frozenset({
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".dcm",
    ".nii",
    ".nii.gz",
    ".mha",
    ".mhd",
})

_MODALITY_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CT ngực-bụng-chậu", ("ct nguc bung chau", "chest abdomen pelvis", "cap ct", "ct cap")),
    ("CT ngực", ("ct nguc", "chest ct", "thoracic ct", "ct chest")),
    ("CT gan", ("ct gan", "liver ct", "hepatic ct", "ct liver")),
    ("CT đại trực tràng", ("ct đại trực tràng", "ct colon", "rectal ct", "ct rectal")),
    ("CT dạ dày", ("ct dạ dày", "ct gastric", "stomach ct", "gastric ct")),
    ("CT tuyến tiền liệt", ("ct tuyến tiền liệt", "prostate ct")),
    ("CT cổ tử cung", ("ct cổ tử cung", "cervical ct", "pelvic ct")),
    ("MRI tuyến tiền liệt", ("mri tuyến tiền liệt", "prostate mri")),
    ("MRI trực tràng", ("mri truc trang", "rectal mri")),
    ("MRI vú", ("mri vú", "breast mri")),
    ("MRI gan", ("mri gan", "liver mri", "hepatic mri")),
    ("MRI dạ dày", ("mri dạ dày", "mri gastric", "stomach mri")),
    ("MRI cổ tử cung", ("mri cổ tử cung", "cervical mri", "pelvic mri")),
    ("Siêu âm vú", ("sieu am vu", "breast ultrasound")),
    ("Siêu âm gan", ("siêu âm gan", "liver ultrasound", "hepatic ultrasound", "fascio")),
    ("Siêu âm tuyến tiền liệt", ("siêu âm tuyến tiền liệt", "prostate ultrasound", "trus")),
    ("X-quang ngực", ("x quang nguc", "chest x ray", "cxr", "chest radiograph")),
    ("Nội soi đại tràng", ("noi soi dai trang", "colonoscopy")),
    ("Mammogram", ("mammogram", "mammo")),
    ("EUS", ("eus", "endoscopic ultrasound")),
    ("PET/CT", ("pet ct", "petct", "pet/ct")),
    ("PET", (" pet ", " pet-", "pet scan", "pet whole body")),
    ("Nội soi", ("noi soi", "endoscopy", "gastroscopy", "egd")),
    ("Siêu âm", ("sieu am", "ultrasound", "sonography")),
    ("CT", (" computed tomography ", " ct ", "ct scan", "ctscan")),
    ("MRI", (" magnetic resonance ", " mri ", "mri scan")),
)

_TARGET_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("liver", ("gan", "liver", "hepatic", "hepat", "hcc", "liver cancer", "liver lesion", "liver tumor", "liver mass", "liver metastasis")),
    ("lung", ("phổi", "lung", "thorax", "thoracic", "chest", "pulmonary", "lung cancer", "lung lesion", "lung tumor", "lung nodule", "lung mass", "pulmonary nodule", "pulmonary mass")),
    ("breast", ("vú", "breast", "mammogram", "mammo", "breast cancer", "breast lesion", "breast mass", "breast tumor", "breast nodule")),
    ("stomach", ("dạ dày", "stomach", "gastric", "gast", "egd", "endoscopy", "nội soi", "stomach cancer", "gastric cancer", "gastric lesion", "gastric tumor", "gastric mass")),
    ("colorectal", ("đại trực tràng", "colorectal", "colon", "rectal", "trực tràng", "colonoscopy", "colorectal cancer", "colon cancer", "rectal cancer", "colon lesion", "rectal lesion", "colon tumor", "rectal tumor")),
    ("prostate", ("tuyến tiền liệt", "prostate", "prostatic", "prostate cancer", "prostate lesion", "prostate tumor", "prostate mass")),
    ("cervical", ("cổ tử cung", "cervical", "cervix", "pap", "hpv", "colposcopy", "cervical cancer", "cervix cancer", "cervical lesion", "cervical tumor")),
)

_MODALITY_TO_TARGET_KEY: dict[str, str] = {
    "Mammogram": "breast",
    "Siêu âm vú": "breast",
    "MRI vú": "breast",
    "X-quang ngực": "lung",
    "CT ngực": "lung",
    "MRI trực tràng": "colorectal",
    "Nội soi đại tràng": "colorectal",
    "CT ngực-bụng-chậu": "colorectal",
    "EUS": "stomach",
    "Nội soi": "stomach",
    "CT gan": "liver",
    "MRI gan": "liver",
    "Siêu âm gan": "liver",
    "CT đại trực tràng": "colorectal",
    "MRI đại trực tràng": "colorectal",
    "CT dạ dày": "stomach",
    "MRI dạ dày": "stomach",
    "CT tuyến tiền liệt": "prostate",
    "MRI tuyến tiền liệt": "prostate",
    "CT cổ tử cung": "cervical",
    "MRI cổ tử cung": "cervical",
    "PET/CT gan": "liver",
    "PET/CT phổi": "lung",
    "PET/CT đại trực tràng": "colorectal",
    "PET/CT dạ dày": "stomach",
    "PET/CT tuyến tiền liệt": "prostate",
    "PET/CT cổ tử cung": "cervical",
    "PET gan": "liver",
    "PET phổi": "lung",
    "PET đại trực tràng": "colorectal",
}

_DICOM_MODALITY_MAP: dict[str, str] = {
    "CT": "CT",
    "MR": "MRI",
    "US": "Siêu âm",
    "PT": "PET",
    "MG": "Mammogram",
    "CR": "X-quang ngực",
    "DX": "X-quang ngực",
    "XA": "X-quang ngực",
}

_DICOM_BODY_PART_TO_TARGET: dict[str, str] = {
    "LIVER": "liver",
    "HEPAT": "liver",
    "CHEST": "lung",
    "LUNG": "lung",
    "BREAST": "breast",
    "CHESTABDPELV": "colorectal",
    "ABDOMEN": "stomach",
    "STOMACH": "stomach",
    "ABDOMENPELVIS": "colorectal",
    "PELVIS": "prostate",
    "PROSTATE": "prostate",
    "RECTUM": "colorectal",
    "COLON": "colorectal",
    "CERVIX": "cervical",
    "UTERUS": "cervical",
    "PELVISNECK": "cervical",
    "WHOLEBODY": "lung",
}

SUPPORTED_MEDICAL_MODALITIES_BY_TARGET_KEY: dict[str, tuple[str, ...]] = {
    "liver": ("ct", "mri", "ultrasound", "pet_ct"),
    "lung": ("xray", "ct", "pet_ct"),
    "breast": ("mammogram", "ultrasound", "mri"),
    "stomach": ("endoscopy", "ct", "mri", "pet_ct", "eus"),
    "colorectal": ("colonoscopy", "ct", "mri", "pet_ct"),
    "prostate": ("mri", "ultrasound", "pet_ct"),
    "cervical": ("ct", "mri", "pet_ct"),
}


@dataclass(frozen=True)
class MedicalDatasetConfig:
    disease_name: str
    dataset_root: Path
    data_yaml_path: Path
    metadata_dir: Path
    reports_dir: Path
    class_names: tuple[str, ...]
    image_size: int


@dataclass(frozen=True)
class MedicalDatasetSummary:
    dataset_root: Path
    created_directories: list[Path]
    data_yaml_path: Path


def create_default_medical_dataset_config(dataset_root: str | Path = MEDICAL_DATASET_ROOT) -> MedicalDatasetConfig:
    root = Path(dataset_root)
    return MedicalDatasetConfig(
        disease_name="medical_7_cancers",
        dataset_root=root,
        data_yaml_path=root / "data.yaml",
        metadata_dir=root / "metadata",
        reports_dir=root / "reports",
        class_names=MEDICAL_CLASS_NAMES,
        image_size=320,
    )


def ensure_medical_dataset_structure(config: MedicalDatasetConfig | None = None) -> MedicalDatasetSummary:
    config = config or create_default_medical_dataset_config()
    created_dirs = [config.dataset_root, config.metadata_dir, config.reports_dir]
    for class_name in config.class_names:
        for split in ("train", "val", "test"):
            created_dirs.append(config.dataset_root / class_name / "processed" / "images" / split)
    for directory in created_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    save_yaml(
        config.data_yaml_path,
        {
            "path": str(config.dataset_root.resolve()),
            "task": "classification",
            "train": str(config.dataset_root.resolve()),
            "val": str(config.dataset_root.resolve()),
            "test": str(config.dataset_root.resolve()),
            "names": {index: name for index, name in enumerate(config.class_names)},
        },
    )
    return MedicalDatasetSummary(
        dataset_root=config.dataset_root,
        created_directories=created_dirs,
        data_yaml_path=config.data_yaml_path,
    )


def iter_medical_class_split_images(dataset_root: str | Path, split: str) -> dict[str, list[Path]]:
    root = Path(dataset_root)
    split_map: dict[str, list[Path]] = {}
    for class_name in MEDICAL_CLASS_NAMES:
        split_dir = root / class_name / "processed" / "images" / split
        split_map[class_name] = list(iter_medical_image_paths(split_dir))
    return split_map


def count_medical_class_split_images(dataset_root: str | Path, split: str) -> dict[str, int]:
    return {class_name: len(paths) for class_name, paths in iter_medical_class_split_images(dataset_root, split).items()}


def is_supported_medical_upload_path(path: str | Path) -> bool:
    source = Path(path)
    if source.is_dir():
        return any(candidate.is_file() and _medical_upload_suffix(candidate) in MEDICAL_UPLOAD_EXTENSIONS for candidate in source.rglob("*"))
    return _medical_upload_suffix(source) in MEDICAL_UPLOAD_EXTENSIONS


def _medical_upload_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def _normalize_medical_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def _collect_medical_text(path: Path) -> str:
    parts = [path.name, *path.parts[-3:]]
    if path.is_dir():
        try:
            candidate = resolve_medical_upload_path(path)
        except Exception:
            candidate = None
        if candidate is not None and candidate.is_file() and _medical_upload_suffix(candidate) == ".dcm":
            parts.append(_collect_dicom_text(candidate))
        return " ".join(parts)

    if _medical_upload_suffix(path) == ".dcm":
        parts.append(_collect_dicom_text(path))
    return " ".join(parts)


def _collect_dicom_text(path: Path) -> str:
    try:
        import pydicom
    except ImportError:  # pragma: no cover
        return ""

    try:
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return ""

    values: list[str] = []
    for key in ("Modality", "SeriesDescription", "ProtocolName", "StudyDescription", "BodyPartExamined"):
        value = getattr(dataset, key, "")
        if value:
            values.append(str(value))
    return " ".join(values)


def _find_first_matching_hint(text: str, hints: tuple[tuple[str, tuple[str, ...]], ...]) -> str | None:
    normalized = f" { _normalize_medical_text(text) } "
    for label, terms in hints:
        if any(term in normalized for term in terms):
            return label
    return None


def infer_medical_upload_context(path: str | Path) -> tuple[str | None, str | None]:
    source = Path(path)
    raw_text = _collect_medical_text(source)
    normalized = _normalize_medical_text(raw_text)

    modality = _infer_medical_modality(source, normalized)
    target_key = _infer_medical_target_key(normalized, modality)
    return target_key, modality


def supported_medical_modalities_for_target(target_key: str | None) -> tuple[str, ...]:
    if target_key is None:
        return ()
    return SUPPORTED_MEDICAL_MODALITIES_BY_TARGET_KEY.get(target_key, ())


def _infer_medical_modality(source: Path, normalized_text: str) -> str | None:
    padded = f" {normalized_text} "
    for label, terms in _MODALITY_HINTS:
        if any(term in padded for term in terms):
            return label

    if source.is_file() and _medical_upload_suffix(source) == ".dcm":
        pydicom = None
        try:
            import pydicom  # type: ignore[assignment]
        except ImportError:  # pragma: no cover
            pass
        if pydicom is not None:
            try:
                dataset = pydicom.dcmread(str(source), stop_before_pixels=True, force=True)
                dicom_modality = str(getattr(dataset, "Modality", "")).upper()
                mapped = _DICOM_MODALITY_MAP.get(dicom_modality)
                if mapped:
                    return mapped
            except Exception:
                pass
    return None


def _infer_medical_target_key(normalized_text: str, modality: str | None) -> str | None:
    if modality and modality in _MODALITY_TO_TARGET_KEY:
        return _MODALITY_TO_TARGET_KEY[modality]
    padded = f" {normalized_text} "
    for target_key, terms in _TARGET_HINTS:
        if any(term in padded for term in terms):
            return target_key
    return None


def resolve_medical_upload_path(path: str | Path) -> Path:
    source = Path(path)
    if source.is_file():
        if not is_supported_medical_upload_path(source):
            raise ValueError(f"Khong ho tro file upload: {source}")
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"Khong tim thay duong dan upload: {source}")

    candidates = sorted(
        candidate
        for candidate in source.rglob("*")
        if candidate.is_file() and _medical_upload_suffix(candidate) in MEDICAL_UPLOAD_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"Thư mục không chứa ảnh hợp lệ: {source}")

    dicom_candidates = [candidate for candidate in candidates if _medical_upload_suffix(candidate) == ".dcm"]
    if dicom_candidates:
        return dicom_candidates[len(dicom_candidates) // 2]
    return candidates[0]


def _normalize_array_to_rgb(array: np.ndarray) -> Image.Image:
    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 0:
        data = data.reshape(1, 1)
    if data.ndim > 3:
        data = np.squeeze(data)
    if data.ndim == 2:
        minimum = float(np.min(data))
        maximum = float(np.max(data))
        if maximum > minimum:
            data = (data - minimum) / (maximum - minimum)
        else:
            data = np.zeros_like(data)
        data = (data * 255.0).clip(0, 255).astype(np.uint8)
        data = np.stack([data] * 3, axis=-1)
        return Image.fromarray(data, mode="RGB")
    if data.ndim == 3 and data.shape[-1] in {3, 4}:
        if data.shape[-1] == 4:
            data = data[..., :3]
        minimum = float(np.min(data))
        maximum = float(np.max(data))
        if maximum > minimum:
            data = (data - minimum) / (maximum - minimum)
        data = (data * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(data, mode="RGB")
    return Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8), mode="RGB")


def _load_dicom_slice(source: Path) -> Image.Image:
    try:
        import pydicom
        from pydicom.pixels import apply_voi_lut
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Không đọc được DICOM. Hãy cài đặt pydicom để mở file .dcm.") from exc

    dataset = pydicom.dcmread(str(source), force=True)
    array = dataset.pixel_array.astype(np.float32)
    try:
        array = apply_voi_lut(array, dataset)
    except Exception:
        pass

    if getattr(dataset, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        array = np.max(array) - array

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    array = array * slope + intercept
    return _normalize_array_to_rgb(array)


def _load_nifti_volume(source: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Không đọc được NIfTI. Hãy cài đặt nibabel để mở file .nii/.nii.gz.") from exc

    image = nib.load(str(source))
    volume = np.asanyarray(image.dataobj)  # type: ignore[attr-defined]
    if volume.ndim == 4:
        volume = volume[..., 0]
    return np.asarray(volume)


def _load_mha_volume(source: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Không đọc được MHA/MHD. Hãy cài đặt SimpleITK để mở file .mha/.mhd.") from exc

    image = sitk.ReadImage(str(source))
    return sitk.GetArrayFromImage(image)


def _load_dicom_series_volume(source: Path) -> np.ndarray:
    try:
        import pydicom
        from pydicom.pixels import apply_voi_lut
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Không đọc được DICOM. Hãy cài đặt pydicom để mở thư mục series .dcm.") from exc

    slice_entries: list[tuple[float, np.ndarray]] = []
    for candidate in sorted(source.rglob("*")):
        if not candidate.is_file() or _medical_upload_suffix(candidate) != ".dcm":
            continue
        dataset = pydicom.dcmread(str(candidate), force=True)
        array = dataset.pixel_array.astype(np.float32)
        try:
            array = apply_voi_lut(array, dataset)
        except Exception:
            pass
        if getattr(dataset, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
            array = np.max(array) - array
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        array = array * slope + intercept
        order = float(getattr(dataset, "InstanceNumber", len(slice_entries)))
        slice_entries.append((order, array))

    if not slice_entries:
        raise FileNotFoundError(f"Thư mục không chứa ảnh DICOM hợp lệ: {source}")

    slice_entries.sort(key=lambda item: item[0])
    slices = []
    for _, array in slice_entries:
        if array.ndim > 2:
            array = np.squeeze(array)
            if array.ndim > 2:
                array = array[..., 0]
        slices.append(np.asarray(array, dtype=np.float32))
    return np.stack(slices, axis=0)


def _load_medical_volume_image(source: Path) -> Image.Image:
    if source.is_dir():
        volume = _load_dicom_series_volume(source)
    else:
        suffix = _medical_upload_suffix(source)
        if suffix == ".nii.gz" or suffix == ".nii":
            volume = _load_nifti_volume(source)
        elif suffix in {".mha", ".mhd"}:
            volume = _load_mha_volume(source)
        elif suffix == ".dcm":
            return _load_dicom_slice(source)
        else:
            raise ValueError(f"Khong ho tro dinh dang volume: {source}")

    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim == 2:
        return _normalize_array_to_rgb(volume)
    if volume.ndim > 3:
        volume = np.squeeze(volume)
    if volume.ndim == 2:
        return _normalize_array_to_rgb(volume)
    if volume.ndim != 3:
        return _normalize_array_to_rgb(volume)

    tiles = _volume_to_rgb_slices(volume)
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    tile_count = len(tiles)
    columns = 3 if tile_count > 4 else 2 if tile_count > 1 else 1
    rows = (tile_count + columns - 1) // columns
    montage = Image.new("RGB", (columns * tile_width, rows * tile_height), (0, 0, 0))
    for index, tile in enumerate(tiles):
        row = index // columns
        column = index % columns
        x = column * tile_width + (tile_width - tile.width) // 2
        y = row * tile_height + (tile_height - tile.height) // 2
        montage.paste(tile, (x, y))
    return montage


def _volume_to_rgb_slices(volume: np.ndarray) -> list[Image.Image]:
    data = np.asarray(volume, dtype=np.float32)
    if data.ndim > 3:
        data = np.squeeze(data)
    if data.ndim == 2:
        return [_normalize_array_to_rgb(data)]
    if data.ndim != 3:
        return [_normalize_array_to_rgb(data)]
    slice_axis = int(np.argmin(data.shape))
    if slice_axis != 0:
        data = np.moveaxis(data, slice_axis, 0)
    return [_normalize_array_to_rgb(data[index]) for index in range(data.shape[0])]


def _load_dicom_image(source: Path) -> Image.Image:
    return _load_dicom_slice(source)


def load_medical_volume_slices(source_path: str | Path) -> list[Image.Image]:
    source = Path(source_path)
    if source.is_dir():
        volume = _load_dicom_series_volume(source)
    else:
        suffix = _medical_upload_suffix(source)
        if suffix in {".nii", ".nii.gz"}:
            volume = _load_nifti_volume(source)
        elif suffix in {".mha", ".mhd"}:
            volume = _load_mha_volume(source)
        elif suffix == ".dcm":
            return [_load_dicom_slice(source)]
        else:
            return []
    return _volume_to_rgb_slices(volume)


def is_medical_volume_source(path: str | Path) -> bool:
    source = Path(path)
    if source.is_dir():
        return True
    return _medical_upload_suffix(source) in {".dcm", ".nii", ".nii.gz", ".mha", ".mhd"}


def load_medical_source_image(source_path: str | Path) -> Image.Image:
    source = Path(source_path)
    if is_medical_volume_source(source):
        return _load_medical_volume_image(source)
    with Image.open(source) as image:
        normalized = ImageOps.exif_transpose(image)
        if normalized.mode not in {"RGB", "L"}:
            if "A" in normalized.getbands():
                canvas = Image.new("RGBA", normalized.size, (0, 0, 0, 0))
                canvas.alpha_composite(normalized.convert("RGBA"))
                normalized = canvas.convert("RGB")
            else:
                normalized = normalized.convert("RGB")
        else:
            normalized = normalized.convert("RGB")
        return normalized.copy()


def normalize_uploaded_image(
    source_path: str | Path,
    target_dir: str | Path,
    *,
    image_size: int = 320,
) -> Path:
    source = Path(source_path)
    if not is_supported_medical_upload_path(source):
        raise ValueError(f"Khong ho tro file upload: {source}")
    destination_dir = Path(target_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    normalized = load_medical_source_image(source)
    normalized.thumbnail((image_size, image_size))
    background = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    offset = ((image_size - normalized.width) // 2, (image_size - normalized.height) // 2)
    background.paste(normalized, offset)
    target_path = destination_dir / f"{source.stem}_{build_artifact_stamp()}.jpg"
    background.save(target_path, format="JPEG", quality=95)
    return target_path
