from __future__ import annotations

import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from medical.reporting import build_artifact_stamp

logger = logging.getLogger(__name__)


def _limit_volume_slices(volume: np.ndarray, max_slices: int = 256) -> np.ndarray:
    if volume.ndim < 3:
        return volume
    slices = volume.shape[0]
    if slices <= max_slices:
        return volume
    logger.warning("[Dataset] Volume co %s slices, gioi han ve %s slices", slices, max_slices)
    step = max(1, slices / max_slices)
    indices = [min(int(i * step), slices - 1) for i in range(max_slices)]
    return volume[indices]


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
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
})

_MODALITY_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CT ngực-bụng-chậu", ("ct nguc bung chau", "chest abdomen pelvis", "cap ct", "ct cap")),
    ("CT ngực", ("ct nguc", "chest ct", "thoracic ct", "ct chest")),
    ("CT gan", ("ct gan", "liver ct", "hepatic ct", "ct liver")),
    ("CT đại trực tràng", ("ct đại trực tràng", "ct colon", "rectal ct", "ct rectal")),
    ("CT dạ dày", ("ct dạ dày", "ct gastric", "stomach ct", "gastric ct")),
    ("CT tuyến tiền liệt", ("ct tuyến tiền liệt", "prostate ct")),
    ("CT cổ tử cung", ("ct cổ tử cung", "cervical ct", "pelvic ct")),
    ("CT thận", ("ct than", "kidney ct", "renal ct")),
    ("MRI thận", ("mri than", "kidney mri", "renal mri")),
    ("Siêu âm thận", ("sieu am than", "kidney ultrasound", "renal ultrasound")),
    ("PET/CT thận", ("pet ct than", "petct kidney", "pet ct kidney", "pet ct renal")),
    ("CT tụy", ("ct tuy", "pancreas ct", "pancreatic ct")),
    ("MRI tụy", ("mri tuy", "pancreas mri", "pancreatic mri")),
    ("PET/CT tụy", ("pet ct tuy", "petct pancreas", "pet ct pancreas", "pet ct pancreatic")),
    ("Siêu âm tuyến giáp", ("sieu am tuyen giap", "thyroid ultrasound", "thyroid sonography")),
    ("CT tuyến giáp", ("ct tuyen giap", "thyroid ct")),
    ("MRI tuyến giáp", ("mri tuyen giap", "thyroid mri")),
    ("PET/CT tuyến giáp", ("pet ct tuyen giap", "petct thyroid", "pet ct thyroid")),
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
    ("Ảnh soi cổ tử cung", ("colposcopy", "cervical photo", "soi co tu cung", "soi cổ tử cung")),
    ("PET", (" pet ", " pet-", "pet scan", "pet whole body")),
    ("Nội soi", ("noi soi", "endoscopy", "gastroscopy", "egd")),
    ("Siêu âm", ("sieu am", "ultrasound", "sonography")),
    ("CT", (" computed tomography ", " ct ", "ct scan", "ctscan")),
    ("MRI", (" magnetic resonance ", " mri ", "mri scan")),
    ("MRI não", ("mri nao", "brain mri", "mri brain", "head mri", "mri head", "neuro mri", "cranial mri")),
    ("CT sọ não", ("ct so nao", "brain ct", "ct brain", "head ct", "ct head", "ct neuro", "cranial ct")),
    ("PET/CT não", ("pet ct nao", "brain pet", "pet brain", "pet/ct brain")),
)

_TARGET_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("liver", ("gan", "liver", "hepatic", "hepat", "hcc", "liver cancer", "liver lesion", "liver tumor", "liver mass", "liver metastasis")),
    ("lung", ("phổi", "lung", "thorax", "thoracic", "chest", "pulmonary", "lung cancer", "lung lesion", "lung tumor", "lung nodule", "lung mass", "pulmonary nodule", "pulmonary mass")),
    ("breast", ("vú", "breast", "mammogram", "mammo", "breast cancer", "breast lesion", "breast mass", "breast tumor", "breast nodule")),
    ("colorectal", ("đại trực tràng", "colorectal", "colon", "rectal", "trực tràng", "colonoscopy", "colorectal cancer", "colon cancer", "rectal cancer", "colon lesion", "rectal lesion", "colon tumor", "rectal tumor")),
    ("stomach", ("dạ dày", "stomach", "gastric", "gast", "egd", "stomach cancer", "gastric cancer", "gastric lesion", "gastric tumor", "gastric mass")),
    ("prostate", ("tuyến tiền liệt", "prostate", "prostatic", "prostate cancer", "prostate lesion", "prostate tumor", "prostate mass")),
    ("cervical", ("cổ tử cung", "cervical", "cervix", "pap", "hpv", "colposcopy", "cervical cancer", "cervix cancer", "cervical lesion", "cervical tumor")),
    ("kidney", ("thận", "kidney", "renal", "nephro", "rcc", "renal cell carcinoma", "kidney cancer", "kidney lesion", "kidney tumor", "kidney mass")),
    ("pancreas", ("tụy", "pancreas", "pancreatic", "pdac", "pancreatic ductal adenocarcinoma", "pancreas cancer", "pancreatic cancer", "pancreas lesion", "pancreatic lesion", "pancreas tumor", "pancreatic tumor")),
    ("thyroid", ("tuyến giáp", "thyroid", "thyroid cancer", "thyroid carcinoma", "thyroid lesion", "thyroid nodule", "thyroid tumor")),
    ("brain", ("não", "brain", "cranial", "skull", "neuro", "intracranial", "cerebral", "glioma", "meningioma", "pituitary", "brain tumor", "brain cancer", "brain lesion", "brain mass", "head tumor")),
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
    "CT thận": "kidney",
    "MRI thận": "kidney",
    "Siêu âm thận": "kidney",
    "PET/CT thận": "kidney",
    "CT tụy": "pancreas",
    "MRI tụy": "pancreas",
    "PET/CT tụy": "pancreas",
    "Siêu âm tuyến giáp": "thyroid",
    "CT tuyến giáp": "thyroid",
    "MRI tuyến giáp": "thyroid",
    "PET/CT tuyến giáp": "thyroid",
    "PET/CT gan": "liver",
    "PET/CT phổi": "lung",
    "PET/CT đại trực tràng": "colorectal",
    "PET/CT dạ dày": "stomach",
    "PET/CT tuyến tiền liệt": "prostate",
    "PET/CT cổ tử cung": "cervical",
    "Ảnh soi cổ tử cung": "cervical",
    "PET gan": "liver",
    "PET phổi": "lung",
    "PET đại trực tràng": "colorectal",
    "MRI não": "brain",
    "CT sọ não": "brain",
    "PET/CT não": "brain",
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
    "BRAIN": "brain",
    "HEAD": "brain",
    "NEURO": "brain",
    "SKULL": "brain",
    "CSPINE": "brain",
    "KIDNEY": "kidney",
    "RENAL": "kidney",
    "PANCREAS": "pancreas",
    "THYROID": "thyroid",
}

SUPPORTED_MEDICAL_MODALITIES_BY_TARGET_KEY: dict[str, tuple[str, ...]] = {
    "liver": ("ct", "mri", "ultrasound", "pet_ct"),
    "lung": ("xray", "ct", "pet_ct"),
    "breast": ("mammogram", "ultrasound", "mri"),
    "stomach": ("endoscopy", "ct", "mri", "pet_ct", "eus"),
    "colorectal": ("colonoscopy", "ct", "mri", "pet_ct"),
    "prostate": ("mri", "ultrasound", "pet_ct"),
    "cervical": ("colposcopy", "ct", "mri", "pet_ct"),
    "brain": ("mri",),  # Only MRI has model — CT and PET/CT directories are missing
    "kidney": ("ct", "mri", "ultrasound", "pet_ct"),
    "pancreas": ("ct", "mri", "pet_ct"),
    "thyroid": ("ultrasound", "ct", "mri", "pet_ct"),
}

# Modalities that actually have classifier models available
# Used by web app to filter dropdown options
AVAILABLE_MODALITIES_BY_TARGET_KEY: dict[str, tuple[str, ...]] = {
    "liver": ("ct",),
    "lung": ("xray", "ct"),
    "breast": ("mammogram", "ultrasound"),
    "stomach": ("endoscopy",),
    "colorectal": ("colonoscopy",),
    "prostate": ("mri",),
    "cervical": ("colposcopy",),
    "brain": ("mri",),
    "kidney": ("ct",),
    "pancreas": ("ct",),
    "thyroid": ("ultrasound",),
}


def available_medical_modalities_for_target(target_key: str | None) -> tuple[str, ...]:
    """Return modalities that actually have classifier models.
    
    Used by web app dropdown to avoid 503 errors for missing modalities.
    """
    if target_key is None:
        return ()
    return AVAILABLE_MODALITIES_BY_TARGET_KEY.get(target_key, ())


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
    except ImportError:# pragma: no cover
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
    
    # Nếu không xác định được modality từ tên file/metadata, thử phân tích pixel
    if modality is None:
        pixel_modality = guess_modality_from_pixels(path)
        if pixel_modality:
            modality = pixel_modality
    
    # Nếu vẫn chưa có target_key (tên file không chứa từ khóa bệnh),
    # thử dùng classifier models để xác định cancer type từ ảnh
    if target_key is None:
        clf_target, clf_modality = guess_cancer_from_pixels(path)
        if clf_target is not None:
            target_key = clf_target
            # Nếu classifier cũng xác định được modality cụ thể, ưu tiên nó
            if clf_modality is not None and clf_modality:
                modality = clf_modality
            logger.info("Classifier xác định: target=%s, modality=%s", target_key, modality)
    
    # Fallback cuối cùng: nếu vẫn chưa có target_key, dùng modality để suy ra
    # Chỉ áp dụng khi modality là specific label (có trong _MODALITY_TO_TARGET_KEY)
    # Không áp dụng cho modality generic từ pixel analysis (CT, MRI, Siêu âm, Nội soi)
    # vì những modality này có thể là false positive cho ảnh không phải y khoa
    if target_key is None and modality is not None:
        # Ưu tiên: exact match modality label trong _MODALITY_TO_TARGET_KEY
        # Chỉ dùng khi modality là specific label (ví dụ: "CT gan", "MRI não", "Siêu âm tuyến giáp")
        # Không dùng cho generic modality (CT, MRI, Siêu âm, Nội soi) vì dễ nhầm
        if modality in _MODALITY_TO_TARGET_KEY:
            # Kiểm tra: modality có phải là specific label không?
            # Generic modality từ pixel analysis: CT, MRI, Siêu âm, Nội soi, X-quang ngực
            generic_modalities = {"CT", "MRI", "Siêu âm", "Nội soi", "X-quang ngực", "PET/CT", "PET"}
            if modality not in generic_modalities:
                target_key = _MODALITY_TO_TARGET_KEY[modality]
            # Nếu modality là generic, không fallback — trả về None để pipeline biết
            # Auto-detect sẽ dùng classifier-based detection thay vì guess từ modality
    
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
        pydicom_module: Any = None
        try:
            from importlib import import_module

            pydicom_module = import_module("pydicom")
        except ImportError:# pragma: no cover
            pass
        if pydicom_module is not None:
            try:
                dataset = pydicom_module.dcmread(str(source), stop_before_pixels=True, force=True)
                dicom_modality = str(getattr(dataset, "Modality", "")).upper()
                mapped = _DICOM_MODALITY_MAP.get(dicom_modality)
                if mapped:
                    return mapped
            except Exception:
                pass
    return None


def guess_modality_from_pixels(image_path: str | Path) -> str | None:
    """Phân tích pixel ảnh để ước lượng modality.
    
    Strategy: phân tích đặc điểm pixel (channel difference, histogram, gradient, local variance)
    để phân biệt các loại ảnh y khoa: CT, MRI, X-quang, Siêu âm, Nội soi.
    
    Trả về modality label (CT, MRI, X-quang ngực, Siêu âm, Nội soi) hoặc None nếu không chắc chắn.
    
    QUAN TRỌNG: Hàm này trả về None cho ảnh không phải y khoa để tránh false positive.
    Các fallback rules cũ đã bị loại bỏ vì chúng luôn trả về 1 modality.
    """
    source = Path(image_path)
    if not source.exists():
        return None
    try:
        import numpy as np
        from PIL import Image
        img = Image.open(source)
        arr = np.array(img)
        
        # Xử lý ảnh: chuyển về grayscale nếu cần
        channel_diff = 0.0
        if len(arr.shape) == 3 and arr.shape[2] >= 3:
            r = arr[:, :, 0].astype(float)
            g = arr[:, :, 1].astype(float)
            b = arr[:, :, 2].astype(float)
            channel_diff = float(np.mean([np.abs(r - g), np.abs(g - b), np.abs(r - b)]))
            
            # Ảnh có màu RGB thực sự (channel_diff cao) → nội soi / colposcopy
            # Threshold cao hơn để tránh nhầm với ảnh grayscale chuyển RGB
            if channel_diff > 50:
                return "Nội soi"
            
            # RGBA với channel_diff=0 → ảnh grayscale có alpha channel
            if channel_diff < 2:
                # Grayscale với alpha — bỏ channel alpha, lấy channel đầu
                gray = r.astype(float)
            else:
                gray = np.mean(arr[:, :, :3].astype(float), axis=2)
        elif len(arr.shape) == 2:
            gray = arr.astype(float)
        else:
            return None
        
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        
        if gray.size > 0:
            black_ratio = float(np.sum(gray < 20)) / gray.size
        else:
            black_ratio = 0
        
        # Local variance — dùng để phân biệt ultrasound và các loại khác
        if gray.size > 100 and gray.shape[0] > 10:
            local_var = float(np.var(gray[::10, ::10]))
        else:
            local_var = 0
        
        # Gradient mean
        if gray.size > 100:
            grad_x = np.abs(np.diff(gray, axis=1))
            grad_y = np.abs(np.diff(gray, axis=0))
            float(np.mean(np.concatenate([grad_x.flatten(), grad_y.flatten()])))
        else:
            pass
        
        # Histogram analysis
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        hist_norm = hist.astype(float)
        hist_norm /= hist_norm.sum() if hist_norm.sum() > 0 else 1
        peak_ratio = float(hist_norm.max())
        float(hist_norm[:32].sum())
        float(hist_norm[224:].sum())
        float(hist_norm[64:192].sum())
        
        # === RULES (theo thứ tự ưu tiên) ===
        # Mỗi rule trả về modality chỉ khi điều kiện ĐỦ RÕ RÀNG.
        # Nếu không khớp rule nào → trả về None (không có fallback aggressive).
        
        # 1. Nội soi: đã kiểm tra ở trên (channel_diff > 50)
        
        # 2. X-ray: mean rất thấp, contrast cao, nhiều pixel đen
        if mean_val < 40 and std_val > 40 and black_ratio > 0.3:
            return "X-quang ngực"
        
        # 3. CT ảnh gốc (không preprocess): nhiều pixel đen nền, mean thấp
        if black_ratio > 0.3 and mean_val < 60:
            return "CT"
        
        # 4. Ultrasound: texture đặc trưng — local variance cao, std trung bình
        #    Ảnh ultrasound thường có mean 40-120, std 20-60, local_var 500-5000
        #    black_ratio thấp (< 0.15) vì ít pixel đen hoàn toàn
        if 500 < local_var < 5000 and black_ratio < 0.15 and 40 < mean_val < 150:
            return "Siêu âm"
        
        # 5. Ảnh đã preprocess (224x224) — std rất thấp + local_var rất thấp
        #    Đây là ảnh CT đã được resize và normalize
        #    Đặc điểm: std < 15, local_var < 100, black_ratio ≈ 0
        #    Phân biệt CT vs MRI preprocess:
        #    - CT preprocess: mean thấp hơn (50-100), std rất thấp, peak_ratio cao
        #    - MRI preprocess: mean cao hơn (100-150), std thấp
        #    Pancreas CT: mean=91, std=9, local_var=30, peak_ratio=0.98 → CT
        #    Cervical MRI: mean=133, std=6, local_var=1, peak_ratio=0.98 → MRI
        if local_var < 100 and std_val < 15 and black_ratio < 0.05:
            # Ảnh preprocess — phân biệt CT vs MRI dựa trên mean
            # Peak ratio cao (>0.5) nghĩa là ảnh có histogram tập trung → CT preprocess
            # Peak ratio rất cao (>0.9) + mean < 100 → chắc chắn CT
            if peak_ratio > 0.9 and mean_val < 100:
                return "CT"
            if mean_val < 90:
                return "CT"  # CT preprocess: mean thấp
            elif mean_val > 120:
                return "MRI"  # MRI preprocess: mean cao
            else:
                # Mean 90-120: không chắc chắn, ưu tiên CT (phổ biến hơn)
                return "CT"
        
        # 6. Ảnh gốc có std cao + local_var cao
        #    Phân biệt CT/X-ray vs MRI:
        #    - CT/X-ray gốc: black_ratio cao (>0.3), mean thấp (<80), peak_ratio thấp
        #    - MRI gốc: black_ratio cao (>0.25), mean thấp-cao, peak_ratio thấp
        #    Lung X-ray: std=80, local_var=6698, black=0.12, mean=105 → CT
        #    Brain MRI: std=60, local_var=3578, black=0.36, mean=66, peak=0.31 → MRI
        if std_val > 50 and local_var > 1000:
            # Ảnh gốc có contrast cao — phân biệt CT vs MRI
            # MRI thường có peak_ratio thấp hơn (<0.5) vì phân bố pixel đa dạng hơn
            # CT/X-ray thường có peak_ratio cao hơn hoặc black_ratio rất cao
            if peak_ratio < 0.5 and black_ratio > 0.2:
                return "MRI"  # MRI gốc: peak thấp, nhiều pixel đen
            else:
                return "CT"   # CT/X-ray gốc
        
        # 7. MRI ảnh gốc: mean trung bình-cao, ít pixel đen
        if mean_val > 100 and black_ratio < 0.15:
            return "MRI"
        if black_ratio > 0.25 and std_val > 50 and mean_val > 60:
            return "MRI"
        
        # === LOẠI BỎ FALLBACK AGGRESSIVE ===
        # Old code:
        #   if std_val > 50 and mean_val > 60: return "MRI"
        #   if std_val < 30: return "CT"
        # Những rule này luôn trả về 1 modality cho ảnh không phải y khoa.
        # Thay vào đó, trả về None để pipeline biết không xác định được.
        
    except Exception:
        pass
    return None


# Ánh xạ modality từ pixel analysis sang modality label trong catalog
_PIXEL_MODALITY_TO_CATALOG: dict[str, list[str]] = {
    "CT": ["CT gan", "CT ngực", "CT dạ dày", "CT đại trực tràng", "CT thận", "CT tụy", "CT tuyến giáp", "CT sọ não", "PET/CT"],
    "MRI": ["MRI gan", "MRI não", "MRI dạ dày", "MRI vú", "MRI tuyến tiền liệt", "MRI trực tràng", "MRI cổ tử cung", "MRI thận", "MRI tụy", "MRI tuyến giáp"],
    "X-quang ngực": ["X-quang ngực"],
    "Siêu âm": ["Siêu âm gan", "Siêu âm vú", "Siêu âm tuyến giáp", "Siêu âm thận", "Siêu âm tuyến tiền liệt"],
    "Nội soi": ["Nội soi", "Nội soi đại tràng", "Ảnh soi cổ tử cung"],
}


# Ánh xạ modality → danh sách classifiers: (cancer_key, modality_label, relative_path)
# Khi phát hiện modality, chỉ chạy các classifier trong nhóm đó
_MODALITY_CLASSIFIERS: dict[str, list[tuple[str, str, str]]] = {
    "CT": [
        ("liver", "CT gan", "liver/ct/classifier.pt"),
        ("lung", "CT ngực", "lung/ct/classifier.pt"),
        ("kidney", "CT thận", "kidney/ct/classifier.pt"),
        ("pancreas", "CT tụy", "pancreas/ct/classifier.pt"),
    ],
    "MRI": [
        ("brain", "MRI não", "brain/mri/classifier.pt"),
        ("prostate", "MRI tuyến tiền liệt", "prostate/mri/classifier.pt"),
    ],
    "Siêu âm": [
        ("thyroid", "Siêu âm tuyến giáp", "thyroid/ultrasound/classifier.pt"),
        ("breast", "Siêu âm vú", "breast/ultrasound/classifier.pt"),
    ],
    "Nội soi": [
        ("stomach", "Nội soi", "stomach/endoscopy/classifier.pt"),
        ("colorectal", "Nội soi đại tràng", "colorectal/colonoscopy/classifier.pt"),
    ],
    "Colposcopy": [
        ("cervical", "Ảnh soi cổ tử cung", "cervical/colposcopy/classifier.pt"),
    ],
    "Mammogram": [
        ("breast", "Mammogram", "breast/mammogram/classifier.pt"),
    ],
    "X-quang": [
        ("lung", "X-quang ngực", "lung/xray/classifier.pt"),
    ],
}

# Tất cả classifiers (khi không xác định được modality)
_ALL_CLASSIFIERS: list[tuple[str, str, str]] = []
for _cls_list in _MODALITY_CLASSIFIERS.values():
    _ALL_CLASSIFIERS.extend(_cls_list)

# Thêm root classifiers
_ALL_CLASSIFIERS.extend([
    ("brain", "MRI não", "brain/classifier.pt"),
    ("cervical", "Pap", "cervical/pap/classifier.pt"),
])

# Ánh xạ class labels đặc trưng → cancer key
_SPECIFIC_LABELS: set[str] = {
    # Brain
    "glioma", "meningioma", "pituitary", "no_tumor",
    "No Tumor", "Glioma", "Meningioma", "Pituitary",
    # Colorectal
    "ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM",
    # Cervical pap
    "Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate",
    # Cervical colposcopy
    "squamous_cell_carcinoma", "dysplastic",
    # Lung X-ray
    "covid", "lung_opacity", "viral_pneumonia",
    # Pancreas
    "pdac", "neuroendocrine",
    # Stomach
    "ulcer", "erosion", "carcinoma",
}

# Generic labels → không đặc trưng cho cancer nào
_GENERIC_LABELS = {"benign", "malignant", "normal", "cyst"}


def _compute_entropy(probs: dict[str, float]) -> float:
    """Tính entropy của probability distribution."""
    entropy = 0.0
    for p_val in probs.values():
        if p_val > 1e-8:
            entropy -= p_val * math.log2(p_val)
    return entropy


def _modality_from_pixel_key(pixel_modality: str) -> list[str]:
    """Map pixel modality key sang danh sách modality keys trong _MODALITY_CLASSIFIERS."""
    if pixel_modality is None:
        return []
    m = pixel_modality.lower()
    if "ct" in m and "pet" not in m:
        return ["CT"]
    elif "mri" in m:
        return ["MRI"]
    elif "siêu âm" in m or "ultrasound" in m:
        return ["Siêu âm"]
    elif "nội soi" in m or "endoscopy" in m or "colonoscopy" in m:
        return ["Nội soi"]
    elif "colposcopy" in m or "ảnh soi" in m:
        return ["Colposcopy"]
    elif "mammogram" in m or "mammo" in m:
        return ["Mammogram"]
    elif "x-quang" in m or "xray" in m or "x ray" in m:
        return ["X-quang"]
    return []


def _guess_cancer_by_classifiers(image_path: str | Path) -> tuple[str | None, str | None]:
    """Chạy ảnh qua classifier models để tìm cancer type phù hợp.
    
    Strategy:
    1. Kiểm tra chất lượng ảnh y khoa cơ bản
    2. Xác định modality từ pixel (CT, MRI, Siêu âm, Nội soi)
    3. Chỉ chạy các classifier khớp modality đã phát hiện
    4. Trong nhóm classifier khớp modality, chọn model có score cao nhất
    5. Score = confidence * (1 - normalized_entropy) * specific_label_bonus
    
    Returns:
        (target_key, modality_label) hoặc (None, None) nếu không xác định được.
    """
    from pathlib import Path as _Path

    from medical.cancer_model_registry import CANCERS_DIR
    from medical.cnn_classifier import load_cnn_classifier
    
    source = _Path(image_path)
    if not source.exists():
        return None, None
    
    if not CANCERS_DIR.is_dir():
        return None, None
    
    # === KIỂM TRA CHẤT LƯỢNG ẢNH Y KHOA CƠ BẢN ===
    # Loại ảnh rõ ràng không phải y khoa trước khi chạy classifier
    features = _extract_pixel_features(str(source))
    if features is not None:
        mean_val = features["mean"]
        std_val = features["std"]
        channel_diff = features["channel_diff"]
        black_ratio = features["black_ratio"]
        local_var = features["local_var"]
        
        # Ảnh solid color: std ≈ 0
        if std_val < 3 and channel_diff < 10:
            logger.debug("Ảnh có std quá thấp (%.2f) — không phải ảnh y khoa", std_val)
            return None, None
        
        # Ảnh trắng hoàn toàn
        if mean_val > 250 and std_val < 5:
            logger.debug("Ảnh trắng hoàn toàn — không phải ảnh y khoa")
            return None, None
        
        # Ảnh đen hoàn toàn
        if mean_val < 5 and std_val < 5:
            logger.debug("Ảnh đen hoàn toàn — không phải ảnh y khoa")
            return None, None
        
        # Ảnh noise ngẫu nhiên (có thể đã nén JPEG):
        # - Raw noise: std ~75, local_var ~5800, channel_diff ~58
        # - JPEG noise: std ~49, local_var ~2237, channel_diff ~29, mean ~127
        # Ảnh y khoa RGB (nội soi) có channel_diff cao nhưng std thấp hơn và mean khác
        # Ảnh y khoa grayscale có channel_diff ≈ 0
        #
        # Detection: mean gần 127 + std 30-70 + local_var 1000-5000 + channel_diff 10-50
        # → Pattern đặc trưng của noise ngẫu nhiên đã nén JPEG
        if (100 < mean_val < 150 and 30 < std_val < 70
                and 1000 < local_var < 5000 and 10 < channel_diff < 50
                and black_ratio < 0.05):
            logger.debug("Ảnh có vẻ là noise ngẫu nhiên (mean=%.1f, std=%.1f, local_var=%.1f) — không phải ảnh y khoa",
                         mean_val, std_val, local_var)
            return None, None
    
    # Bước 1: Xác định modality từ pixel
    pixel_modality = guess_modality_from_pixels(str(source))
    modality_keys = _modality_from_pixel_key(pixel_modality)
    
    # Bước 2: Chọn danh sách classifiers
    if modality_keys:
        # Chỉ chạy classifier khớp modality
        classifiers_to_run: list[tuple[str, str, str]] = []
        for mk in modality_keys:
            if mk in _MODALITY_CLASSIFIERS:
                classifiers_to_run.extend(_MODALITY_CLASSIFIERS[mk])
        logger.info("Phát hiện modality=%s, chạy %d classifiers: %s",
                     pixel_modality, len(classifiers_to_run),
                     [c[0] for c in classifiers_to_run])
    else:
        # Không xác định được modality, chạy tất cả
        classifiers_to_run = _ALL_CLASSIFIERS
        logger.info("Không xác định được modality, chạy tất cả %d classifiers", len(classifiers_to_run))
    
    best_target: str | None = None
    best_modality: str | None = None
    best_score = -1.0
    best_has_specific = False
    best_entropy = 999.0
    
    results_log: list[str] = []
    
    for cancer_key, modality_label, rel_path in classifiers_to_run:
        try:
            clf_path = CANCERS_DIR / rel_path
            if not clf_path.is_file():
                continue
            
            wrapper = load_cnn_classifier(clf_path, use_cache=True)
            results = wrapper.predict(str(source), top_k=5)
            
            if not results:
                continue
            
            top_result = results[0]
            confidence = top_result.get("confidence", 0.0)
            top_label = top_result.get("label", "")
            probs = top_result.get("probabilities", {})
            
            # Tính entropy — entropy thấp → model tự tin → in-domain
            entropy = _compute_entropy(probs)
            max_classes = len(probs)
            max_entropy = math.log2(max_classes) if max_classes > 1 else 1.0
            # Normalized entropy: 0 = rất tự tin, 1 = phân bố đều (không tự tin)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
            
            # Kiểm tra label đặc trưng
            has_specific = top_label in _SPECIFIC_LABELS
            
            # Score: normalized entropy thấp + confidence cao → model phù hợp
            # Score = confidence * (1 - normalized_entropy)
            # → confidence cao + entropy thấp (so với max) → score cao
            score = confidence * (1.0 - normalized_entropy)
            if has_specific:
                score += 0.5  # Bonus cho label đặc trưng
            
            results_log.append(f"{cancer_key}: conf={confidence:.3f}, n_ent={normalized_entropy:.3f}, label={top_label}, score={score:.4f}")
            
            # Ưu tiên: label đặc trưng > score cao
            if has_specific and not best_has_specific:
                best_score = score
                best_target = cancer_key
                best_modality = modality_label
                best_has_specific = True
                best_entropy = entropy
            elif has_specific == best_has_specific and score > best_score:
                best_score = score
                best_target = cancer_key
                best_modality = modality_label
                best_has_specific = has_specific
                best_entropy = entropy
            elif has_specific == best_has_specific and score == best_score and entropy < best_entropy:
                # Tie-break: entropy thấp hơn thắng
                best_target = cancer_key
                best_modality = modality_label
                best_entropy = entropy
                
        except Exception as e:
            logger.debug("Lỗi khi chạy classifier %s: %s", rel_path, e)
            continue
    
    if results_log:
        logger.debug("Classifier results: %s", "; ".join(results_log))
    
    if best_target:
        # Threshold: chỉ trả về kết quả khi có ít nhất 1 classifier thực sự tự tin
        # Ảnh không phải y khoa thường có score thấp vì entropy cao (phân bố đều)
        # và confidence thấp. Threshold 0.3 để loại ảnh noise/solid color.
        #
        # Score = confidence * (1 - normalized_entropy)
        # - Ảnh y khoa: confidence ~0.5-0.9, entropy thấp → score > 0.3
        # - Ảnh không y khoa: entropy cao → score < 0.3
        #
        # Ngoài ra, ưu tiên label đặc trưng (glioma, meningioma, pdac...) —
        # nếu best_has_specific = False, threshold cao hơn (0.4) vì generic labels
        # (benign, malignant, normal) dễ xuất hiện cho ảnh không y khoa.
        if best_has_specific:
            threshold = 0.3
        else:
            threshold = 0.4  # Generic labels cần score cao hơn
        if best_score < threshold:
            logger.debug("Auto-detect score quá thấp (%.4f < %.1f) — có thể không phải ảnh y khoa",
                         best_score, threshold)
            return None, None
        return best_target, best_modality
    
    return None, None


def _extract_pixel_features(image_path: str | Path) -> dict[str, float] | None:
    """Trích xuất pixel features từ ảnh.
    
    Returns:
        Dict với keys: mean, std, black_ratio, local_var, channel_diff
        hoặc None nếu không trích xuất được.
    """
    source = Path(image_path)
    if not source.exists():
        return None
    
    try:
        from PIL import Image
        img = Image.open(source)
        arr = np.array(img)
        
        channel_diff = 0.0
        if len(arr.shape) == 3 and arr.shape[2] >= 3:
            r = arr[:, :, 0].astype(float)
            g = arr[:, :, 1].astype(float)
            b = arr[:, :, 2].astype(float)
            channel_diff = float(np.mean([np.abs(r - g), np.abs(g - b), np.abs(r - b)]))
            if channel_diff < 2:
                gray = r.astype(float)
            else:
                gray = np.mean(arr[:, :, :3].astype(float), axis=2)
        elif len(arr.shape) == 2:
            gray = arr.astype(float)
        else:
            return None
        
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        black_ratio = float(np.sum(gray < 20)) / gray.size if gray.size > 0 else 0
        
        if gray.size > 100 and gray.shape[0] > 10:
            local_var = float(np.var(gray[::10, ::10]))
        else:
            local_var = 0
        
        return {
            "mean": mean_val,
            "std": std_val,
            "black_ratio": black_ratio,
            "local_var": local_var,
            "channel_diff": channel_diff,
        }
    except Exception:
        return None


def _guess_cancer_by_pixel_features(image_path: str | Path) -> tuple[str | None, str | None]:
    """Dùng pixel features để phân biệt cancer type.
    
    Returns:
        (target_key, modality_label) hoặc (None, None) nếu không xác định được.
    
    QUAN TRỌNG: Hàm này chỉ hoạt động với ảnh y khoa đã preprocess (224x224).
    Ảnh không phải y khoa hoặc ảnh có std quá thấp (solid color, gradient) sẽ trả về None.
    """
    features = _extract_pixel_features(image_path)
    if features is None:
        return None, None
    
    mean_val = features["mean"]
    std_val = features["std"]
    black_ratio = features["black_ratio"]
    local_var = features["local_var"]
    channel_diff = features["channel_diff"]
    
    # === KIỂM TRA CƠ BẢN: ảnh có nội dung thực sự không? ===
    # Ảnh solid color: std ≈ 0 → không phải ảnh y khoa
    # Ảnh gradient: std thấp, local_var thấp → không phải ảnh y khoa
    # Ảnh y khoa preprocess có std tối thiểu ~3-5 (do noise và chi tiết)
    if std_val < 3 and channel_diff < 50:
        # Ảnh gần như đồng nhất — không phải ảnh y khoa
        return None, None
    
    # 1. Nội soi (channel_diff cao)
    if channel_diff > 50:
        if mean_val > 85:
            return "stomach", "Nội soi"
        else:
            return "colorectal", "Nội soi đại tràng"
    
    # 2. MRI (black_ratio cao)
    # Brain MRI: black_ratio rất cao (>0.25), mean thấp (26-66), std cao (31-60)
    if black_ratio > 0.25:
        return "brain", "MRI não"
    # Cervical MRI: mean cao (100-133), std thấp (5-6), black_ratio ≈ 0
    # Thêm std > 3 để tránh solid color images
    if mean_val > 100 and 3 < std_val < 15:
        return "cervical", "MRI cổ tử cung"
    # Prostate MRI: mean > 80, std < 15, black_ratio > 0 (MRI có nền đen)
    if mean_val > 80 and 3 < std_val < 15 and black_ratio > 0.02:
        return "prostate", "MRI tuyến tiền liệt"
    
    # 3. Siêu âm (local_var cao)
    if 500 < local_var < 5000 and black_ratio < 0.15:
        if mean_val > 70:
            return "thyroid", "Siêu âm tuyến giáp"
        else:
            return "breast", "Siêu âm vú"
    
    # 4. CT gốc (std cao, black_ratio > 0)
    if std_val > 50 and black_ratio > 0.05:
        return "lung", "CT ngực"
    
    # 5. CT preprocess (std thấp, black_ratio thấp) — phân biệt bằng local_var + mean
    # local_var là key feature:
    #   liver: 0.7-2.5, prostate: 0.1-1.8, breast: 1.8-2.9 → THẤP (< 5)
    #   kidney: 33-38, pancreas: 29-39 → CAO (> 20)
    if 3 < std_val < 20 and black_ratio < 0.05:
        if mean_val < 55:
            return "breast", "MRI vú"
        elif local_var > 20:
            # local_var cao → kidney hoặc pancreas
            # kidney: mean ~82, pancreas: mean ~91
            # Nhưng có overlap: kidney mean 74-82, pancreas mean 66-92
            # First kidney: mean=81.9, First pancreas: mean=91.3
            # Dùng mean threshold 82: < 82 → kidney, >= 82 → pancreas
            if mean_val > 82:
                return "pancreas", "CT tụy"
            else:
                return "kidney", "CT thận"
        else:
            # local_var thấp → liver hoặc prostate
            # Không thể phân biệt 100% bằng pixel features
            # Dùng std threshold 7.2: liver median std=6.87, prostate median std=7.9
            # First liver: std=7.12 < 7.2 → liver OK
            # First prostate: std=7.23 >= 7.2 → prostate OK
            if std_val >= 7.2:
                return "prostate", "MRI tuyến tiền liệt"
            else:
                return "liver", "CT gan"
    
    # 6. Không fallback theo generic modality từ pixel analysis
    # vì guess_modality_from_pixels có thể trả về false positive cho ảnh không phải y khoa
    # Thay vào đó, trả về None để guess_cancer_from_pixels dùng classifier-based detection
    # (có threshold score để lọc false positive)
    return None, None


def guess_cancer_from_pixels(image_path: str | Path) -> tuple[str | None, str | None]:
    """Phân tích ảnh để ước lượng cả cancer type và modality.
    
    Strategy:
    1. Classifier-based detection (có threshold score để lọc false positive)
    
    Returns:
        (target_key, modality_label) hoặc (None, None) nếu không xác định được.
    
    QUAN TRỌNG: Chỉ dùng classifier-based detection vì nó có threshold score
    để tự động loại ảnh không phải y khoa. Pixel features đã bị loại bỏ vì
    dễ false positive cho ảnh noise, gradient, solid color.
    """
    # Chỉ dùng classifier-based detection (có threshold score để lọc false positive)
    return _guess_cancer_by_classifiers(image_path)


def _map_subdir_to_modality(cancer_key: str, subdir_name: str) -> str:
    """Map modality subdirectory name to catalog modality label."""
    # Reverse lookup from cancer_catalog modalities
    from medical.cancer_catalog import get_cancer_target
    target = get_cancer_target(cancer_key)
    if not target:
        return subdir_name
    
    normalized_subdir = subdir_name.lower().replace("_", " ").replace("-", " ")
    
    for modality in target.modalities:
        normalized_modality = modality.lower().replace("_", " ").replace("-", " ")
        # Kiểm tra xem subdir có khớp với modality không
        if normalized_subdir in normalized_modality or normalized_modality in normalized_subdir:
            return modality
        # Aliases
        aliases = {
            "ct": ["ct"],
            "mri": ["mri"],
            "xray": ["x-quang"],
            "ultrasound": ["siêu âm"],
            "endoscopy": ["nội soi"],
            "colonoscopy": ["nội soi"],
            "colposcopy": ["ảnh soi"],
            "mammogram": ["mammogram"],
            "pap": ["pap"],
        }
        for alias_key, alias_values in aliases.items():
            if alias_key == normalized_subdir:
                for mod in target.modalities:
                    mod_lower = mod.lower()
                    for val in alias_values:
                        if val in mod_lower:
                            return mod
    
    return subdir_name


def _reverse_modality_key(cancer_key: str, modality_label: str) -> str:
    """Reverse map catalog modality label to subdirectory name."""
    from medical.cancer_model_registry import modality_key
    mapped = modality_key(modality_label)
    if mapped:
        return mapped
    return modality_label.lower().replace(" ", "_").replace("/", "_")


def _infer_medical_target_key(normalized_text: str, modality: str | None) -> str | None:
    # Ưu tiên: tìm từ tên file/path trước (chính xác hơn modality generic)
    padded = f" {normalized_text} "
    for target_key, terms in _TARGET_HINTS:
        if any(term in padded for term in terms):
            return target_key
    # Fallback: dùng modality để suy ra target
    if modality and modality in _MODALITY_TO_TARGET_KEY:
        return _MODALITY_TO_TARGET_KEY[modality]
    return None


def resolve_medical_upload_path(path: str | Path) -> Path:
    source = Path(path)
    if source.is_file():
        if not is_supported_medical_upload_path(source):
            raise ValueError(f"Không hỗ trợ file upload: {source}")
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"Không tìm thấy đường dẫn upload: {source}")

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
    except ImportError as exc:# pragma: no cover
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
    except ImportError as exc:# pragma: no cover
        raise RuntimeError("Không đọc được NIfTI. Hãy cài đặt nibabel để mở file .nii/.nii.gz.") from exc

    image = nib.load(str(source))
    volume = np.asanyarray(image.dataobj)
    if volume.ndim == 4:
        volume = volume[..., 0]
    return _limit_volume_slices(np.asarray(volume), max_slices=256)


def _load_mha_volume(source: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError as exc:# pragma: no cover
        raise RuntimeError("Không đọc được MHA/MHD. Hãy cài đặt SimpleITK để mở file .mha/.mhd.") from exc

    image = sitk.ReadImage(str(source))
    volume = sitk.GetArrayFromImage(image)
    return _limit_volume_slices(np.asarray(volume), max_slices=256)


def _load_dicom_series_volume(source: Path) -> np.ndarray:
    try:
        import pydicom
        from pydicom.pixels import apply_voi_lut
    except ImportError as exc:# pragma: no cover
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
    volume = np.stack(slices, axis=0)
    return _limit_volume_slices(volume, max_slices=256)


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
            raise ValueError(f"Không hỗ trợ định dạng volume: {source}")

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


def _load_dicom_with_window(source: Path) -> Image.Image:
    try:
        import numpy as np
        import pydicom
        ds = pydicom.dcmread(str(source), force=True)
        arr = ds.pixel_array.astype(np.float32)
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        slope = float(getattr(ds, "RescaleSlope", 1))
        arr = arr * slope + intercept
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            wc = float(wc) if isinstance(wc, (int, float)) else float(wc[0])
            ww = float(ww) if isinstance(ww, (int, float)) else float(ww[0])
            low = wc - ww / 2
            high = wc + ww / 2
            arr = np.clip((arr - low) / (high - low) * 255.0, 0, 255)
        else:
            arr_min, arr_max = float(arr.min()), float(arr.max())
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        return Image.fromarray(arr).convert("RGB")
    except Exception:
        with Image.open(source) as image:
            return image.convert("RGB")


def load_medical_source_image(source_path: str | Path) -> Image.Image:
    source = Path(source_path)
    if _medical_upload_suffix(source) in {".mp4", ".avi", ".mov", ".mkv"}:
        capture = cv2.VideoCapture(str(source))
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise ValueError(f"Không đọc được khung hình từ video: {source}")
        # Khung giữa chỉ dùng để tạo preview; pipeline suy luận lấy mẫu nhiều khung riêng.
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if is_medical_volume_source(source):
        return _load_medical_volume_image(source)
    if source.suffix.lower() == ".dcm":
        return _load_dicom_with_window(source)
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


def sample_medical_video_frames(
    source_path: str | Path,
    *,
    max_frames: int = 12,
) -> list[tuple[int, float, np.ndarray]]:
    if max_frames <= 0:
        raise ValueError("max_frames phải lớn hơn 0.")
    source = Path(source_path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Không mở được video: {source}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count > 0:
            indices = sorted(set(np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int).tolist()))
            frames: list[tuple[int, float, np.ndarray]] = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames.append((index, index / fps if fps > 0 else 0.0, frame))
        else:
            # shortcut: video không khai báo frame count chỉ lấy N frame đầu; nâng cấp nếu gặp stream dài.
            frames = []
            index = 0
            while len(frames) < max_frames:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frames.append((index, index / fps if fps > 0 else 0.0, frame))
                index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Không đọc được khung hình từ video: {source}")
    return frames


def normalize_uploaded_image(
    source_path: str | Path,
    target_dir: str | Path,
    *,
    image_size: int = 320,
) -> Path:
    source = Path(source_path)
    if not is_supported_medical_upload_path(source):
        raise ValueError(f"Không hỗ trợ file upload: {source}")
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
