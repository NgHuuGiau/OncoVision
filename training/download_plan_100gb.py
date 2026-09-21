"""Manifest + downloader gioi han 100GB cho 11 nhom (tiep tuc thu muc dataset co san).

Nguon chinh thong (da kiem chung):
- MSD (Liver/Lung/Pancreas/Colon/Prostate): AWS Open Data registry.opendata.aws/msd,
  mirror MONAI https://msd-for-monai.s3-us-west-2.amazonaws.com/Task{03,05,06,07,10}_*.tar
- Phoi: DICOM-LIDC-IDRI-Nodules 2.5GB (doi.org/10.7937/tcia.2018.h7umfurq) thay vi full 133GB
- Than: KiTS23 https://kits-challenge.org/kits23 (489 train, CC BY-NC-SA 4.0) - lay 100 case dau
- Tien liet tuyen: PI-CAI Zenodo 6624726, 5 folds ~27GB (labels da co)
- Giap: TN3K full (~1GB) + SIPaKMeD + BreastDCEDL demo + BraTS subset

Quota: tong cong <= 100GB (tinh ca 7GB dataset + 6.5GB models hien co).
# ponytail: 1 file, skip-neu-co, resume, khong fake pathology labels
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

QUOTA_GB = 100.0

PLAN = [
    {"key": "gan", "folder": "dataset/Ung thư gan/raw/MSD_Liver",
     "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
     "size_gb": 2.5, "note": "MSD Task03 train 131 vols. Khong co pathology/cancer-non-cancer cap benh nhan."},
    {"key": "phoi-msd", "folder": "dataset/Ung thư phổi/raw/MSD_Lung",
     "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
     "size_gb": 1.5, "note": "MSD Task06 train 64 vols. Nhan cancer/non-cancer can doi chieu LIDC."},
    {"key": "phoi-lidc-sub", "folder": "dataset/Ung thư phổi/raw/LIDC-IDRI",
     "url": "https://doi.org/10.7937/tcia.2018.h7umfurq",
     "size_gb": 2.5, "note": "DICOM-LIDC-Nodules SEG/SR 875 ca. Full LIDC 133GB VUOT quota - khong tai. Can TCIA Data Retriever, tai thu cong."},
    {"key": "tuy", "folder": "dataset/Ung thư tụy/raw/MSD_Pancreas",
     "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
     "size_gb": 8.2, "note": "MSD Task07 train 282 vols (HF mirror 8.16GB)."},
    {"key": "dai-truc-trang", "folder": "dataset/Ung thư đại trực tràng/raw/MSD_Colon",
     "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar",
     "size_gb": 2.0, "note": "MSD Task10 train 126 vols. Khong du nhan cancer/non-cancer."},
    {"key": "tuyen-tien-liet-msd", "folder": "dataset/Ung thư tuyến tiền liệt/raw/MSD_Prostate",
     "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar",
     "size_gb": 1.0, "note": "Mask giai phau, KHONG dung lam ground-truth cancer."},
    {"key": "tuyen-tien-liet-picai", "folder": "dataset/Ung thư tuyến tiền liệt/raw/PI-CAI/images",
     "url": "https://zenodo.org/records/6624726",
     "size_gb": 27.0, "note": "5 folds zip (~5.4GB/fold). Labels da co. Tai thu cong theo fold + verify md5. Co histopathology/follow-up - NGUON XAC NHAN tot nhat."},
    {"key": "than-kits23-100", "folder": "dataset/Ung thư thận/raw/KiTS23",
     "url": "https://kits-challenge.org/kits23",
     "size_gb": 15.0, "note": "100/489 case dau (~15GB). Full 489 vuot quota khi cong PI-CAI. Theo README repo kits23 de fetch imaging."},
    {"key": "vu-breastdcedl", "folder": "dataset/Ung thư vú/raw/BreastDCEDL",
     "url": "https://www.cancerimagingarchive.net/collection/breast-dcedl",
     "size_gb": 3.0, "note": "Demo + metadata HR/HER2/pCR. Chua du ca lanh/binh thuong/kho."},
    {"key": "co-tu-cung", "folder": "dataset/Ung thư cổ tử cung/raw/SIPaKMeD",
     "url": "https://www.kaggle.com/datasets/prahladmeena/sipakmed",
     "size_gb": 2.0, "note": "Te bao hoc BMP. Khong thay the chan doan pathology cap benh nhan."},
    {"key": "tuyen-giap-tn3k-full", "folder": "dataset/Ung thư tuyến giáp/raw/TN3K",
     "url": "https://github.com/haParse/TN3K",
     "size_gb": 1.0, "note": "Giai nen archive day du (hien chi 30 anh test). Co nhan benign/malignant."},
    {"key": "nao-brats-sub", "folder": "dataset/Ung thư não/raw/BraTS-subset",
     "url": "http://medicaldecathlon.com/",
     "size_gb": 5.0, "note": "MSD Task01 Brain Tumour subset thay vi full 750 vols. Brain-Mets hien co giu nguyen."},
    {"key": "da-day-gcss", "folder": "dataset/Ung thư dạ dày/raw/GCSS",
     "url": "TCGA-STAD via GDC",
     "size_gb": 4.0, "note": "Archive da verify MD5. Can doi chieu TCGA clinical cho pathology."},
]

MANUAL = {"phoi-lidc-sub", "tuyen-tien-liet-picai", "than-kits23-100", "co-tu-cung", "vu-breastdcedl", "da-day-gcss"}


def _dir_size_gb(path: Path) -> float:
    total = 0
    if path.is_dir():
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    return total / (1024 ** 3)


def main(download: bool = False) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    used = _dir_size_gb(Path("dataset")) + _dir_size_gb(Path("models"))
    planned = sum(p["size_gb"] for p in PLAN)
    print(f"used now: {used:.1f}GB | plan adds: ~{planned:.1f}GB | quota: {QUOTA_GB}GB")
    if used + planned > QUOTA_GB:
        print("OVER QUOTA - cat giam: bo KiTS23 full, chi lay 100 case; LIDC chi lay DICOM-Nodules.")
        return 2
    for p in PLAN:
        dest = Path(p["folder"])
        exists = dest.exists() and any(dest.iterdir()) if dest.exists() else False
        tag = "RESUME" if exists else "NEW"
        man = " [MANUAL]" if p["key"] in MANUAL else ""
        print(f"[{tag}]{man} {p['key']}: ~{p['size_gb']}GB -> {p['folder']}")
        print(f"    src: {p['url']}")
        print(f"    note: {p['note']}")
        if download and p["key"] not in MANUAL and p["url"].startswith("http"):
            dest.mkdir(parents=True, exist_ok=True)
            fname = dest / Path(p["url"]).name
            if fname.exists():
                print(f"    skip (da co {fname.name})")
                continue
            print(f"    downloading {fname.name} ...")
            urllib.request.urlretrieve(p["url"], fname)
    print("OK - chay voi --download de tai cac muc tu-dong (MSD tars). Muc MANUAL tai thu cong theo URL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(download="--download" in sys.argv[1:]))
