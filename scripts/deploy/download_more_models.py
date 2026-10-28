"""Tai model them tu cac repo moi tim thay."""
from __future__ import annotations

import urllib.request
from pathlib import Path

CANCERS_DIR = Path("models/pretrained/cancers")
CANCERS_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS = [
    # YOLO Medical
    ("lung", "https://huggingface.co/MedicalVision/yolos_tiny_50ep/resolve/main/model.safetensors", "lung_yolo_tiny.safetensors"),
    
    # YOLOv11 Tumor Detection (additional weights)
    ("lung", "https://huggingface.co/sajjadhadi/YOLOv11-Tumor-Detection/resolve/main/weights/best.pt", "lung_yolo11_best.pt"),
    ("lung", "https://huggingface.co/sajjadhadi/YOLOv11-Tumor-Detection/resolve/main/weights/last.pt", "lung_yolo11_last.pt"),
    
    # MONAI Lung Nodule Detection
    ("lung", "https://huggingface.co/MONAI/lung_nodule_ct_detection/resolve/main/models/model.pt", "lung_monai_nodule.pt"),
    ("lung", "https://huggingface.co/monai-test/lung_nodule_ct_detection/resolve/main/models/model.pt", "lung_monai_nodule_test.pt"),
    
    # Breast Tumor Detection
    ("breast", "https://huggingface.co/RsGoksel/Breast-Tumor-Mass-Detection/resolve/main/Mass_Detection.pt", "breast_mass_detection.pt"),
    
    # Liver Tumor Segmentation (additional)
    ("liver", "https://huggingface.co/adityasai369/Liver_tumor_segmentation/resolve/main/tumourmodel3.pth", "liver_tumor_seg2.pth"),
    ("liver", "https://huggingface.co/Husstavo/LiverTumorSegmentation/resolve/main/liver_unet.pth", "liver_unet2.pth"),
]


def main() -> None:
    for key, url, name in DOWNLOADS:
        dest = CANCERS_DIR / key / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"[skip] {dest}")
            continue
        print(f"[dl] {key} <- {url}")
        try:
            urllib.request.urlretrieve(url, str(dest))
            print(f"  OK {dest.stat().st_size:,} bytes")
        except Exception as exc:
            print(f"  ERR {exc}")


if __name__ == "__main__":
    main()