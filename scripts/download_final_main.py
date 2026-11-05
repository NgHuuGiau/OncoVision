#!/usr/bin/env python3
"""Main script to download all models"""

import sys
import os
from pathlib import Path

sys.path.insert(0, r"D:\Project\OncoVision\scripts")

from download_final_part1 import BASE, ROOT, ensure_dir
from download_final_part2 import download_yolo_roboflow
from download_final_part3 import download_yolo_models, print_classifiers_manual, download_3d_models, download_modality

def main():
    BASE = Path(r"D:\Project\OncoVision\models\pretrained\cancers")
    ROOT = BASE.parent
    
    print("=" * 70)
    print("DOWNLOADING ALL MISSING MODELS")
    print("=" * 70)
    
    # 1. COLORECTAL DETECTOR
    print("\n[1/18] Colorectal Detector (goktug14/yolov5_kvasir_polyp)")
    if (Path(r"D:\Project\OncoVision\models\pretrained\cancers") / "colorectal" / "colonoscopy" / "detector.pt").exists():
        print("  OK: Already exists")
    else:
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download("goktug14/yolov5_kvasir_polyp", "weights/best.pt", local_dir=str(Path(r"D:\Project\OncoVision\models\pretrained\cancers") / "colorectal" / "colonoscopy"), local_dir_use_symlinks=False)
            os.rename(local, Path(r"D:\Project\OncoVision\models\pretrained\cancers") / "colorectal" / "colonoscopy" / "detector.pt")
            print("  OK: Downloaded")
        except Exception as e:
            print(f"  FAIL: {e}")
            print("  Manual: https://universe.roboflow.com/goktug14/yolov5_kvasir_polyp")
    
    # YOLO Models
    from download_final_part3 import download_yolo_models, print_classifiers_manual, download_3d_models, download_modality
    download_yolo_models()
    
    # Classifiers - Manual
    print("\n--- CLASSIFIERS (Manual download from GitHub Releases / Kaggle) ---")
    classifiers = [
        ("HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch", "brain/classifier.pt"),
        ("DIAGNijmegen/TDSC", "thyroid/ultrasound/classifier.pt"),
        ("PICAI/PICAI-grand-challenge", "prostate/mri/classifier.pt"),
        ("andrewmvd/sipakmed", "cervical/pap/classifier.pt"),
        ("josedolz/LiviaNET", "stomach/endoscopy/classifier.pt"),
        ("yhygao/CBIM-Medical-Image-Segmentation", "liver/ct/classifier.pt"),
    ]
    print("\n--- CLASSIFIERS (Manual download from GitHub Releases / Kaggle Models) ---")
    for repo, target in classifiers:
        print(f"  DOWNLOAD: {repo} -> {target}")
        print(f"  Link: https://github.com/{repo}/releases")
        print(f"  Or Kaggle: search '{repo}' -> Models tab")
    
    # 3D Models
    from download_final_part3 import download_3d_models, download_modality
    download_3d_models()
    
    # Modality Classifier
    download_modality()
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE!")
    print("=" * 70)
    print("\nMANUAL DOWNLOAD STILL NEEDED FOR:")
    print("  - Roboflow models not on HF Hub (search universe.roboflow.com)")
    print("  - Kaggle classifiers (need kaggle.json API token)")
    print("  - GitHub Releases classifiers (manual download from Releases page)")

if __name__ == "__main__":
    main()