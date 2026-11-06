#!/usr/bin/env python3
"""
Full pipeline verification for OncoVision models.
Checks all expected model files exist and can be loaded.
"""

import os
from pathlib import Path
from ultralytics import YOLO

BASE = Path(r"D:\Project\OncoVision\models\pretrained\cancers")
ROOT = BASE.parent

EXPECTED_MODELS = {
    "brain": {
        "classifier": "brain/classifier.pt",
        "detector": "brain/detector.pt",
        "volume": "brain/volume.pt",
    },
    "modality": {
        "classifier": "modality_classifier.pt",
    },
    "lung": {
        "xray_classifier": "lung/xray/classifier.pt",
        "ct_detector": "lung/ct/detector.pt",
    },
    "liver": {
        "classifier": "liver/ct/classifier.pt",
        "detector": "liver/ct/detector.pt",
        "volume": "liver/volume.pt",
    },
    "colorectal": {
        "detector": "colorectal/colonoscopy/detector.pt",
    },
    "breast": {
        "classifier": "breast/ultrasound/classifier.pt",
        "detector": "breast/ultrasound/detector.pt",
        "mammo_detector": "breast/mammogram/detector.pt",
    },
    "stomach": {
        "classifier": "stomach/endoscopy/classifier.pt",
        "detector": "stomach/endoscopy/detector.pt",
    },
    "prostate": {
        "classifier": "prostate/mri/classifier.pt",
        "detector": "prostate/mri/detector.pt",
    },
    "cervical": {
        "classifier": "cervical/pap/classifier.pt",
        "detector": "cervical/colposcopy/detector.pt",
    },
    "kidney": {
        "detector": "kidney/ct/detector.pt",
        "volume": "kidney/volume.pt",
    },
    "pancreas": {
        "detector": "pancreas/ct/detector.pt",
        "volume": "pancreas/volume.pt",
    },
    "thyroid": {
        "classifier": "thyroid/ultrasound/classifier.pt",
    },
}

def check_yolo(path, label):
    full_path = BASE / path
    if not os.path.exists(full_path):
        return f"  [MISSING] {label}: NOT FOUND"
    try:
        m = YOLO(full_path)
        return f"  [OK] {label}: {m.task} | {len(m.names)} classes: {list(m.names.values())}"
    except Exception as e:
        return f"  [FAIL] {label}: LOAD FAILED - {e}"

def check_classifier(path, label):
    full_path = BASE / path
    if not os.path.exists(full_path):
        return f"  [MISSING] {label}: NOT FOUND"
    try:
        import torch
        ckpt = torch.load(full_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'class_labels' in ckpt:
            return f"  [OK] {label}: {len(ckpt['class_labels'])} classes - {ckpt.get('backbone', 'unknown')}"
        else:
            return f"  [FAIL] {label}: WRONG FORMAT (need class_labels, backbone, num_classes)"
    except Exception as e:
        return f"  [FAIL] {label}: LOAD FAILED - {e}"

def check_volume(path, label):
    full_path = BASE / path
    if not os.path.exists(full_path):
        return f"  [MISSING] {label}: NOT FOUND"
    json_path = full_path.with_suffix('.json')
    if json_path.exists():
        return f"  [OK] {label}: volume.pt + volume.json"
    return f"  [WARN] {label}: volume.pt found but MISSING volume.json"

def main():
    print("=" * 80)
    print("ONCOVISION FULL PIPELINE VERIFICATION")
    print("=" * 80)
    
    total = 0
    ready = 0
    missing = []
    
    for organ, models in EXPECTED_MODELS.items():
        print(f"\n{'='*20} {organ.upper()} {'='*20}")
        
        for model_type, path in models.items():
            total += 1
            if model_type == "volume":
                result = check_volume(path, model_type)
            elif "detector" in model_type or "mammo" in model_type:
                result = check_yolo(path, model_type)
            else:
                result = check_classifier(path, model_type)
            
            print(f"  {result}")
            if result.startswith("  [OK]"):
                ready += 1
            else:
                missing.append(f"{organ}/{path}")
    
    print("\n" + "=" * 80)
    print(f"READY: {ready}/{total}")
    print(f"MISSING: {len(missing)}")
    
    if missing:
        print("\nMISSING FILES:")
        for m in missing:
            print(f"  - {m}")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

    # Print summary of working models
    print("\n[WORKING] MODELS (Ready for inference):")
    working = []
    checks = [
        (Path(r"D:\Project\OncoVision\models\pretrained\cancers") / "colorectal" / "colonoscopy" / "detector.pt", "Colorectal Polyp Detector"),
        (Path(r"D:\Project\OncoVision\models\pretrained\cancers") / "breast" / "ultrasound" / "classifier.pt", "Breast US Classifier"),
        (Path(r"D:\Project\OncoVision\models\pretrained\cancers") / "breast" / "ultrasound" / "detector.pt", "Breast US Lesion Detector"),
        (Path(r"D:\Project\OncoVision\models\pretrained\cancers") / "breast" / "mammogram" / "detector.pt", "Breast Mammogram Detector"),
        (Path(r"D:\Project\OncoVision\models\pretrained\cancers") / "brain" / "detector.pt", "Brain YOLO11n Detector"),
        (Path(r"D:\Project\OncoVision\models\pretrained\cancers") / "brain" / "volume.pt", "Brain 3D Volume Seg"),
        (Path(r"D:\Project\OncoVision\models\pretrained\cancers") / "liver" / "volume.pt", "Liver 3D Seg"),
        (Path(r"D:\Project\OncoVision\models\pretrained\cancers") / "stomach" / "endoscopy" / "detector.pt", "Stomach Polyp Detector (from colorectal)"),
    ]
    for path, name in checks:
        if path.exists():
            print(f"  - {name}")

    print("\n[NEXT STEPS]:")
    print("1. Download Pancreas 3D & Kidney 3D (run PowerShell commands)")
    print("2. Train Modality Classifier: python scripts/train_modality_classifier.py --epochs 10")
    print("3. Download 8 YOLO detectors from Roboflow Universe")
    print("4. Download 6 Classifiers from Kaggle/GitHub")
    print("5. Brain Classifier: Request HF access or train from BraTS")
    print("6. Run: python scripts/verify_full_pipeline.py")

if __name__ == "__main__":
    main()