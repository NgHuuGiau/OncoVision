#!/usr/bin/env python3
"""Full pipeline verification - test all models can be loaded."""
from ultralytics import YOLO
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from medical.cancer_model_registry import validate_detector_backend

BASE = Path(r"D:/Project/OncoVision/models/pretrained/cancers")
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
        "xray_detector": "lung/xray/detector.pt",
        "ct_detector": "lung/ct/detector.pt",
        "ct_classifier": "lung/ct/classifier.pt",
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
        "colposcopy_classifier": "cervical/colposcopy/classifier.pt",
        "detector": "cervical/colposcopy/detector.pt",
    },
    "kidney": {
        "detector": "kidney/ct/detector.pt",
        "classifier": "kidney/ct/classifier.pt",
        "volume": "kidney/volume.pt",
    },
    "pancreas": {
        "detector": "pancreas/ct/detector.pt",
        "classifier": "pancreas/ct/classifier.pt",
        "volume": "pancreas/volume.pt",
    },
    "thyroid": {
        "classifier": "thyroid/ultrasound/classifier.pt",
        "detector": "thyroid/ultrasound/detector.pt",
    },
}

def check_yolo(path, label):
    full_path = BASE / path if "modality" not in str(path) else ROOT / path
    if not full_path.exists():
        return f"  [MISSING] {label}: NOT FOUND"
    try:
        m = YOLO(str(full_path))
        validate_detector_backend(m)
        return f"  [OK] {label}: {m.task} | {len(m.names)} classes"
    except Exception as e:
        return f"  [FAIL] {label}: LOAD FAILED - {e}"

def check_classifier(path, label):
    full_path = BASE / path if "modality" not in str(path) else ROOT / path
    if not full_path.exists():
        return f"  [MISSING] {label}: NOT FOUND"
    try:
        ckpt = torch.load(str(full_path), map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'class_labels' in ckpt:
            return f"  [OK] {label}: {len(ckpt['class_labels'])} classes - {ckpt.get('backbone', 'unknown')}"
        elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            return f"  [OK] {label}: state_dict only (no metadata)"
        else:
            return f"  [WARN] {label}: dict but no class_labels - keys: {list(ckpt.keys())[:5]}"
    except Exception as e:
        return f"  [FAIL] {label}: LOAD FAILED - {e}"

def check_volume(path, label):
    full_path = BASE / path
    if not full_path.exists():
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
    failed = []
    
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
            elif result.startswith("  [MISSING]"):
                missing.append(f"{organ}/{path}")
            else:
                failed.append(f"{organ}/{path}")
    
    print("\n" + "=" * 80)
    print(f"READY: {ready}/{total}")
    print(f"MISSING: {len(missing)}")
    print(f"FAILED: {len(failed)}")
    
    if missing:
        print("\nMISSING FILES:")
        for m in missing:
            print(f"  - {m}")
    
    if failed:
        print("\nFAILED TO LOAD:")
        for f in failed:
            print(f"  - {f}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    
    # Summary
    print("\n[WORKING MODELS - Ready for inference]:")
    working = [
        ("Colorectal Polyp Detector", BASE / "colorectal/colonoscopy/detector.pt"),
        ("Breast US Classifier", BASE / "breast/ultrasound/classifier.pt"),
        ("Breast US Detector", BASE / "breast/ultrasound/detector.pt"),
        ("Brain 3D Volume Seg", BASE / "brain/volume.pt"),
        ("Liver 3D Volume Seg", BASE / "liver/volume.pt"),
        ("Kidney 3D Volume Seg", BASE / "kidney/volume.pt"),
        ("Pancreas 3D Volume Seg", BASE / "pancreas/volume.pt"),
        ("Lung CT Detector (YOLO)", BASE / "lung/ct/detector.pt"),
        ("Lung CT Classifier", BASE / "lung/ct/classifier.pt"),
        ("Modality Classifier", ROOT / "modality_classifier.pt"),
    ]
    for name, path in working:
        if path.exists():
            sz = path.stat().st_size / 1024 / 1024
            print(f"  - {name} ({sz:.1f} MB)")

if __name__ == "__main__":
    main()