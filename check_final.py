#!/usr/bin/env python3
"""Check all model files - FIXED to match actual directory structure"""
import os
from pathlib import Path

base = Path(r"D:\Project\OncoVision\models\pretrained\cancers")

# Expected files with their ACTUAL locations
expected = {
    # Brain
    'brain/detector.pt': 'brain/mri/detector.pt',
    'brain/volume.pt': 'brain/mri/volume.pt2',  # Note: .pt2 exists, not .pt
    'brain/volume.json': 'brain/mri/volume.json',
    'brain/classifier.pt': 'brain/mri/classifier.pt',
    
    # Liver
    'liver/volume.pt': 'liver/ct/volume.pt',  # MISSING - only volume.pt2 exists
    'liver/volume.json': 'liver/ct/volume.json',
    'liver/ct/classifier.pt': 'liver/ct/classifier.pt',
    'liver/ct/detector.pt': 'liver/ct/detector.pt',
    
    # Breast
    'breast/ultrasound/classifier.pt': 'breast/ultrasound/classifier.pt',
    'breast/ultrasound/detector.pt': 'breast/ultrasound/detector.pt',
    'breast/mammogram/detector.pt': 'breast/mammogram/detector.pt',  # MISSING
    
    # Stomach
    'stomach/endoscopy/classifier.pt': 'stomach/endoscopy/classifier.pt',
    'stomach/endoscopy/detector.pt': 'stomach/endoscopy/detector.pt',
    
    # Colorectal
    'colorectal/colonoscopy/detector.pt': 'colorectal/colonoscopy/detector.pt',
    
    # Lung
    'lung/xray/classifier.pt': 'lung/xray/classifier.pt',  # MISSING
    'lung/ct/detector.pt': 'lung/ct/detector.pt',
    
    # Kidney
    'kidney/ct/detector.pt': 'kidney/ct/detector.pt',
    'kidney/volume.pt': 'kidney/volume.pt',
    'kidney/volume.json': 'kidney/volume.json',
    
    # Pancreas
    'pancreas/ct/detector.pt': 'pancreas/ct/detector.pt',
    'pancreas/volume.pt': 'pancreas/volume.pt',
    'pancreas/volume.json': 'pancreas/volume.json',
    
    # Prostate
    'prostate/mri/classifier.pt': 'prostate/mri/classifier.pt',
    'prostate/mri/detector.pt': 'prostate/mri/detector.pt',
    
    # Cervical
    'cervical/pap/classifier.pt': 'cervical/pap/classifier.pt',  # MISSING
    'cervical/colposcopy/detector.pt': 'cervical/colposcopy/detector.pt',
    
    # Thyroid
    'thyroid/ultrasound/classifier.pt': 'thyroid/ultrasound/classifier.pt',
    
    # Root level
    'modality_classifier.pt': 'modality_classifier.pt',  # MISSING
}

print("=" * 70)
print("MODEL STATUS CHECK (using ACTUAL paths)")
print("=" * 70)

missing = []
for expected_path, actual_path in expected.items():
    p = base / actual_path
    status = "[OK] EXISTS" if p.exists() else "[MISSING] MISSING"
    if not p.exists():
        missing.append((expected_path, actual_path))
    print(f"  {expected_path:<45} -> {actual_path:<40} {status}")

print()
print("=" * 70)
if missing:
    print(f"MISSING: {len(missing)} files you MUST download manually:")
    for exp, act in missing:
        print(f"  - {exp} (place at: models/pretrained/cancers/{act})")
else:
    print("ALL MODELS PRESENT! 🎉")
print("=" * 70)