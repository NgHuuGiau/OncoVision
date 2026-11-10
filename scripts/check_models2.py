import os
os.chdir(r'D:\Project\OncoVision')

import torch

# Test loading the suspicious volume models
volume_paths = [
    r'D:\Project\OncoVision\models\pretrained\cancers\brain\mri\brain_volume_model.pt2',
    r'D:\Project\OncoVision\models\pretrained\cancers\lung\ct\volume.pt2',
    r'D:\Project\OncoVision\models\pretrained\cancers\liver\ct\volume.pt2',
    r'D:\Project\OncoVision\models\pretrained\cancers\prostate\mri\volume.pt2',
    r'D:\Project\OncoVision\models\pretrained\cancers\kidney\ct\volume.pt2',
    r'D:\Project\OncoVision\models\pretrained\cancers\pancreas\ct\volume.pt2',
    r'D:\Project\OncoVision\models\pretrained\cancers\stomach\endoscopy\stomach_volume_model.pt2',
    r'D:\Project\OncoVision\models\pretrained\cancers\cervical\colposcopy\cervical_volume_model.pt2',
    r'D:\Project\OncoVision\models\pretrained\cancers\thyroid\ultrasound\thyroid_volume_model.pt2',
]

print("=== Testing volume models (.pt2) ===")
for path in volume_paths:
    if os.path.exists(path):
        try:
            m = torch.load(path, map_location='cpu', weights_only=False)
            print(f"[OK] {os.path.basename(path)}: OK - type: {type(m)}")
        except Exception as e:
            print(f"[FAIL] {os.path.basename(path)}: FAILED - {str(e)[:80]}")
    else:
        print(f"[MISSING] {os.path.basename(path)}: FILE NOT FOUND")

print("\n=== Testing classifier models ===")
classifier_paths = [
    r'D:\Project\OncoVision\models\pretrained\cancers\brain\mri\classifier.pt',
    r'D:\Project\OncoVision\models\pretrained\cancers\lung\ct\classifier.pt',
    r'D:\Project\OncoVision\models\pretrained\cancers\liver\ct\classifier.pt',
    r'D:\Project\OncoVision\models\pretrained\cancers\breast\ultrasound\classifier.pt',
    r'D:\Project\OncoVision\models\pretrained\cancers\modality_classifier.pt',
]

for path in classifier_paths:
    if os.path.exists(path):
        try:
            m = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(m, dict):
                print(f"[OK] {os.path.basename(path)}: OK - keys: {list(m.keys())}")
            else:
                print(f"[OK] {os.path.basename(path)}: OK - type: {type(m)}")
        except Exception as e:
            print(f"[FAIL] {os.path.basename(path)}: FAILED - {str(e)[:80]}")
    else:
        print(f"[MISSING] {os.path.basename(path)}: FILE NOT FOUND")