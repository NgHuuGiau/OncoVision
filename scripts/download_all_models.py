#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime

BASE_DIR = r"D:\Project\OncoVision\models\pretrained"

MODELS_TO_DOWNLOAD = [
    {"target": "brain", "modality": "mri", "type": "classifier",
     "url": "https://huggingface.co/DenizAydin/brain-tumor-classification-resnet50/resolve/main/pytorch_model.bin",
     "filename": "raw_brain_resnet50.pth", "backbone": "resnet50", "head": "mlp",
     "class_labels": "glioma,meningioma,no_tumor,pituitary", "repo": "DenizAydin/brain-tumor-classification-resnet50"},
    {"target": "thyroid", "modality": "ultrasound", "type": "classifier",
     "url": "https://huggingface.co/mrm8488/thyroid-ultrasound-resnet18-tirads/resolve/main/pytorch_model.bin",
     "filename": "raw_thyroid_resnet18.pth", "backbone": "resnet18", "head": "mlp",
     "class_labels": "benign,malignant", "repo": "mrm8488/thyroid-ultrasound-resnet18-tirads"},
    {"target": "breast", "modality": "ultrasound", "type": "classifier",
     "url": "https://huggingface.co/albertvillanova/breast-cancer-ultrasound-resnet34/resolve/main/pytorch_model.bin",
     "filename": "raw_breast_resnet34.pth", "backbone": "resnet34", "head": "mlp",
     "class_labels": "normal,benign,malignant", "repo": "albertvillanova/breast-cancer-ultrasound-resnet34"},
    {"target": "lung", "modality": "ct", "type": "classifier",
     "url": "https://huggingface.co/AnkitA1/lung-cancer-resnet50/resolve/main/pytorch_model.bin",
     "filename": "raw_lung_resnet50.pth", "backbone": "resnet50", "head": "mlp",
     "class_labels": "benign,malignant,normal", "repo": "AnkitA1/lung-cancer-resnet50"},
    {"target": "liver", "modality": "ct", "type": "classifier",
     "url": "https://huggingface.co/focal-liver-lesion-ultrasound-resnet18/resolve/main/pytorch_model.bin",
     "filename": "raw_liver_resnet18.pth", "backbone": "resnet18", "head": "mlp",
     "class_labels": "benign,malignant,hemangioma,hcc", "repo": "focal-liver-lesion-ultrasound-resnet18"},
    {"target": "prostate", "modality": "mri", "type": "classifier", "url": None,
     "filename": "raw_prostate_resnet18.pth", "backbone": "resnet18", "head": "mlp",
     "class_labels": "benign,malignant", "repo": "local_train_required"},
    {"target": "kidney", "modality": "ct", "type": "classifier", "url": None,
     "filename": "raw_kidney_resnet18.pth", "backbone": "resnet18", "head": "mlp",
     "class_labels": "benign,malignant,cyst", "repo": "local_train_required"},
    {"target": "pancreas", "modality": "ct", "type": "classifier", "url": None,
     "filename": "raw_pancreas_resnet18.pth", "backbone": "resnet18", "head": "mlp",
     "class_labels": "normal,pdac,neuroendocrine", "repo": "local_train_required"},
    {"target": "stomach", "modality": "endoscopy", "type": "classifier", "url": None,
     "filename": "raw_stomach_resnet34.pth", "backbone": "resnet34", "head": "mlp",
     "class_labels": "normal,ulcer,erosion,carcinoma", "repo": "local_train_required"},
    {"target": "colorectal", "modality": "endoscopy", "type": "classifier", "url": None,
     "filename": "raw_colorectal_resnet34.pth", "backbone": "resnet34", "head": "mlp",
     "class_labels": "normal,polyp,adenoma,carcinoma", "repo": "local_train_required"},
    {"target": "cervical", "modality": "endoscopy", "type": "classifier", "url": None,
     "filename": "raw_cervical_resnet18.pth", "backbone": "resnet18", "head": "mlp",
     "class_labels": "normal,dysplastic,squamous_cell_carcinoma", "repo": "local_train_required"},
]

def download_file(url, save_path):
    print(f"[DL] {url}")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"[OK] Saved: {save_path}"); return True
    except Exception as e:
        print(f"[ERR] {e}"); return False

def setup_classifier(item):
    target, modality = item["target"], item["modality"]
    out_dir = os.path.join(BASE_DIR, "cancers", target, modality)
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(BASE_DIR, item["filename"])

    if item["url"]:
        if not os.path.exists(raw_path):
            if not download_file(item["url"], raw_path): return
    else:
        if not os.path.exists(raw_path):
            print(f"[INFO] {target}/{modality}: No public model. Creating dummy...")
            import torch
            from torch import nn
            from torchvision import models
            dummy = models.resnet18(weights=None)
            dummy.fc = nn.Linear(dummy.fc.in_features, len(item["class_labels"].split(",")))
            torch.save({"model_state_dict": dummy.state_dict()}, raw_path)
            print(f"[OK] Dummy created: {raw_path}")

    cmd = [sys.executable, "scripts/export_to_contract.py", "classifier",
           "--weights-path", raw_path, "--backbone", item["backbone"], "--head", item["head"],
           "--class-labels", item["class_labels"], "--image-size", "320",
           "--output-dir", out_dir, "--repo", item["repo"],
           "--repo-type", "huggingface" if item["url"] else "local",
           "--notes", f"Auto converted for {target}/{modality}" + (" (DUMMY)" if not item["url"] else "")]
    print(f"[CONVERT] {target}/{modality}...")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0: print(f"[ERR] {result.stderr}")
    else: print(f"[OK] {out_dir}/classifier.pt")

    val_json = os.path.join(out_dir, "clinical_validation.json")
    if not os.path.exists(val_json):
        data = {"target": target, "modality": modality, "model_file": "classifier.pt",
                "test_set": {"name": f"{target}_{modality}_val", "cases": 100},
                "metrics": {"accuracy": 0.0, "balanced_accuracy": 0.0, "auc_macro": 0.0, "note": "PLACEHOLDER"},
                "evaluated_at": datetime.utcnow().isoformat()+"Z", "evaluator": "auto"}  # noqa: DTZ003 - stamp UTC with Z suffix
        with open(val_json, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
        print("[OK] clinical_validation.json created")

def main():
    print("="*60); print("DOWNLOAD & CONVERT 11 CANCER GROUPS"); print("="*60)
    os.makedirs(BASE_DIR, exist_ok=True)
    for item in MODELS_TO_DOWNLOAD: setup_classifier(item)
    print("\n[STATUS] Running run_medical.py status...")
    subprocess.run([sys.executable, "run_medical.py", "status"], check=False)
    print("\n" + "="*60 + "\nDONE. Check output above.\n" + "="*60)

if __name__ == "__main__": main()
