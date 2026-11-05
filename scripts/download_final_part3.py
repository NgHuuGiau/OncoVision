#!/usr/bin/env python3
"""Part 3: Main download logic"""

import os
import io
import zipfile
import requests
import shutil
from pathlib import Path

from download_final_part1 import BASE, ROOT, ensure_dir, download_file
from download_final_part2 import download_and_extract_zip, download_yolo_roboflow

def download_yolo_models():
    print("\n--- YOLO Detectors via HF Hub ---")
    yolo_models = [
        ("arjans-workspace/brain-tumor-detection", "best.pt", "brain", "detector.pt"),
        ("shree-sn/lung_nodule_detection", "best.pt", "lung/ct", "detector.pt"),
        ("liver2/liver-tumor", "best.pt", "liver/ct", "detector.pt"),
        ("kidney-tumor/testing", "best.pt", "kidney/ct", "detector.pt"),
    ]
    
    for repo, filename, organ_dir, target_name in yolo_models:
        target_dir = Path(r"D:\Project\OncoVision\models\pretrained\cancers") / organ_dir
        if download_yolo_roboflow(repo, "best.pt", str(Path(r"D:\Project\OncoVision\models\pretrained\cancers") / organ_dir), "detector.pt"):
            print(f"  OK: {repo}")
        else:
            print(f"  FAIL: {repo} - Manual: https://universe.roboflow.com")

def print_classifiers_manual():
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

def download_3d_models():
    print("\n--- 3D Models (MONAI) ---")
    
    # Pancreas 3D
    print("\n[Pancreas 3D - ~1GB]")
    try:
        url = "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/pancreas_ct_dints_segmentation_v0.4.3.zip"
        target_dir = Path("models/pretrained/cancers/pancreas")
        print("  Downloading ~1GB ZIP...")
        resp = requests.get("https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/pancreas_ct_dints_segmentation_v0.4.3.zip", stream=True, timeout=600)
        if resp.status_code == 200:
            zip_data = io.BytesIO(resp.content)
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                model_file = "pancreas_ct_dints_segmentation/models/model.pt"
                config_file = "pancreas_ct_dints_segmentation/configs/inference.json"
                if model_file in z.namelist():
                    z.extract(model_file, "models/pretrained/cancers/pancreas/")
                    os.rename("models/pretrained/cancers/pancreas/" + model_file, "models/pretrained/cancers/pancreas/volume.pt")
                    print("  OK: volume.pt")
                if config_file in z.namelist():
                    z.extract(config_file, "models/pretrained/cancers/pancreas/")
                    os.rename("models/pretrained/cancers/pancreas/" + config_file, "models/pretrained/cancers/pancreas/volume.json")
                    print("  OK: volume.json")
    except Exception as e:
        print(f"  FAIL: {e}")
    
    # Kidney 3D
    print("\n[Kidney 3D]")
    try:
        resp = requests.get("https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/renalStructures_UNEST_segmentation_v0.2.2.zip", stream=True, timeout=600)
        if resp.status_code == 200:
            zip_data = io.BytesIO(resp.content)
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                model_file = "renalStructures_UNEST_segmentation/models/model.pt"
                config_file = "renalStructures_UNEST_segmentation/configs/inference.json"
                if model_file in z.namelist():
                    z.extract(model_file, "models/pretrained/cancers/kidney/")
                    os.rename("models/pretrained/cancers/kidney/" + model_file, "models/pretrained/cancers/kidney/volume.pt")
                    print("  OK: kidney/volume.pt")
                if config_file in z.namelist():
                    z.extract(config_file, "models/pretrained/cancers/kidney/")
                    os.rename("models/pretrained/cancers/kidney/" + config_file, "models/pretrained/cancers/kidney/volume.json")
                    print("  OK: kidney/volume.json")
    except Exception as e:
        print(f"  FAIL: {e}")

def download_modality():
    print("\n[Modality Classifier - DINOv2-small]")
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download("t22000t/dinov2-small-mednist", "pytorch_model.bin", local_dir="models/pretrained", local_dir_use_symlinks=False)
        os.rename(local, "models/pretrained/modality_classifier.pt")
        print("  OK: modality_classifier.pt")
    except Exception as e:
        print(f"  FAIL: {e}")
        print("  Manual: https://huggingface.co/t22000t/dinov2-small-mednist")