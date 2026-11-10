#!/usr/bin/env python
"""
Download missing volume models from MONAI Model Zoo
"""
import os
import sys
import requests
from pathlib import Path

MODEL_URLS = {
    "brain": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_01/swin_unetr_brain_tumor_segmentation.pt",
    "liver": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_01/swin_unetr_liver_segmentation.pt",
    "prostate": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_01/swin_unetr_prostate_segmentation.pt",
    "lung": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_01/unet3d_lung_nodule_segmentation.pt",
    "pancreas": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_01/unet3d_pancreas_segmentation.pt",
    "spleen": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_01/unet3d_spleen_segmentation.pt",
    "colorectal": "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_01/unet3d_colorectal_polyp_segmentation.pt",
}

TARGETS = {
    "brain": {"modality": "mri", "url_key": "brain"},
    "liver": {"modality": "ct", "url_key": "liver"},
    "prostate": {"modality": "mri", "url_key": "prostate"},
    "lung": {"modality": "ct", "url_key": "lung"},
    "pancreas": {"modality": "ct", "url_key": "pancreas"},
    "spleen": {"modality": "ct", "url_key": "spleen"},
    "colorectal": {"modality": "colonoscopy", "url_key": "colorectal"},
}

def download_model(url, dest_path):
    """Download a model file"""
    try:
        print(f"Downloading to {dest_path}...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  Downloaded {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def main():
    import requests
    base_dir = Path("models/pretrained/cancers")
    
    print("Downloading missing volume models from MONAI Model Zoo...")
    print("=" * 60)
    
    success = 0
    for cancer_key, info in TARGETS.items():
        modality = info["modality"]
        url_key = info["url_key"]
        url = MODEL_URLS.get(url_key)
        
        if not url:
            print(f"SKIP {cancer_key}: No URL mapping")
            continue
            
        modality_dir = Path("models/pretrained/cancers") / cancer_key / info["modality"]
        dest_path = Path("models/pretrained/cancers") / cancer_key / info["modality"] / "volume.pt2"
        
        if dest_path.exists() and dest_path.stat().st_size > 10 * 1024 * 1024:
            print(f"SKIP {cancer_key}: Already exists")
            continue
            
        print(f"\nDownloading {cancer_key} ({info['modality']})...")
        if download_model(MODEL_URLS[info["url_key"]], dest_path):
            # Create manifest
            manifest = {
                "model_name": f"{cancer_key}_segmentation",
                "model_type": "volume_segmentation_3d",
                "cancer_type": cancer_key,
                "modality": info["modality"],
                "model_file": "volume.pt2",
                "preprocessing": {"normalize": "ct" if "ct" in info["modality"] else "standard", "spacing": [1.0, 1.0, 1.0], "orientation": "RAS"},
                "postprocessing": {"threshold": 0.5, "largest_connected_component": True},
                "inference": {"patch_size": [96, 96, 96], "overlap": 0.5, "batch_size": 1, "device": "cuda"},
                "metadata": {"created_by": "OncoVision Downloader", "version": "1.0", "framework": "pytorch"}
            }
            
            manifest_path = Path("models/pretrained/cancers") / cancer_key / info["modality"] / "volume.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, 'w') as f:
                json.dump({"model_name": f"{cancer_key}_segmentation", "model_type": "volume_segmentation_3d", "cancer_type": cancer_key, "modality": info["modality"], "model_file": "volume.pt2"}, f, indent=2)
            
            print(f"  SUCCESS")
            success += 1
        else:
            print(f"  FAILED")
    
    print(f"\nCompleted: {success}/{len(TARGETS)} models downloaded")

if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()