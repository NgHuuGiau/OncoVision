from huggingface_hub import hf_hub_download, list_repo_files
import os

# 1. Brain Tumor YOLO Detector from alanjafari/BrainTumorAI
print("Downloading Brain Tumor YOLO Detector...")
try:
    files = list_repo_files("alanjafari/BrainTumorAI")
    print(f"Available files: {files}")

    pt_files = [f for f in files if f.endswith('.pt')]
    if 'best.pt' in pt_files:
        target_file = 'best.pt'
    elif 'last.pt' in pt_files:
        target_file = 'last.pt'
    elif pt_files:
        target_file = pt_files[0]
    else:
        print("No .pt files found in alanjafari/BrainTumorAI")
        target_file = None

    if target_file:
        downloaded = hf_hub_download(
            repo_id="alanjafari/BrainTumorAI",
            filename=target_file,
            local_dir="D:/Project/OncoVision/models/pretrained/cancers/brain",
            local_dir_use_symlinks=False
        )
        final_path = "D:/Project/OncoVision/models/pretrained/cancers/brain/detector.pt"
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(downloaded, final_path)
        print(f"Saved as {final_path}")
except Exception as e:
    print(f"Error downloading YOLO detector: {e}")

# 2. Brain 3D Volume Segmentation from MONAI
print("\nDownloading Brain 3D Volume Segmentation (MONAI)...")
try:
    files = list_repo_files("MONAI/brats_mri_segmentation")
    print(f"Available files: {files}")

    model_files = [f for f in files if f.endswith(('.pt', '.pth', '.ts', '.bin', '.safetensors'))]
    print(f"Model files: {model_files}")

    if model_files:
        target_file = None
        for ext in ['.pt', '.pth', '.ts']:
            for f in model_files:
                if f.endswith(ext):
                    target_file = f
                    break
            if target_file:
                break
        if not target_file:
            target_file = model_files[0]

        downloaded = hf_hub_download(
            repo_id="MONAI/brats_mri_segmentation",
            filename=target_file,
            local_dir="D:/Project/OncoVision/models/pretrained",
            local_dir_use_symlinks=False
        )
        final_path = "D:/Project/OncoVision/models/pretrained/brain_volume_model.pt"
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(downloaded, final_path)
        print(f"Saved as {final_path}")
except Exception as e:
    print(f"Error downloading MONAI model: {e}")

# 3. Brain Tumor Classifier
print("\nDownloading Brain Tumor Classifier...")
try:
    files = list_repo_files("him-09/brain-tumor-cl")
    print(f"Available files: {files}")

    model_files = [f for f in files if f.endswith(('.pt', '.pth', '.bin', '.safetensors'))]
    print(f"Model files: {model_files}")

    if model_files:
        target_file = model_files[0]
        downloaded = hf_hub_download(
            repo_id="him-09/brain-tumor-cl",
            filename=target_file,
            local_dir="D:/Project/OncoVision/models/pretrained/cancers/brain",
            local_dir_use_symlinks=False
        )
        final_path = "D:/Project/OncoVision/models/pretrained/cancers/brain/classifier.pt"
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(downloaded, final_path)
        print(f"Saved as {final_path}")
    else:
        print("No model files found, trying alternative...")
        raise Exception("No model files in him-09/brain-tumor-cl")
except Exception as e:
    print(f"Error with him-09/brain-tumor-cl: {e}")
    print("Trying alternative classifier...")
    try:
        files = list_repo_files("keremberke/brain-tumor-classification")
        print(f"Alternative repo files: {files}")
        model_files = [f for f in files if f.endswith(('.pt', '.pth', '.bin', '.safetensors'))]
        if model_files:
            target_file = model_files[0]
            downloaded = hf_hub_download(
                repo_id="keremberke/brain-tumor-classification",
                filename=target_file,
                local_dir="D:/Project/OncoVision/models/pretrained/cancers/brain",
                local_dir_use_symlinks=False
            )
            final_path = "D:/Project/OncoVision/models/pretrained/cancers/brain/classifier.pt"
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(downloaded, final_path)
            print(f"Saved as {final_path}")
    except Exception as e2:
        print(f"Alternative also failed: {e2}")

print("\nAll downloads complete!")