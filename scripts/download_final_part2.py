#!/usr/bin/env python3
"""Part 2: Zip extraction and YOLO download functions"""
import zipfile
import io
import os

def download_and_extract_zip(url, target_dir, model_in_zip, config_in_zip, model_out, config_out):
    try:
        print(f"  Downloading ZIP from {url}...")
        resp = requests.get(url, stream=True, timeout=600)
        if resp.status_code != 200:
            print(f"  FAIL HTTP {resp.status_code}")
            return False
        zip_data = io.BytesIO(resp.content)
        with zipfile.ZipFile(zip_data) as z:
            if model_in_zip in z.namelist():
                z.extract(model_in_zip, target_dir)
                os.rename(
                    os.path.join(target_dir, model_in_zip),
                    os.path.join(target_dir, model_out)
                )
                print(f"  OK: {model_out}")
            else:
                print(f"  FAIL Model not in zip")
                return False
            if config_in_zip in z.namelist():
                z.extract(config_in_zip, target_dir)
                os.rename(
                    os.path.join(target_dir, config_in_zip),
                    os.path.join(target_dir, config_out)
                )
                print(f"  OK: {config_out}")
            else:
                print(f"  WARN Config not found")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def download_yolo_roboflow(repo, filename, target_dir, target_name):
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id=repo, 
            filename=filename, 
            local_dir=target_dir, 
            local_dir_use_symlinks=False
        )
        final_path = os.path.join(target_dir, target_name)
        if local_path != final_path:
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(local_path, final_path)
        print(f"  OK: {target_name}")
        return True
    except Exception as e:
        print(f"  FAIL {repo}/{filename}: {e}")
        return False