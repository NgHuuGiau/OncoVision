#!/usr/bin/env python3
"""Part 1: Setup and imports"""
import os
import sys
import io
import zipfile
import requests
import shutil
from pathlib import Path

BASE = Path(r"D:\Project\OncoVision\models\pretrained\cancers")
ROOT = BASE.parent

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def download_file(url, dest_path):
    try:
        print(f"  Downloading: {url}")
        resp = requests.get(url, stream=True, timeout=300)
        if resp.status_code == 200:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"  OK: {dest_path}")
            return True
        else:
            print(f"  FAIL HTTP {resp.status_code}: {url}")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False