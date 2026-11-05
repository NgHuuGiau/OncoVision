import sys
sys.path.insert(0, '.')

from medical.cancer_model_registry import target_model_paths
import os

targets = ['brain', 'lung', 'breast', 'liver', 'stomach', 'colorectal', 'prostate', 'cervical', 'kidney', 'pancreas', 'thyroid']

print("=== MODEL FILES CHECK ===")
for t in targets:
    paths = target_model_paths(t)
    print(f"\n=== {t} ===")
    for k in ('classifier', 'detector', 'volume_model', 'volume_manifest'):
        p = paths.get(k)
        if p:
            exists = os.path.exists(p)
            size = os.path.getsize(p) if exists else 0
            print(f"  {k:15s}: {'EXISTS' if exists else 'MISSING'} ({size:,} bytes)  {p}")

print("\n=== MODALITY SUBDIRECTORIES ===")
for t in targets:
    base = os.path.join('models', 'pretrained', 'cancers', t)
    if os.path.exists(base):
        for mod in os.listdir(base):
            mod_path = os.path.join(base, mod)
            if os.path.isdir(mod_path):
                files = [f for f in os.listdir(mod_path) if f.endswith(('.pt', '.json', '.safetensors', '.pth'))]
                print(f"  {t}/{mod}: {files}")