import os
import sys

base = r'D:\Project\OncoVision\models\pretrained\cancers'

print('=== MODEL FILES & SIZES ===')
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith(('.pt', '.pt2', '.pth', '.onnx', '.safetensors', '.json')):
            p = os.path.join(root, f)
            sz = os.path.getsize(p) / 1024 / 1024
            rel = os.path.relpath(p, base)
            status = "OK" if sz > 10 else "CORRUPT/SMALL"
            print(f'{rel:60s} {sz:6.1f} MB {status}')

# Summary
print('\n=== SUMMARY ===')
cancers = ['brain', 'breast', 'cervical', 'colorectal', 'kidney', 'liver', 'lung', 'pancreas', 'prostate', 'stomach', 'thyroid']
for c in cancers:
    c_path = os.path.join(base, c)
    if os.path.exists(c_path):
        pt2_files = [f for f in os.listdir(c_path) if f.endswith('.pt2')]
        pt_files = [f for f in os.listdir(c_path) if f.endswith('.pt') and not f.endswith('.pt2')]
        json_files = [f for f in os.listdir(c_path) if f.endswith('.json')]
        pt2_size = sum(os.path.getsize(os.path.join(c_path, f)) for f in pt2_files) / 1024 / 1024
        pt_size = sum(os.path.getsize(os.path.join(c_path, f)) for f in pt_files) / 1024 / 1024
        has_classifier = any('classifier' in f for f in pt_files)
        has_detector = any('detector' in f for f in pt_files)
        has_volume = len(pt2_files) > 0
        print(f'{c:12s} | Classifier: {"✅" if has_classifier else "❌"} | Detector: {"✅" if has_detector else "❌"} | Volume: {"✅" if has_volume else "❌"} ({pt2_size:.1f}MB) | JSON: {len(json_files)}')
    else:
        print(f'{c:12s} | MISSING FOLDER')