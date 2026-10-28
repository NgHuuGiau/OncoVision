from pathlib import Path
import torch
from safetensors import safe_open

models = [
    'models/pretrained/cancers/lung/lung_yolo_tiny.safetensors',
    'models/pretrained/cancers/lung/lung_yolo11_best.pt',
    'models/pretrained/cancers/lung/lung_yolo11_last.pt',
    'models/pretrained/cancers/lung/lung_monai_nodule.pt',
    'models/pretrained/cancers/lung/lung_monai_nodule_test.pt',
    'models/pretrained/cancers/breast/breast_mass_detection.pt',
    'models/pretrained/cancers/liver/liver_tumor_seg.pth',
    'models/pretrained/cancers/liver/liver_unet.pth',
    'models/pretrained/cancers/liver/liver_tumor_seg.pth',
    'models/pretrained/cancers/liver/liver_unet.pth',
]

for m in models:
    p = Path(m)
    if not p.exists():
        print('NOT FOUND:', m)
        continue
    try:
        if p.suffix == '.safetensors':
            with safe_open(str(p), framework='pt') as f:
                keys = list(f.keys())
                print(f'{p.name}: {len(keys)} tensors (safetensors)')
        else:
            ckpt = torch.load(str(p), map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict):
                print(f'{p.name}: {len(ckpt)} keys, type=dict')
                if 'state_dict' in ckpt:
                    print(f'  state_dict: {len(ckpt["state_dict"])} tensors')
                elif 'model_state_dict' in ckpt:
                    print(f'  model_state_dict: {len(ckpt["model_state_dict"])} tensors')
            else:
                print(f'{p.name}: type={type(ckpt)}')
    except Exception as e:
        print(f'FAIL {p.name}: {type(e).__name__}: {str(e)[:80]}')