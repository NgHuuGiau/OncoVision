import torch
from pathlib import Path
from safetensors import safe_open

CANCERS_DIR = Path('models/pretrained/cancers')

# Batch 1: Liver, Thyroid, Brain
safetensor_models = [
    ('liver/liver_swin.safetensors', 'liver/liver_swin.pt', ['benign', 'malignant'], 'swin'),
    ('liver/model.safetensors', 'liver/model.pt', ['benign', 'malignant'], 'resnet50'),
    ('thyroid/thyroid_efficientnet.safetensors', 'thyroid/thyroid_efficientnet.pt', ['benign', 'malignant'], 'efficientnet_b0'),
    ('thyroid/thyroid_resnet50.safetensors', 'thyroid/thyroid_resnet50.pt', ['benign', 'malignant'], 'resnet50'),
    ('thyroid/thyroid_vit.safetensors', 'thyroid/thyroid_vit.pt', ['benign', 'malignant'], 'vit'),
    ('brain/glioma_segformer.safetensors', 'brain/glioma_segformer.pt', ['glioma', 'meningioma', 'pituitary', 'no_tumor'], 'vit'),
    ('brain/meningioma_segformer.safetensors', 'brain/meningioma_segformer.pt', ['glioma', 'meningioma', 'pituitary', 'no_tumor'], 'vit'),
    ('brain/model.safetensors', 'brain/brain_classifier_from_safetensors.pt', ['glioma', 'meningioma', 'pituitary', 'no_tumor'], 'resnet50'),
]

for src_name, dst_name, labels, backbone in safetensor_models:
    src = CANCERS_DIR / src_name
    dst = CANCERS_DIR / dst_name
    if not src.exists():
        print(f'SKIP {src_name}: not found')
        continue
    if dst.exists():
        print(f'SKIP {dst_name}: exists')
        continue
    try:
        with safe_open(str(src), framework='pt') as f:
            state = {k: f.get_tensor(k) for k in f.keys()}
        if not any(k.startswith('backbone.') for k in state.keys()):
            state = {f'backbone.{k}': v for k, v in state.items()}
        wrapped = {
            'model_state_dict': state,
            'class_labels': labels,
            'num_classes': len(labels),
            'backbone': backbone,
            'dropout': 0.3,
            'head': 'mlp',
            'image_size': 320,
            'interpolation': 'bilinear',
            'temperature': 1.0,
            'source_file': src_name,
        }
        torch.save(wrapped, str(dst))
        print(f'OK: {src_name} -> {dst_name} (backbone={backbone})')
    except Exception as e:
        print(f'FAIL {src_name}: {e}')