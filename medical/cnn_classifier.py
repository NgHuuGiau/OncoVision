from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from medical.dashboard import write_training_progress
from medical.losses import ASLLoss, BalancedSoftmaxLoss, FocalLoss, FocalTverskyLoss, LDAMLoss
from medical.network_policy import resolve_pretrained


_BACKBONE_REGISTRY: dict[str, tuple[Any, int]] = {}


def _register_backbone(name: str, builder: Any, feature_dim: int) -> None:
    _BACKBONE_REGISTRY[name] = (builder, feature_dim)


def _get_backbone_builder(name: str) -> tuple[Any, int]:
    name = name.lower()
    if name in _BACKBONE_REGISTRY:
        return _BACKBONE_REGISTRY[name]
    raise ValueError(f"Unsupported backbone: {name}")


def _build_resnet_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    else:
        model = resnet50(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    return model, feature_dim


def _build_efficientnet_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import efficientnet_b0, efficientnet_b2, efficientnet_b3
    from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
    from torchvision.models import (
        EfficientNet_B0_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights,
        EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights,
    )
    weights_map = {
        "efficientnet_b0": EfficientNet_B0_Weights.DEFAULT if pretrained else None,
        "efficientnet_b2": EfficientNet_B2_Weights.DEFAULT if pretrained else None,
        "efficientnet_b3": EfficientNet_B3_Weights.DEFAULT if pretrained else None,
        "efficientnet_v2_s": EfficientNet_V2_S_Weights.DEFAULT if pretrained else None,
        "efficientnet_v2_m": EfficientNet_V2_M_Weights.DEFAULT if pretrained else None,
        "efficientnet_v2_l": EfficientNet_V2_L_Weights.DEFAULT if pretrained else None,
    }
    weights = weights_map.get(name)
    if name == "efficientnet_b0":
        model = efficientnet_b0(weights=weights)
    elif name == "efficientnet_b2":
        model = efficientnet_b2(weights=weights)
    elif name == "efficientnet_b3":
        model = efficientnet_b3(weights=weights)
    elif name == "efficientnet_v2_s":
        model = efficientnet_v2_s(weights=weights)
    elif name == "efficientnet_v2_m":
        model = efficientnet_v2_m(weights=weights)
    else:
        model = efficientnet_v2_l(weights=weights)
    feature_dim = model.classifier[1].in_features
    model.classifier = nn.Identity()
    return model, feature_dim


def _build_convnext_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "convnext_tiny":
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        feature_dim = model.classifier[2].in_features
        model.classifier = nn.Identity()
    elif name == "convnext_small":
        from torchvision.models import convnext_small, ConvNeXt_Small_Weights
        model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT if pretrained else None)
        feature_dim = model.classifier[2].in_features
        model.classifier = nn.Identity()
    elif name == "convnext_base":
        from torchvision.models import convnext_base, ConvNeXt_Base_Weights
        model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
        feature_dim = model.classifier[2].in_features
        model.classifier = nn.Identity()
    elif name in ("convnextv2_tiny", "convnext_tiny"):
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        feature_dim = model.classifier[2].in_features
        model.classifier = nn.Identity()
    elif name in ("convnextv2_base", "convnext_base"):
        from torchvision.models import convnext_base, ConvNeXt_Base_Weights

        model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
        feature_dim = model.classifier[2].in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported ConvNeXt backbone: {name}")
    return model, feature_dim


def _build_swin_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "swin_t":
        from torchvision.models import swin_t, Swin_T_Weights
        model = swin_t(weights=Swin_T_Weights.DEFAULT if pretrained else None)
        feature_dim = model.head.in_features
    elif name == "swin_s":
        from torchvision.models import swin_s, Swin_S_Weights
        model = swin_s(weights=Swin_S_Weights.DEFAULT if pretrained else None)
        feature_dim = model.head.in_features
    elif name == "swin_b":
        from torchvision.models import swin_b, Swin_B_Weights
        model = swin_b(weights=Swin_B_Weights.DEFAULT if pretrained else None)
        feature_dim = model.head.in_features
    elif name == "swinv2_t":
        from torchvision.models import swin_v2_t, Swin_V2_T_Weights
        model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT if pretrained else None)
        feature_dim = model.head.in_features
    elif name == "swinv2_s":
        from torchvision.models import swin_v2_s, Swin_V2_S_Weights
        model = swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT if pretrained else None)
        feature_dim = model.head.in_features
    elif name == "swinv2_b":
        from torchvision.models import swin_v2_b, Swin_V2_B_Weights
        model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT if pretrained else None)
        feature_dim = model.head.in_features
    else:
        raise ValueError(f"Unsupported Swin backbone: {name}")
    model.head = nn.Identity()
    return model, feature_dim


def _build_vit_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "vit_b_16":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
    elif name == "vit_l_16":
        from torchvision.models import vit_l_16, ViT_L_16_Weights
        model = vit_l_16(weights=ViT_L_16_Weights.DEFAULT if pretrained else None)
    elif name == "vit_h_14":
        from torchvision.models import vit_h_14, ViT_H_14_Weights
        model = vit_h_14(weights=ViT_H_14_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unsupported ViT backbone: {name}")
    feature_dim = model.heads.head.in_features
    model.heads.head = nn.Identity()
    return model, feature_dim


def _build_regnet_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "regnet_y_400mf":
        from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights
        model = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.DEFAULT if pretrained else None)
    elif name == "regnet_y_800mf":
        from torchvision.models import regnet_y_800mf, RegNet_Y_800MF_Weights
        model = regnet_y_800mf(weights=RegNet_Y_800MF_Weights.DEFAULT if pretrained else None)
    elif name == "regnet_y_1_6gf":
        from torchvision.models import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights
        model = regnet_y_1_6gf(weights=RegNet_Y_1_6GF_Weights.DEFAULT if pretrained else None)
    elif name == "regnet_y_3_2gf":
        from torchvision.models import regnet_y_3_2gf, RegNet_Y_3_2GF_Weights
        model = regnet_y_3_2gf(weights=RegNet_Y_3_2GF_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unsupported RegNet backbone: {name}")
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feature_dim


def _build_maxvit_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "maxvit_t":
        from torchvision.models import maxvit_t, MaxVit_T_Weights
        model = maxvit_t(weights=MaxVit_T_Weights.DEFAULT if pretrained else None)
        feature_dim = model.classifier[3].in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported MaxViT backbone: {name}")
    return model, feature_dim


_register_backbone("resnet18", _build_resnet_backbone, 512)
_register_backbone("resnet50", _build_resnet_backbone, 2048)
_register_backbone("efficientnet_b0", _build_efficientnet_backbone, 1280)
_register_backbone("efficientnet_b2", _build_efficientnet_backbone, 1408)
_register_backbone("efficientnet_b3", _build_efficientnet_backbone, 1536)
_register_backbone("efficientnet_v2_s", _build_efficientnet_backbone, 1280)
_register_backbone("efficientnet_v2_m", _build_efficientnet_backbone, 1280)
_register_backbone("efficientnet_v2_l", _build_efficientnet_backbone, 1280)
_register_backbone("convnext_tiny", _build_convnext_backbone, 768)
_register_backbone("convnext_small", _build_convnext_backbone, 768)
_register_backbone("convnext_base", _build_convnext_backbone, 1024)
_register_backbone("convnextv2_tiny", _build_convnext_backbone, 768)
_register_backbone("convnextv2_base", _build_convnext_backbone, 1024)
_register_backbone("swin_t", _build_swin_backbone, 768)
_register_backbone("swin_s", _build_swin_backbone, 768)
_register_backbone("swin_b", _build_swin_backbone, 1024)
_register_backbone("swinv2_t", _build_swin_backbone, 768)
_register_backbone("swinv2_s", _build_swin_backbone, 768)
_register_backbone("swinv2_b", _build_swin_backbone, 1024)
_register_backbone("vit_b_16", _build_vit_backbone, 768)
_register_backbone("vit_l_16", _build_vit_backbone, 1024)
_register_backbone("regnet_y_400mf", _build_regnet_backbone, 440)
_register_backbone("regnet_y_800mf", _build_regnet_backbone, 784)
_register_backbone("regnet_y_1_6gf", _build_regnet_backbone, 888)
_register_backbone("regnet_y_3_2gf", _build_regnet_backbone, 1512)
_register_backbone("maxvit_t", _build_maxvit_backbone, 512)


def _build_medical_augmentations(image_size: int = 320, modality: str = "default") -> transforms.Compose:
    base = [
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.RandomAdjustSharpness(sharpness_factor=0.3, p=0.4),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomEqualize(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ]
    return transforms.Compose(base)


def _build_base_transform(image_size: int = 320) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class MedicalImageDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        class_labels: tuple[str, ...],
        image_size: int = 320,
        augment: bool = False,
        modality: str = "default",
    ) -> None:
        self.samples = samples
        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        if augment:
            self.transform = _build_medical_augmentations(image_size=image_size, modality=modality)
        else:
            self.transform = _build_base_transform(image_size=image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, class_index = self.samples[idx]
        with Image.open(image_path) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
        return self.transform(image), class_index


class MedicalCNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.dropout = dropout
        builder, feature_dim = _get_backbone_builder(backbone)
        self.backbone = builder(backbone, resolve_pretrained(pretrained, context=f"cnn:{backbone}"))
        if isinstance(self.backbone, nn.Module):
            self.backbone = self.backbone
        else:
            self.backbone = self.backbone[0] if isinstance(self.backbone, tuple) else self.backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, max(64, feature_dim // 2)),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(max(64, feature_dim // 2), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            # Gộp không gian: global average pool về [B, C] để tương thích mọi backbone.
            features = features.mean(dim=(-2, -1)) if features.dim() == 4 else features.flatten(start_dim=1)
        return self.classifier(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return torch.softmax(logits, dim=1)


@dataclass(frozen=True)
class CNNClassifierResult:
    label: str
    confidence: float
    probabilities: dict[str, float]


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module, original_state: dict[str, Any]) -> None:
        model.load_state_dict(original_state, strict=False)


class CheckpointAveraging:
    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self.checkpoints: list[dict[str, Any]] = []

    def add(self, state_dict: dict[str, Any]) -> None:
        self.checkpoints.append({k: v.clone().cpu() for k, v in state_dict.items()})
        if len(self.checkpoints) > self.window_size:
            self.checkpoints.pop(0)

    def average(self) -> dict[str, Any]:
        if not self.checkpoints:
            return {}
        avg_state = {}
        for key in self.checkpoints[0].keys():
            avg_state[key] = torch.stack([c[key].float() for c in self.checkpoints]).mean(dim=0)
        return avg_state


class MedicalCNNClassifierWrapper:
    def __init__(
        self,
        model: MedicalCNNClassifier,
        class_labels: tuple[str, ...],
        device: str | None = None,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.class_labels = class_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = float(temperature)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        source: str | Path | np.ndarray,
        *,
        top_k: int = 3,
        tta: bool = False,
    ) -> list[dict[str, Any]]:
        if tta:
            probs = self._predict_with_tta(source)
        else:
            image_tensor = _load_image_as_tensor(source, assume_bgr=True)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            probs = self.model.predict_proba(image_tensor).cpu().numpy()[0]
        probs = self._apply_temperature(probs)
        ranked = np.argsort(-probs)
        results = []
        for idx in ranked[: max(1, top_k)]:
            results.append({
                "label": self.class_labels[idx],
                "confidence": float(probs[idx]),
                "probabilities": {self.class_labels[i]: float(probs[i]) for i in range(len(self.class_labels))},
            })
        return results

    def explain(self, source: str | Path | np.ndarray, *, top_k: int = 1, tta: bool = False, alpha: float = 0.5) -> list[dict[str, Any]]:
        try:
            from medical.explainability import MedicalGradCAMExplainer
            explainer = MedicalGradCAMExplainer(self, image_size=self.model.backbone_name and 320, device=self.device)
            gradcam_results = explainer.explain(source, top_k=top_k, tta=tta, alpha=alpha)
            return [
                {"label": result.label, "confidence": result.confidence, "heatmap": result.heatmap, "overlay": result.overlay}
                for result in gradcam_results
            ]
        except Exception:
            return []

    def _predict_with_tta(self, source: str | Path | np.ndarray) -> np.ndarray:
        base_tensor = _load_image_as_tensor(source, assume_bgr=True)
        base_tensor = base_tensor.unsqueeze(0).to(self.device)
        tta_transforms = [
            lambda x: x,
            lambda x: torch.flip(x, dims=[3]),
            lambda x: torch.flip(x, dims=[2]),
            lambda x: torch.roll(x, shifts=10, dims=[3]),
            lambda x: torch.roll(x, shifts=10, dims=[2]),
        ]
        all_probs = []
        for transform in tta_transforms:
            augmented = transform(base_tensor)
            probs = self.model.predict_proba(augmented).cpu().numpy()[0]
            all_probs.append(probs)
        return np.mean(all_probs, axis=0)

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        if self.temperature == 1.0:
            return probs
        logits = np.log(np.clip(probs, 1e-8, 1.0)) / self.temperature
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "class_labels": self.class_labels,
            "backbone": self.model.backbone_name,
            "num_classes": self.model.num_classes,
            "dropout": self.model.dropout,
            "temperature": self.temperature,
        }, target)
        return target

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "MedicalCNNClassifierWrapper":
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = MedicalCNNClassifier(
            num_classes=checkpoint["num_classes"],
            backbone=checkpoint.get("backbone", "resnet50"),
            pretrained=False,
            dropout=checkpoint.get("dropout", 0.3),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        class_labels = checkpoint["class_labels"]
        temperature = checkpoint.get("temperature", 1.0)
        return cls(model=model, class_labels=tuple(class_labels), device=device, temperature=temperature)


def _load_image_as_tensor(source: str | Path | np.ndarray, image_size: int = 320, *, assume_bgr: bool = True) -> torch.Tensor:
    transform = _build_base_transform(image_size=image_size)
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            source = np.stack([source] * 3, axis=-1)
        elif source.ndim == 3 and source.shape[-1] == 3 and assume_bgr:
            source = source[:, :, ::-1]
        image = Image.fromarray(source.astype(np.uint8), mode="RGB")
    else:
        with Image.open(source) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
    return transform(image)


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float | None = None
        self.counter = 0
        self.early_stop = False
        self.delta = min_delta if mode == "min" else -min_delta

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        if self.mode == "min":
            improved = score < self.best_score + self.delta
        else:
            improved = score > self.best_score - self.delta
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def _estimate_safe_batch_size(backbone: str, device: str | None = None) -> int:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" or not torch.cuda.is_available():
        return 8
    try:
        props = torch.cuda.get_device_properties(device)
        memory_mb = props.total_memory / (1024 * 1024)
    except Exception:
        return 8
    backbone_lower = backbone.lower()
    if backbone_lower in {"resnet50", "convnext_tiny", "convnext_small", "convnext_base", "swin_t", "swin_s", "swin_b",
                           "swinv2_t", "swinv2_s", "swinv2_b", "regnet_y_1_6gf", "regnet_y_3_2gf", "maxvit_t"}:
        per_image = 220.0
    elif backbone_lower in {"efficientnet_b0", "efficientnet_b2", "efficientnet_b3",
                             "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
                             "regnet_y_400mf", "regnet_y_800mf", "convnextv2_tiny", "convnextv2_base"}:
        per_image = 180.0
    elif backbone_lower.startswith("vit_"):
        per_image = 400.0
    else:
        per_image = 200.0
    estimated = int(memory_mb / per_image)
    safe = max(4, min(64, estimated))
    power = 1
    while power * 2 <= safe:
        power *= 2
    return max(4, power)


def _create_optimizer(model: nn.Module, learning_rate: float, weight_decay: float, optimizer_type: str = "adamw") -> torch.optim.Optimizer:
    if optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    elif optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def _create_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int, warmup_epochs: int = 3, scheduler_type: str = "cosine_warmup_restart") -> torch.optim.lr_scheduler.LRScheduler:
    if scheduler_type == "cosine_warmup_restart":
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs))
            return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    elif scheduler_type == "cosine_restart":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, num_epochs // 4), T_mult=2)
    elif scheduler_type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]["lr"], total_steps=max(1, num_epochs), pct_start=0.3, anneal_strategy="cos")
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.1)
    elif scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))


def _set_deterministic_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _training_progress_line(epoch: int, num_epochs: int, batch_idx: int, total_batches: int, running_loss: float, elapsed: float, phase: str = "train") -> str:
    width = 30
    total_batches = max(1, total_batches)
    filled = int(width * batch_idx / total_batches)
    bar = "#" * filled + "." * (width - filled)
    pct = int(100 * batch_idx / total_batches)
    return f"[train:{phase}] ep {epoch}/{num_epochs} [{bar}] {batch_idx}/{total_batches} ({pct}%) loss={running_loss:.4f} {elapsed:.0f}s"


def _compute_gradient_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return float(total_norm ** 0.5)


def train_cnn_classifier(
    samples: list[tuple[Path, int]],
    class_labels: tuple[str, ...],
    *,
    image_size: int = 320,
    backbone: str = "resnet50",
    pretrained: bool = True,
    dropout: float = 0.3,
    batch_size: int | None = None,
    num_epochs: int = 30,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str | None = None,
    val_samples: list[tuple[Path, int]] | None = None,
    early_stopping_patience: int = 7,
    label_smoothing: float = 0.1,
    mixed_precision: bool = True,
    warmup_epochs: int = 3,
    class_weights: list[float] | tuple[float, ...] | None = None,
    gradient_accumulation_steps: int = 1,
    progress_tag: str | None = None,
    verbose: bool = False,
    resume_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    enable_ema: bool = False,
    ema_decay: float = 0.999,
    enable_checkpoint_averaging: bool = False,
    checkpoint_averaging_window: int = 5,
    scheduler_type: str = "cosine_warmup_restart",
    optimizer_type: str = "adamw",
    gradient_clip_norm: float = 1.0,
    modality: str = "default",
    loss_function: str = "cross_entropy",
    focal_gamma: float = 2.0,
    focal_tversky_alpha: float = 0.7,
    focal_tversky_beta: float = 0.3,
    ldam_cls_num_list: list[int] | None = None,
    asl_gamma_pos: float = 0.0,
    asl_gamma_neg: float = 4.0,
    asl_clip: float = 0.05,
    balanced_softmax_beta: float = 0.5,
) -> tuple[MedicalCNNClassifierWrapper, dict[str, Any]]:
    if not samples:
        raise FileNotFoundError("Khong co du lieu train cho CNN classifier.")

    _set_deterministic_seed()
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if batch_size is None:
        batch_size = _estimate_safe_batch_size(backbone=backbone, device=str(device_obj))
    effective_batch = batch_size * max(1, int(gradient_accumulation_steps))

    train_dataset = MedicalImageDataset(samples, class_labels, image_size=image_size, augment=True, modality=modality)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=len(train_dataset) > batch_size,
    )

    val_loader = None
    if val_samples:
        val_dataset = MedicalImageDataset(val_samples, class_labels, image_size=image_size, augment=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

    num_classes = len(class_labels)
    model = MedicalCNNClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
    )

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    start_epoch = 0

    _loaded_checkpoint: dict[str, Any] | None = None
    if resume_path is not None or (checkpoint_path is not None and checkpoint_path.exists()):
        ckpt_file = checkpoint_path if (checkpoint_path is not None and checkpoint_path.exists()) else Path(resume_path)
        try:
            _loaded_checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            if isinstance(_loaded_checkpoint, dict) and "model_state_dict" in _loaded_checkpoint:
                model.load_state_dict(_loaded_checkpoint["model_state_dict"], strict=False)
                original_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                if "epoch" in _loaded_checkpoint:
                    start_epoch = int(_loaded_checkpoint["epoch"])
                print(f"[train] resume tu checkpoint: {ckpt_file} (epoch {start_epoch})", flush=True)
        except Exception as exc:
            print(f"[train] loi khi load checkpoint ({exc}), bat dau tu dau.", flush=True)

    model = model.to(device_obj)

    weight_tensor = None
    if class_weights is not None and len(class_weights) == num_classes:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device_obj)

    criterion = _build_loss(loss_function, weight_tensor, label_smoothing, num_classes, focal_gamma, focal_tversky_alpha, focal_tversky_beta, ldam_cls_num_list, asl_gamma_pos, asl_gamma_neg, asl_clip, balanced_softmax_beta)
    optimizer = _create_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay, optimizer_type=optimizer_type)
    scheduler = _create_scheduler(optimizer, num_epochs=num_epochs, warmup_epochs=warmup_epochs, scheduler_type=scheduler_type)
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")

    scaler = torch.amp.GradScaler(device_obj.type, enabled=mixed_precision and device_obj.type == "cuda")
    ema = EMA(model, decay=ema_decay) if enable_ema else None
    checkpoint_avg = CheckpointAveraging(window_size=checkpoint_averaging_window) if enable_checkpoint_averaging else None

    if _loaded_checkpoint is not None:
        ckpt = _loaded_checkpoint
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "history" in ckpt:
            history = ckpt["history"]
        if "best_model_state" in ckpt and ckpt["best_model_state"]:
            best_model_state = ckpt["best_model_state"]
        if "best_val_f1" in ckpt:
            best_val_f1 = float(ckpt["best_val_f1"])
        if "early_stopping_state" in ckpt:
            es_state = ckpt["early_stopping_state"]
            if isinstance(es_state, dict):
                early_stopping.best_score = es_state.get("best_score")
                early_stopping.counter = int(es_state.get("counter", 0))
                early_stopping.early_stop = bool(es_state.get("early_stop", False))
        if ema is not None and "ema_shadow" in ckpt:
            ema.shadow = ckpt["ema_shadow"]

    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(
        f"[train] khoi dong CNN: backbone={backbone} epochs={num_epochs} "
        f"batch={batch_size} device={device_obj.type} train={len(train_dataset)} anh",
        flush=True,
    )
    _progress_tty = sys.stdout.isatty()
    _progress_total_batches = len(train_loader)
    _progress_every = 1 if (_progress_tty or verbose) else max(1, _progress_total_batches // 25)
    _progress_start = time.time()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_roc_auc": [],
        "lr": [],
        "grad_norm": [],
        "effective_batch": [float(effective_batch)],
    }

    best_model_state: dict[str, Any] | None = None
    best_val_f1 = 0.0

    for epoch in range(start_epoch, num_epochs):
        print(f"[train] --- epoch {epoch + 1}/{num_epochs} ---", flush=True)
        model.train()
        total_loss = 0.0
        accumulated_steps = 0
        batch_idx = 0
        processed = 0
        for images, labels in train_loader:
            images = images.to(device_obj, non_blocking=True)
            labels = labels.to(device_obj, non_blocking=True)
            with torch.amp.autocast(device_obj.type, enabled=scaler.is_enabled()):
                outputs = model(images)
                loss = criterion(outputs, labels) / max(1, gradient_accumulation_steps)
            scaler.scale(loss).backward()
            accumulated_steps += 1
            if accumulated_steps >= gradient_accumulation_steps:
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulated_steps = 0
            total_loss += loss.item() * images.size(0) * max(1, gradient_accumulation_steps)
            processed += images.size(0)
            batch_idx += 1
            if batch_idx % _progress_every == 0 or batch_idx == _progress_total_batches:
                running_loss = total_loss / max(1, processed)
                line = _training_progress_line(
                    epoch + 1, num_epochs, batch_idx, _progress_total_batches, running_loss, time.time() - _progress_start
                )
                if _progress_tty:
                    print(f"\r{line}", end="", flush=True)
                else:
                    print(line, flush=True)

        if accumulated_steps > 0:
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if ema is not None:
            ema.update(model)
        if checkpoint_avg is not None:
            checkpoint_avg.add(model.state_dict())

        scheduler.step()
        epoch_loss = total_loss / len(train_dataset)
        history["train_loss"].append(epoch_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["grad_norm"].append(_compute_gradient_norm(model))

        val_loss = 0.0
        val_acc = 0.0
        val_f1 = 0.0
        val_roc_auc = 0.0
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            correct = 0
            total = 0
            val_total_batches = len(val_loader)
            val_batch_idx = 0
            val_processed = 0
            all_probs = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device_obj, non_blocking=True)
                    labels = labels.to(device_obj, non_blocking=True)
                    with torch.amp.autocast(device_obj.type, enabled=scaler.is_enabled()):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    val_loss_sum += loss.item() * images.size(0)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_probs.append(probs)
                    all_labels.append(labels.cpu().numpy())
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    val_processed += images.size(0)
                    val_batch_idx += 1
                    if val_batch_idx % _progress_every == 0 or val_batch_idx == val_total_batches:
                        val_running = val_loss_sum / max(1, val_processed)
                        val_line = _training_progress_line(
                            epoch + 1, num_epochs, val_batch_idx, val_total_batches, val_running, time.time() - _progress_start, phase="val"
                        )
                        if _progress_tty:
                            print(f"\r{val_line}", end="", flush=True)
                        else:
                            print(val_line, flush=True)
            val_loss = val_loss_sum / total
            val_acc = correct / total
            all_probs = np.concatenate(all_probs, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            val_f1 = _compute_macro_f1(all_probs, all_labels, num_classes)
            from medical.metrics import compute_roc_auc
            try:
                roc_result = compute_roc_auc(all_labels, all_probs, list(class_labels))
                val_roc_auc = float(np.nanmean(list(roc_result["per_class"].values()))) if roc_result["per_class"] else 0.0
            except Exception:
                val_roc_auc = 0.0
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            history["val_roc_auc"].append(val_roc_auc)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = {
                    "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "val_f1": best_val_f1,
                    "val_acc": val_acc,
                }

            if early_stopping(val_f1):
                if best_model_state is not None:
                    model.load_state_dict(best_model_state["model_state_dict"])
                break

        if _progress_tty:
            print(flush=True)
        write_training_progress(
            backend="cnn",
            tag=progress_tag,
            epoch=epoch + 1,
            num_epochs=num_epochs,
            train_loss=epoch_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=float(optimizer.param_groups[0]["lr"]),
            backbone=backbone,
            best_val_f1=best_val_f1,
            stopped_early=early_stopping.early_stop,
        )
        print(
            f"[train] epoch {epoch + 1}/{num_epochs} "
            f"loss={epoch_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"val_roc_auc={val_roc_auc:.4f} best_val_f1={best_val_f1:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
            + (" [early-stop]" if early_stopping.early_stop else ""),
            flush=True,
        )

        if checkpoint_path is not None:
            ckpt = {
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if hasattr(scaler, "state_dict") else None,
                "epoch": epoch + 1,
                "history": history,
                "best_model_state": best_model_state,
                "best_val_f1": best_val_f1,
                "early_stopping_state": {
                    "best_score": early_stopping.best_score,
                    "counter": early_stopping.counter,
                    "early_stop": early_stopping.early_stop,
                },
                "effective_batch": effective_batch,
            }
            if ema is not None:
                ckpt["ema_shadow"] = {k: v.cpu() for k, v in ema.shadow.items()}
            try:
                torch.save(ckpt, checkpoint_path, _use_new_zipfile_serialization=False)
            except Exception:
                pass

    if checkpoint_avg is not None and checkpoint_avg.checkpoints:
        avg_state = checkpoint_avg.average()
        if avg_state:
            model.load_state_dict(avg_state, strict=False)
    elif ema is not None:
        ema.apply(model)
    elif best_model_state is not None:
        model.load_state_dict(best_model_state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(original_state, strict=False)

    wrapper = MedicalCNNClassifierWrapper(model=model, class_labels=class_labels, device=str(device_obj))
    return wrapper, history


def _compute_macro_f1(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    preds = np.argmax(probs, axis=1)
    f1s = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        precision = tp / max(tp + fp, 1e-6)
        recall = tp / max(tp + fn, 1e-6)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        f1s.append(f1)
    return float(np.mean(f1s))


def _build_loss(
    loss_function: str,
    weight_tensor: torch.Tensor | None,
    label_smoothing: float,
    num_classes: int,
    focal_gamma: float,
    focal_tversky_alpha: float,
    focal_tversky_beta: float,
    ldam_cls_num_list: list[int] | None,
    asl_gamma_pos: float,
    asl_gamma_neg: float,
    asl_clip: float,
    balanced_softmax_beta: float,
) -> nn.Module:
    if loss_function == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=weight_tensor.tolist() if weight_tensor is not None else None)
    elif loss_function == "focal_tversky":
        return FocalTverskyLoss(alpha=focal_tversky_alpha, beta=focal_tversky_beta)
    elif loss_function == "ldam":
        cls_num_list = ldam_cls_num_list or [1] * num_classes
        return LDAMLoss(cls_num_list=cls_num_list)
    elif loss_function == "asl":
        return ASLLoss(gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip)
    elif loss_function == "balanced_softmax":
        cls_num_list = ldam_cls_num_list or [1] * num_classes
        return BalancedSoftmaxLoss(cls_num_list=cls_num_list, beta=balanced_softmax_beta)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weight_tensor)


def load_cnn_classifier(path: str | Path, device: str | None = None) -> MedicalCNNClassifierWrapper:
    return MedicalCNNClassifierWrapper.load(path, device=device)


def is_cnn_classifier_path(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    if path.suffix != ".pt":
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return isinstance(checkpoint, dict) and "model_state_dict" in checkpoint and "class_labels" in checkpoint
    except Exception:
        return False


__all__ = [
    "MedicalCNNClassifier",
    "MedicalCNNClassifierWrapper",
    "CNNClassifierResult",
    "MedicalImageDataset",
    "train_cnn_classifier",
    "load_cnn_classifier",
    "is_cnn_classifier_path",
    "EarlyStopping",
    "EMA",
    "CheckpointAveraging",
    "_build_loss",
]
