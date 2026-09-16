from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from medical.network_policy import resolve_pretrained

logger = logging.getLogger(__name__)


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        feature_size = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        feature_size = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        feature_size = model.classifier[2].in_features
        model.classifier = nn.Identity()
    elif name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        feature_size = model.classifier[1].in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    return model, feature_size


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
        self.backbone, feature_size = _build_backbone(
            backbone,
            resolve_pretrained(pretrained, context=f"cnn:{backbone}"),
        )
        hidden_size = max(64, feature_size // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.mean(dim=(-2, -1)) if features.dim() == 4 else features.flatten(start_dim=1)
        return self.classifier(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.inference_mode():
            return torch.softmax(self(x), dim=1)


def _build_base_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_image_as_tensor(
    source: str | Path | np.ndarray,
    image_size: int = 320,
    *,
    assume_bgr: bool = True,
) -> torch.Tensor:
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            source = np.stack([source] * 3, axis=-1)
        elif source.ndim == 3 and source.shape[-1] == 3 and assume_bgr:
            source = source[:, :, ::-1]
        image = Image.fromarray(source.astype(np.uint8), mode="RGB")
    else:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    return _build_base_transform(image_size)(image)


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
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.temperature = float(temperature)
        self.model.to(self.device).eval()
        self._use_fp16 = self.device.type == "cuda"
        if self._use_fp16:
            self.model.half()

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
            tensor = _load_image_as_tensor(source, assume_bgr=True).unsqueeze(0).to(self.device)
            if self._use_fp16:
                tensor = tensor.half()
            probs = self.model.predict_proba(tensor).cpu().numpy()[0]
        probs = self._apply_temperature(probs)
        ranked = np.argsort(-probs)[: max(1, top_k)]
        return [
            {
                "label": self.class_labels[index],
                "confidence": float(probs[index]),
                "probabilities": {label: float(probs[i]) for i, label in enumerate(self.class_labels)},
            }
            for index in ranked
        ]

    def explain(
        self,
        source: str | Path | np.ndarray,
        *,
        top_k: int = 1,
        tta: bool = False,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        try:
            from medical.explainability import MedicalGradCAMExplainer

            explainer = MedicalGradCAMExplainer(self, image_size=320, device=self.device)
            return [
                {"label": item.label, "confidence": item.confidence, "heatmap": item.heatmap, "overlay": item.overlay}
                for item in explainer.explain(source, top_k=top_k, tta=tta, alpha=alpha)
            ]
        except Exception:
            logger.warning("Không thể tạo Grad-CAM cho ảnh này.", exc_info=True)
            return []

    def _predict_with_tta(self, source: str | Path | np.ndarray) -> np.ndarray:
        tensor = _load_image_as_tensor(source, assume_bgr=True).unsqueeze(0).to(self.device)
        operations = (
            lambda x: x,
            lambda x: torch.flip(x, dims=[3]),
            lambda x: torch.flip(x, dims=[2]),
            lambda x: torch.roll(x, shifts=10, dims=[3]),
            lambda x: torch.roll(x, shifts=10, dims=[2]),
        )
        predictions = []
        for operation in operations:
            augmented = operation(tensor)
            if self._use_fp16:
                augmented = augmented.half()
            predictions.append(self.model.predict_proba(augmented).cpu().numpy()[0])
        return np.mean(predictions, axis=0)

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        if self.temperature == 1.0:
            return probs
        logits = np.log(np.clip(probs, 1e-8, 1.0)) / self.temperature
        values = np.exp(logits - np.max(logits))
        return values / np.sum(values)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> MedicalCNNClassifierWrapper:
        logger.info("Đang nạp model suy luận từ %s", path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = MedicalCNNClassifier(
            num_classes=checkpoint["num_classes"],
            backbone=checkpoint.get("backbone", "resnet50"),
            pretrained=False,
            dropout=checkpoint.get("dropout", 0.3),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(
            model,
            tuple(checkpoint["class_labels"]),
            device=device,
            temperature=checkpoint.get("temperature", 1.0),
        )


def load_cnn_classifier(path: str | Path, device: str | None = None) -> MedicalCNNClassifierWrapper:
    return MedicalCNNClassifierWrapper.load(path, device=device)


def is_cnn_classifier_path(path: str | Path) -> bool:
    source = Path(path)
    if not source.is_file() or source.suffix != ".pt":
        return False
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return isinstance(checkpoint, dict) and {"model_state_dict", "class_labels"}.issubset(checkpoint)


__all__ = ["MedicalCNNClassifier", "MedicalCNNClassifierWrapper", "is_cnn_classifier_path", "load_cnn_classifier"]
