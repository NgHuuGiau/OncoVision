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
    if name.startswith("timm:"):
        import timm
        timm_name = name.split("timm:", 1)[1]
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        feature_size = model.num_features
        model.reset_classifier(0)
    elif name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        feature_size = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        feature_size = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "resnet34":
        from torchvision.models import ResNet34_Weights, resnet34

        model = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
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
    elif name.startswith("vit"):
        import timm
        # Support vit_base_patch16_224, vit_large_patch16_224, etc.
        timm_name = name if name.startswith("timm:") else f"timm:{name}"
        model = timm.create_model(timm_name.replace("timm:", ""), pretrained=pretrained, num_classes=0)
        feature_size = model.num_features
        model.reset_classifier(0)
    elif name.startswith("swin"):
        import timm
        timm_name = name if name.startswith("timm:") else f"timm:{name}"
        model = timm.create_model(timm_name.replace("timm:", ""), pretrained=pretrained, num_classes=0)
        feature_size = model.num_features
        model.reset_classifier(0)
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
        head: str = "mlp",
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.dropout = dropout
        self.head = head
        self.backbone, feature_size = _build_backbone(
            backbone,
            resolve_pretrained(pretrained, context=f"cnn:{backbone}"),
        )
        if head == "linear":
            # Head ngoài (checkpoint bên thứ ba) chỉ có 1 lớp Linear trên đặc trưng backbone.
            self.classifier = nn.Linear(feature_size, num_classes)
        else:
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


def _build_base_transform(image_size: int, interpolation: str = "bilinear") -> transforms.Compose:
    mode = InterpolationMode.BICUBIC if interpolation == "bicubic" else InterpolationMode.BILINEAR
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=mode),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_image_as_tensor(
    source: str | Path | np.ndarray,
    image_size: int = 320,
    *,
    assume_bgr: bool = True,
    interpolation: str = "bilinear",
) -> torch.Tensor:
    image: Image.Image
    if isinstance(source, np.ndarray):
        image_array: np.ndarray = source
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.ndim == 3 and image_array.shape[-1] == 3 and assume_bgr:
            image_array = image_array[:, :, ::-1]
        image = Image.fromarray(image_array.astype(np.uint8), mode="RGB")
    else:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    return _build_base_transform(image_size, interpolation)(image)


class MedicalCNNClassifierWrapper:
    def __init__(
        self,
        model: MedicalCNNClassifier,
        class_labels: tuple[str, ...],
        device: str | None = None,
        temperature: float = 1.0,
        image_size: int = 320,
        interpolation: str = "bilinear",
    ) -> None:
        self.model = model
        self.class_labels = class_labels
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.temperature = float(temperature)
        self.image_size = int(image_size)
        self.interpolation = interpolation
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
            tensor = _load_image_as_tensor(
                source, image_size=self.image_size, assume_bgr=True, interpolation=self.interpolation
            ).unsqueeze(0).to(self.device)
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

            explainer = MedicalGradCAMExplainer(
                self, image_size=self.image_size, device=self.device, interpolation=self.interpolation
            )
            return [
                {"label": item.label, "confidence": item.confidence, "heatmap": item.heatmap, "overlay": item.overlay}
                for item in explainer.explain(source, top_k=top_k, tta=tta, alpha=alpha)
            ]
        except Exception:
            logger.warning("Không thể tạo Grad-CAM cho ảnh này.", exc_info=True)
            return []

    def _predict_with_tta(self, source: str | Path | np.ndarray) -> np.ndarray:
        tensor = _load_image_as_tensor(
            source, image_size=self.image_size, assume_bgr=True, interpolation=self.interpolation
        ).unsqueeze(0).to(self.device)
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
        logger.info("Dang nap model suy luan tu %s", path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Format 1: OncoVision native format
            if "model_state_dict" in checkpoint and "class_labels" in checkpoint and "num_classes" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                class_labels = checkpoint["class_labels"]
                num_classes = checkpoint["num_classes"]
                backbone = checkpoint.get("backbone", "resnet50")
                dropout = checkpoint.get("dropout", 0.3)
                head = checkpoint.get("head", "mlp")
                image_size = int(checkpoint.get("image_size", 320))
                interpolation = checkpoint.get("interpolation", "bilinear")
                temperature = checkpoint.get("temperature", 1.0)

            # Format 2: External format with state_dict + class_names
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                class_labels = checkpoint.get("class_labels") or checkpoint.get("class_names")
                if class_labels is None:
                    raise ValueError("Checkpoint thieu class_labels/class_names")
                num_classes = len(class_labels)
                # Infer backbone from state_dict keys
                backbone = cls._infer_backbone(list(state_dict.keys()))
                dropout = checkpoint.get("dropout", 0.3)
                head = checkpoint.get("head", "mlp")
                image_size = int(checkpoint.get("image_size", 320))
                interpolation = checkpoint.get("interpolation", "bilinear")
                temperature = checkpoint.get("temperature", 1.0)

                # Fix key prefix: external models use features.x.x, we need backbone.features.x.x
                if not any(k.startswith("backbone.") for k in state_dict.keys()):
                    logger.info("Adding backbone. prefix to state_dict keys")
                    state_dict = {f"backbone.{k}": v for k, v in state_dict.items()}

            # Format 1 (incomplete): có model_state_dict + class_labels nhưng thiếu num_classes
            elif "model_state_dict" in checkpoint and "class_labels" in checkpoint:
                raise ValueError("Checkpoint Format 1 thiếu num_classes")
            # Format 2: External format with state_dict + class_names
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                class_labels = checkpoint.get("class_labels") or checkpoint.get("class_names")
                if class_labels is None:
                    raise ValueError("Checkpoint thieu class_labels/class_names")
                num_classes = len(class_labels)
                # Infer backbone from state_dict keys
                backbone = cls._infer_backbone(list(state_dict.keys()))
                dropout = checkpoint.get("dropout", 0.3)
                head = checkpoint.get("head", "mlp")
                image_size = int(checkpoint.get("image_size", 320))
                interpolation = checkpoint.get("interpolation", "bilinear")
                temperature = checkpoint.get("temperature", 1.0)

                # Fix key prefix: external models use features.x.x, we need backbone.features.x.x
                if not any(k.startswith("backbone.") for k in state_dict.keys()):
                    logger.info("Adding backbone. prefix to state_dict keys")
                    state_dict = {f"backbone.{k}": v for k, v in state_dict.items()}

            # Format 3: Raw state_dict (no wrapper)
            else:
                state_dict = checkpoint
                # Try to infer class_labels from checkpoint or use default
                class_labels = checkpoint.get("class_labels") or checkpoint.get("class_names")
                if class_labels is None:
                    raise ValueError("Checkpoint thieu class_labels/class_names, khong the tu dinh")
                num_classes = len(class_labels)
                backbone = cls._infer_backbone(list(state_dict.keys()))
                dropout = 0.3
                head = "mlp"
                image_size = 320
                interpolation = "bilinear"
                temperature = 1.0

                if not any(k.startswith("backbone.") for k in state_dict.keys()):
                    logger.info("Adding backbone. prefix to state_dict keys")
                    state_dict = {f"backbone.{k}": v for k, v in state_dict.items()}

        else:
            raise ValueError("Checkpoint khong hop le: khong phai dict")

        # Build model
        model = MedicalCNNClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False,
            dropout=dropout,
            head=head,
        )
        model.load_state_dict(state_dict, strict=False)  # strict=False to ignore missing classifier head keys

        if "image_size" not in (checkpoint if isinstance(checkpoint, dict) else {}):
            logger.warning(
                "Checkpoint %s khong khai bao image_size; mac dinh resize 320px — "
                "sai do phan giai train se lam giam chat luong phan loai.",
                path,
            )

        return cls(
            model,
            tuple(class_labels),
            device=device,
            temperature=temperature,
            image_size=image_size,
            interpolation=interpolation,
        )

    @staticmethod
    def _infer_backbone(keys: list[str]) -> str:
        """Infer backbone architecture from state_dict keys."""
        keys_str = " ".join(keys)
        if "backbone.features" in keys_str or "features.0" in keys_str or "features.1" in keys_str:
            if "classifier.1.weight" in " ".join(keys):
                return "efficientnet_b0"
            return "efficientnet_b0"  # default for features.x pattern
        if "backbone.layer" in " ".join(keys) or "fc.weight" in " ".join(keys):
            return "resnet50"
        if "backbone.stages" in " ".join(keys) or "classifier.2.weight" in " ".join(keys):
            return "convnext_tiny"
        if "backbone.patch_embed" in " ".join(keys) or "blocks." in " ".join(keys):
            return "vit"
        if "encoder." in " ".join(keys) or "decoder." in " ".join(keys):
            return "unet"
        # Default fallback based on common patterns
        if any("features." in k for k in keys):
            return "efficientnet_b0"
        return "resnet50"


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
    return isinstance(checkpoint, dict) and {"model_state_dict", "class_labels", "num_classes"}.issubset(checkpoint)


__all__ = ["MedicalCNNClassifier", "MedicalCNNClassifierWrapper", "is_cnn_classifier_path", "load_cnn_classifier"]
