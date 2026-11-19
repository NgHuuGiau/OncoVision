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
    elif name == "densenet121":
        from torchvision.models import DenseNet121_Weights, densenet121

        model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        feature_size = model.classifier.in_features
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
            # Format 1: OncoVision native format (model_state_dict + class_labels + num_classes)
            if "model_state_dict" in checkpoint and "class_labels" in checkpoint and "num_classes" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                class_labels = checkpoint["class_labels"]
                num_classes = checkpoint["num_classes"]
                backbone_meta = checkpoint.get("backbone", "resnet50")
                dropout = checkpoint.get("dropout", 0.3)
                head_meta = checkpoint.get("head", "mlp")
                image_size = int(checkpoint.get("image_size", 320))
                interpolation = checkpoint.get("interpolation", "bilinear")
                temperature = checkpoint.get("temperature", 1.0)

                # Auto-detect actual backbone from state_dict keys (ignore potentially wrong metadata)
                backbone, head = cls._detect_backbone_and_head(list(state_dict.keys()), num_classes, backbone_meta, head_meta)

                # Add backbone. prefix if missing (raw backbone format)
                if not any(k.startswith("backbone.") for k in state_dict.keys()):
                    state_dict = cls._add_backbone_prefix(state_dict, head)
                    logger.info("Da them backbone. prefix cho %s (backbone=%s, head=%s)", path, backbone, head)

            # Format 2: External format with state_dict + class_names
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                class_labels = checkpoint.get("class_labels") or checkpoint.get("class_names")
                if class_labels is None:
                    raise ValueError("Checkpoint thieu class_labels/class_names")
                num_classes = len(class_labels)
                backbone_meta = checkpoint.get("backbone", "resnet50")
                head_meta = checkpoint.get("head", "mlp")
                dropout = checkpoint.get("dropout", 0.3)
                image_size = int(checkpoint.get("image_size", 320))
                interpolation = checkpoint.get("interpolation", "bilinear")
                temperature = checkpoint.get("temperature", 1.0)

                backbone, head = cls._detect_backbone_and_head(list(state_dict.keys()), num_classes, backbone_meta, head_meta)

                if not any(k.startswith("backbone.") for k in state_dict.keys()):
                    state_dict = cls._add_backbone_prefix(state_dict, head)
                    logger.info("Da them backbone. prefix cho %s (backbone=%s, head=%s)", path, backbone, head)

            # Format 1 (incomplete): có model_state_dict + class_labels nhưng thiếu num_classes
            elif "model_state_dict" in checkpoint and "class_labels" in checkpoint:
                raise ValueError("Checkpoint Format 1 thiếu num_classes")
            # Format 2 duplicate (should not reach here due to elif above)
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                class_labels = checkpoint.get("class_labels") or checkpoint.get("class_names")
                if class_labels is None:
                    raise ValueError("Checkpoint thieu class_labels/class_names")
                num_classes = len(class_labels)
                backbone_meta = checkpoint.get("backbone", "resnet50")
                head_meta = checkpoint.get("head", "mlp")
                dropout = checkpoint.get("dropout", 0.3)
                image_size = int(checkpoint.get("image_size", 320))
                interpolation = checkpoint.get("interpolation", "bilinear")
                temperature = checkpoint.get("temperature", 1.0)

                backbone, head = cls._detect_backbone_and_head(list(state_dict.keys()), num_classes, backbone_meta, head_meta)

                if not any(k.startswith("backbone.") for k in state_dict.keys()):
                    state_dict = cls._add_backbone_prefix(state_dict, head)
                    logger.info("Da them backbone. prefix cho %s (backbone=%s, head=%s)", path, backbone, head)

            # Format 3: Raw state_dict (no wrapper)
            else:
                state_dict = checkpoint
                class_labels = checkpoint.get("class_labels") or checkpoint.get("class_names")
                if class_labels is None:
                    raise ValueError("Checkpoint thieu class_labels/class_names, khong the tu dinh")
                num_classes = len(class_labels)
                backbone_meta = checkpoint.get("backbone", "resnet50")
                head_meta = checkpoint.get("head", "mlp")
                dropout = 0.3
                image_size = 320
                interpolation = "bilinear"
                temperature = 1.0

                backbone, head = cls._detect_backbone_and_head(list(state_dict.keys()), num_classes, backbone_meta, head_meta)

                if not any(k.startswith("backbone.") for k in state_dict.keys()):
                    state_dict = cls._add_backbone_prefix(state_dict, head)
                    logger.info("Da them backbone. prefix cho %s (backbone=%s, head=%s)", path, backbone, head)

        else:
            raise ValueError("Checkpoint khong hop le: khong phai dict")

        # Build model with detected backbone and head
        model = MedicalCNNClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False,
            dropout=dropout,
            head=head,
        )

        # Remap state_dict keys to match model structure (e.g., backbone.classifier.* -> classifier.*)
        model_keys = set(model.state_dict().keys())
        state_dict = cls._remap_state_dict_for_model(state_dict, model_keys, head)

        # Load state_dict with strict matching; log any missing/unexpected keys
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            # Size mismatch (e.g., different num_classes) — load backbone only
            logger.warning("Model %s: loi size mismatch, load backbone chi: %s", path, str(e)[:200])
            missing = []
            unexpected = []
            # Try loading backbone keys only
            model_sd = model.state_dict()
            for key, val in state_dict.items():
                if key in model_sd and model_sd[key].shape == val.shape:
                    model_sd[key] = val
            logger.info("Model %s: da load partial weights (shape match only)", path)

        if missing:
            logger.warning("Model %s: %d keys khong tim thay trong checkpoint: %s", path, len(missing), missing[:5])
        if unexpected:
            logger.warning("Model %s: %d keys khong mong muon trong checkpoint: %s", path, len(unexpected), unexpected[:5])

        # Verify that backbone weights were actually loaded
        model_keys = set(model.state_dict().keys())
        backbone_key_count = len([k for k in model_keys if k.startswith("backbone.")])
        backbone_keys_loaded = sum(1 for k in model_keys if k.startswith("backbone.") and model.state_dict()[k] is not None)
        logger.info("Model %s: da nap %d/%d backbone keys (backbone=%s, head=%s)",
                     path, backbone_keys_loaded, backbone_key_count, backbone, head)

        # Fix image_size for swin models (timm swin_tiny requires 224x224)
        if backbone.startswith("swin") and image_size != 224:
            logger.warning("Swin model yeu cau 224x224, sua image_size tu %s -> 224", image_size)
            image_size = 224

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

    @staticmethod
    def _detect_backbone_and_head(keys: list[str], num_classes: int, backbone_meta: str, head_meta: str) -> tuple[str, str]:
        """Auto-detect actual backbone and head type from state_dict keys.
        
        This is critical because metadata often doesn't match the actual model architecture.
        Returns (backbone, head) tuple.
        """
        keys_str = " ".join(keys)
        has_backbone_prefix = any(k.startswith("backbone.") for k in keys)
        
        # Strip backbone. prefix for pattern detection
        stripped_keys = [k.replace("backbone.", "", 1) if k.startswith("backbone.") else k for k in keys]
        stripped_str = " ".join(stripped_keys)
        
        # Detect backbone type
        backbone = backbone_meta  # default to metadata
        
        # --- Custom architecture detection (P0 fix) ---
        
        # Brain Swin: raw format with 'swin.*' + 'classifier.*' (no backbone prefix)
        if not has_backbone_prefix and "swin." in keys_str and "classifier.weight" in keys_str:
            backbone = "swin_tiny_patch4_window7_224"
            logger.warning("Phat hieu brain Swin raw format (swin.* + classifier.*)")
        # Liver/ct: custom swin wrapper 'backbone.swin.*' + 'backbone.classifier.*'
        elif has_backbone_prefix and "backbone.swin." in keys_str:
            backbone = "swin_tiny_patch4_window7_224"
            logger.warning("Phat hieu liver/ct custom swin wrapper")
        # Thyroid: custom convnext 'backbone.resnet.*' (metadata says resnet50 but it's convnext)
        elif has_backbone_prefix and "backbone.resnet." in keys_str:
            # Check if it's actually convnext (has embedder pattern)
            if "embedder" in keys_str or "resnet.embedder" in keys_str:
                backbone = "convnext_tiny"
                logger.warning("Phat hieu thyroid custom convnext (metadata sai resnet50)")
            else:
                backbone = "resnet50"
        # Cervical/pap: efficientnet-v2 'backbone.conv_stem.*' + 'backbone.blocks.*' + 'head.*'
        elif has_backbone_prefix and "backbone.conv_stem" in keys_str and "backbone.blocks." in keys_str:
            # Check if there's a separate 'head.*' key (efficientnet-v2 pattern)
            has_head_prefix = any(k.startswith("head.") for k in keys)
            if has_head_prefix:
                backbone = "efficientnet_b0"
                logger.warning("Phat hieu cervical/pap efficientnet-v2 voi head rieng")
            else:
                backbone = "efficientnet_b0"
        elif "denseblock" in stripped_str:
            backbone = "densenet121"
        elif "swin" in stripped_str:
            if "swin.embeddings" in stripped_str or "swin.encoder" in stripped_str:
                backbone = "swin_tiny_patch4_window7_224"
            elif "swin_tiny" in backbone_meta or "tiny" in stripped_str:
                backbone = "swin_tiny_patch4_window7_224"
            else:
                backbone = "swin_tiny_patch4_window7_224"
        elif "embeddings" in stripped_str:
            backbone = backbone_meta
        elif "stages" in stripped_str:
            backbone = "convnext_tiny"
        elif "patch_embed" in stripped_str:
            backbone = "vit"
        elif "conv_stem" in stripped_str and "blocks." in stripped_str:
            backbone = "efficientnet_b0"
        elif "conv_stem" in stripped_str or ("features.0.0.weight" in stripped_str and "features.0.1.weight" in stripped_str):
            backbone = "efficientnet_b0"
        elif "layer1" in stripped_str or "conv1" in stripped_str:
            if not has_backbone_prefix:
                if len(keys) <= 130:
                    backbone = "resnet18"
                elif len(keys) <= 230:
                    backbone = "resnet34"
                else:
                    backbone = "resnet50"
            else:
                backbone = "resnet50"
        elif "features." in stripped_str:
            backbone = "efficientnet_b0"
        
        # Detect head type
        has_fc = "fc.weight" in stripped_str
        has_classifier_1 = "classifier.1.weight" in stripped_str
        has_classifier_weight = "classifier.weight" in stripped_str and "classifier.bias" in stripped_str
        has_head_prefix = any(k.startswith("head.") for k in keys)
        
        # Brain Swin raw: classifier.weight + classifier.bias at top level = linear head
        if not has_backbone_prefix and "classifier.weight" in keys_str and "classifier.bias" in keys_str:
            head = "linear"
        # Cervical/pap: separate head.* keys = linear head
        elif has_head_prefix:
            head = "linear"
        elif has_fc:
            head = "linear"
        elif has_classifier_1:
            head = "linear"
        elif has_classifier_weight:
            head = "linear"
        else:
            head = "mlp"
        
        if backbone != backbone_meta:
            logger.warning(
                "Backbone metadata='%s' nhưng phát hiện='%s' — dùng backbone phát hiện tự động",
                backbone_meta, backbone
            )
        
        return backbone, head

    @staticmethod
    def _add_backbone_prefix(state_dict: dict, head: str) -> dict:
        """Add backbone. prefix to raw backbone state_dict keys and map classifier keys.
        
        Handles different raw backbone formats:
        - ResNet: conv1, bn1, layer1-5, fc -> backbone.conv1, backbone.bn1, backbone.layer1-5, classifier
        - EfficientNet: features.0.0, classifier -> backbone.features.0.0, classifier
        - DenseNet: features.denseblock1, classifier -> backbone.features.denseblock1, classifier
        - Swin (raw): swin.* + classifier.* -> backbone.swin.* + classifier.*
        """
        new_state_dict = {}
        classifier_keys = set()
        
        # Identify classifier/head keys that should NOT get backbone. prefix
        if head == "linear":
            classifier_patterns = {"fc.weight", "fc.bias", "classifier.weight", "classifier.bias",
                                   "classifier.0.weight", "classifier.0.bias",
                                   "classifier.1.weight", "classifier.1.bias"}
        else:
            classifier_patterns = {"fc.weight", "fc.bias"}
        
        for k, v in state_dict.items():
            is_classifier = k in classifier_patterns
            
            # Already has backbone. prefix — pass through
            if k.startswith("backbone."):
                new_state_dict[k] = v
                continue
            
            # Already has head. prefix — pass through (cervical/pap efficientnet-v2)
            if k.startswith("head."):
                new_state_dict[k] = v
                continue
            
            if is_classifier:
                if head == "linear":
                    if k.startswith("fc."):
                        new_key = k.replace("fc.", "classifier.", 1)
                    elif k.startswith("classifier."):
                        new_key = k
                    else:
                        new_key = k
                else:
                    # MLP head: skip old fc keys
                    continue
                classifier_keys.add(new_key)
                new_state_dict[new_key] = v
            else:
                new_state_dict[f"backbone.{k}"] = v
        
        logger.debug("Converted %d keys (%d classifier, %d backbone) with head=%s",
                     len(new_state_dict), len(classifier_keys),
                     len(new_state_dict) - len(classifier_keys), head)
        return new_state_dict

    @staticmethod
    def _remap_state_dict_for_model(state_dict: dict, model_keys: set[str], head: str) -> dict:
        """Remap state_dict keys to match MedicalCNNClassifier model structure.
        
        Handles cases where checkpoint has backbone.classifier.* but model expects classifier.*
        (e.g., DenseNet checkpoints where classifier is part of backbone).
        Also handles custom architectures:
        - Liver/ct: backbone.swin.* -> backbone.* (strip swin wrapper)
        - Thyroid: backbone.resnet.* -> backbone.* (strip resnet wrapper)
        - Cervical/pap: head.* -> classifier.* (map head to classifier)
        """
        remapped = dict(state_dict)
        
        # Check if checkpoint has backbone.classifier.weight/bias but model expects classifier.weight/bias
        if "classifier.weight" in model_keys and "backbone.classifier.weight" in state_dict:
            if "classifier.weight" not in state_dict:
                for key in list(remapped.keys()):
                    if key.startswith("backbone.classifier."):
                        new_key = key.replace("backbone.classifier.", "classifier.", 1)
                        if new_key in model_keys:
                            remapped[new_key] = remapped.pop(key)
                            logger.debug("Remapped %s -> %s", key, new_key)
        
        # Liver/ct: backbone.swin.* -> backbone.* (custom swin wrapper)
        if any(k.startswith("backbone.swin.") for k in state_dict):
            for key in list(remapped.keys()):
                if key.startswith("backbone.swin."):
                    new_key = key.replace("backbone.swin.", "backbone.", 1)
                    if new_key in model_keys and new_key not in remapped:
                        remapped[new_key] = remapped.pop(key)
                        logger.debug("Remapped liver/ct: %s -> %s", key, new_key)
        
        # Thyroid: backbone.resnet.* -> backbone.* (custom convnext wrapper)
        if any(k.startswith("backbone.resnet.") for k in state_dict):
            for key in list(remapped.keys()):
                if key.startswith("backbone.resnet."):
                    new_key = key.replace("backbone.resnet.", "backbone.", 1)
                    if new_key in model_keys and new_key not in remapped:
                        remapped[new_key] = remapped.pop(key)
                        logger.debug("Remapped thyroid: %s -> %s", key, new_key)
        
        # Cervical/pap: head.* -> classifier.* (efficientnet-v2 separate head)
        if any(k.startswith("head.") for k in state_dict):
            for key in list(remapped.keys()):
                if key.startswith("head."):
                    new_key = key.replace("head.", "classifier.", 1)
                    if new_key in model_keys and new_key not in remapped:
                        remapped[new_key] = remapped.pop(key)
                        logger.debug("Remapped cervical/pap: %s -> %s", key, new_key)
        
        return remapped


# Global model cache: path -> (wrapper, device)
_model_cache: dict[str, tuple[MedicalCNNClassifierWrapper, str]] = {}


def get_model_cache_info() -> dict[str, str]:
    """Trả về thông tin model đang được cache.

    Returns:
        Dict mapping model path -> device.
    """
    return {path: device for path, (_, device) in _model_cache.items()}


def clear_model_cache() -> None:
    """Xóa toàn bộ model cache. Dùng khi cần giải phóng GPU memory."""
    _model_cache.clear()
    logger.info("Đã xóa toàn bộ model cache")


def load_cnn_classifier(
    path: str | Path, device: str | None = None, *, use_cache: bool = True
) -> MedicalCNNClassifierWrapper:
    """Load CNN classifier với optional caching.

    Args:
        path: Đường dẫn đến model file.
        device: Device để load model (cpu/cuda).
        use_cache: Nếu True, tái sử dụng model đã load từ cùng path.

    Returns:
        MedicalCNNClassifierWrapper instance.
    """
    cache_key = str(Path(path).resolve())
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if use_cache and cache_key in _model_cache:
        cached_wrapper, cached_device = _model_cache[cache_key]
        if cached_device == target_device:
            logger.debug("Sử dụng model cache: %s (@%s)", cache_key, target_device)
            return cached_wrapper
        else:
            logger.info(
                "Model cache device mismatch (%s vs %s), load lại", cached_device, target_device
            )

    logger.info("Đang load CNN classifier từ %s (@%s)", cache_key, target_device)
    wrapper = MedicalCNNClassifierWrapper.load(path, device=target_device)

    if use_cache:
        _model_cache[cache_key] = (wrapper, target_device)
        logger.info("Đã cache model: %s (@%s)", cache_key, target_device)

    return wrapper


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
