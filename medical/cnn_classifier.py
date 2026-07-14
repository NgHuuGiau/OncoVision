from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import (
    convnext_tiny,
    efficientnet_b0,
    efficientnet_b2,
    efficientnet_b3,
    resnet18,
    resnet50,
)
from torchvision.transforms import InterpolationMode


def _build_medical_augmentations(image_size: int = 320) -> transforms.Compose:
    return transforms.Compose([
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
    ])


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
    ) -> None:
        self.samples = samples
        self.class_labels = class_labels
        self.num_classes = len(class_labels)

        if augment:
            self.transform = _build_medical_augmentations(image_size=image_size)
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

        if backbone == "resnet18":
            self.backbone = resnet18(weights="DEFAULT" if pretrained else None)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet50":
            self.backbone = resnet50(weights="DEFAULT" if pretrained else None)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            self.backbone = efficientnet_b0(weights="DEFAULT" if pretrained else None)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "efficientnet_b2":
            self.backbone = efficientnet_b2(weights="DEFAULT" if pretrained else None)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "efficientnet_b3":
            self.backbone = efficientnet_b3(weights="DEFAULT" if pretrained else None)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "convnext_tiny":
            self.backbone = convnext_tiny(weights="DEFAULT" if pretrained else None)
            feature_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                "Choose from: resnet18, resnet50, efficientnet_b0, efficientnet_b2, efficientnet_b3, convnext_tiny"
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, max(64, feature_dim // 2)),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(max(64, feature_dim // 2), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
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
            image_tensor = _load_image_as_tensor(source)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            probs = self.model.predict_proba(image_tensor).cpu().numpy()[0]

        probs = self._apply_temperature(probs)
        ranked = np.argsort(-probs)
        results = []
        for idx in ranked[: max(1, top_k)]:
            results.append(
                {
                    "label": self.class_labels[idx],
                    "confidence": float(probs[idx]),
                    "probabilities": {self.class_labels[i]: float(probs[i]) for i in range(len(self.class_labels))},
                }
            )
        return results

    def _predict_with_tta(self, source: str | Path | np.ndarray) -> np.ndarray:
        base_tensor = _load_image_as_tensor(source)
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
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "class_labels": self.class_labels,
                "backbone": self.model.backbone_name,
                "num_classes": self.model.num_classes,
                "dropout": self.model.dropout,
                "temperature": self.temperature,
            },
            target,
        )
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


def _load_image_as_tensor(source: str | Path | np.ndarray, image_size: int = 320) -> torch.Tensor:
    transform = _build_base_transform(image_size=image_size)
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            source = np.stack([source] * 3, axis=-1)
        elif source.ndim == 3 and source.shape[-1] == 3:
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
    """Estimate a safe batch size based on available GPU memory.

    Uses rough per-image memory footprints for common backbones under mixed
    precision. Falls back to a conservative CPU default when no GPU is
    available or the memory query fails.
    """
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
    if backbone_lower in {"resnet50", "convnext_tiny"}:
        per_image = 200.0
    elif backbone_lower in {"efficientnet_b0", "efficientnet_b2", "efficientnet_b3"}:
        per_image = 150.0
    else:
        per_image = 180.0

    estimated = int(memory_mb / per_image)
    safe = max(4, min(64, estimated))
    # Round down to nearest power of 2.
    power = 1
    while power * 2 <= safe:
        power *= 2
    return max(4, power)


def get_recommended_batch_size(backbone: str, device: str | None = None) -> int:
    """Public helper returning the auto-selected batch size for ``backbone``."""
    return _estimate_safe_batch_size(backbone=backbone, device=device)


def _create_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def _create_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int, warmup_epochs: int = 3) -> torch.optim.lr_scheduler.LRScheduler:
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs))
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
        )
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
) -> tuple[MedicalCNNClassifierWrapper, dict[str, Any]]:
    if not samples:
        raise FileNotFoundError("Khong co du lieu train cho CNN classifier.")

    _set_deterministic_seed()
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if batch_size is None:
        batch_size = _estimate_safe_batch_size(backbone=backbone, device=str(device_obj))
    effective_batch = batch_size * max(1, int(gradient_accumulation_steps))

    train_dataset = MedicalImageDataset(samples, class_labels, image_size=image_size, augment=True)
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

    model = model.to(device_obj)

    if class_weights is not None and len(class_weights) != num_classes:
        raise ValueError("class_weights length must match the number of classes")
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device_obj)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weight_tensor)
    optimizer = _create_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
    scheduler = _create_scheduler(optimizer, num_epochs=num_epochs, warmup_epochs=warmup_epochs)
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")

    scaler = torch.amp.GradScaler(device_obj.type, enabled=mixed_precision and device_obj.type == "cuda")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "effective_batch": [float(effective_batch)],
    }

    best_model_state: dict[str, Any] | None = None
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        accumulated_steps = 0
        for images, labels in train_loader:
            images = images.to(device_obj, non_blocking=True)
            labels = labels.to(device_obj, non_blocking=True)
            with torch.amp.autocast(device_obj.type, enabled=scaler.is_enabled()):
                outputs = model(images)
                loss = criterion(outputs, labels) / max(1, gradient_accumulation_steps)
            scaler.scale(loss).backward()
            accumulated_steps += 1
            if accumulated_steps >= gradient_accumulation_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulated_steps = 0
            total_loss += loss.item() * images.size(0) * max(1, gradient_accumulation_steps)

        if accumulated_steps > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        epoch_loss = total_loss / len(train_dataset)
        history["train_loss"].append(epoch_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        val_acc = 0.0
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device_obj, non_blocking=True)
                    labels = labels.to(device_obj, non_blocking=True)
                    with torch.amp.autocast(device_obj.type, enabled=scaler.is_enabled()):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    val_loss_sum += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_loss = val_loss_sum / total
            val_acc = correct / total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {
                    "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                }

            if early_stopping(val_acc):
                if best_model_state is not None:
                    model.load_state_dict(best_model_state["model_state_dict"])
                break

    wrapper = MedicalCNNClassifierWrapper(model=model, class_labels=class_labels, device=str(device_obj))
    return wrapper, history


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
]
