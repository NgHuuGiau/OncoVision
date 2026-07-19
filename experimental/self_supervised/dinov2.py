"""Self-supervised pretraining for medical images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SelfSupervisedConfig:
    method: str = "dinov2"
    backbone: str = "vit_large_patch14_dinov2.lvd142m"
    image_size: int = 224
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    output_dir: str = "output/medical/self_supervised"
    device: str | None = None


class DINOv2Pretrainer:
    def __init__(self, config: SelfSupervisedConfig | None = None) -> None:
        self.config = config or SelfSupervisedConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.student_head = nn.Sequential(
            nn.Linear(self.model.embed_dim, self.model.embed_dim),
            nn.GELU(),
            nn.Linear(self.model.embed_dim, 256),
        )
        self.teacher_head = nn.Sequential(
            nn.Linear(self.model.embed_dim, self.model.embed_dim),
            nn.GELU(),
            nn.Linear(self.model.embed_dim, 256),
        )
        self.student_head.to(self.device)
        self.teacher_head.to(self.device)
        self._freeze_teacher()

    def _build_model(self) -> nn.Module:
        try:
            model = torch.hub.load("facebookresearch/dinov2", self.config.backbone, pretrained=True)
            return model
        except Exception:
            try:
                from transformers import Dinov2Model
                model = Dinov2Model.from_pretrained("facebook/dinov2-large")
                return model
            except Exception as exc:
                raise RuntimeError(f"Khong the tai DINOv2 backbone: {exc}") from exc

    def _freeze_teacher(self) -> None:
        for param in self.teacher_head.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_teacher(self, momentum: float = 0.996) -> None:
        for student_param, teacher_param in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            teacher_param.data = momentum * teacher_param.data + (1.0 - momentum) * student_param.data

    def train_step(self, images: torch.Tensor, teacher_momentum: float = 0.996) -> torch.Tensor:
        images = images.to(self.device)
        views1 = self._augment(images)
        views2 = self._augment(images)
        student_feat1 = self.model(views1).last_hidden_state.mean(dim=1)
        student_feat2 = self.model(views2).last_hidden_state.mean(dim=1)
        student_out1 = self.student_head(student_feat1)
        student_out2 = self.student_head(student_feat2)
        with torch.no_grad():
            teacher_feat1 = self.model(views1).last_hidden_state.mean(dim=1)
            teacher_feat2 = self.model(views2).last_hidden_state.mean(dim=1)
            teacher_out1 = self.teacher_head(teacher_feat1)
            teacher_out2 = self.teacher_head(teacher_feat2)
        loss = self._dino_loss(student_out1, teacher_out2) + self._dino_loss(student_out2, teacher_out1)
        self._update_teacher(teacher_momentum)
        return loss

    def _augment(self, images: torch.Tensor) -> torch.Tensor:
        augmented = images.clone()
        if torch.rand(1).item() > 0.5:
            augmented = torch.flip(augmented, dims=[3])
        if torch.rand(1).item() > 0.5:
            shift = torch.randint(-10, 11, (1,)).item()
            augmented = torch.roll(augmented, shifts=shift, dims=3)
        noise = torch.randn_like(augmented) * 0.01
        augmented = torch.clamp(augmented + noise, 0.0, 1.0)
        return augmented

    def _dino_loss(self, student_out: torch.Tensor, teacher_out: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        student_out = F.normalize(student_out, dim=-1)
        teacher_out = F.normalize(teacher_out.detach(), dim=-1)
        logits = torch.matmul(student_out, teacher_out.T) / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "student_head": self.student_head.state_dict(),
            "teacher_head": self.teacher_head.state_dict(),
            "config": self.config,
        }, target)
        return target

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "student_head" in checkpoint:
            self.student_head.load_state_dict(checkpoint["student_head"])
        if "teacher_head" in checkpoint:
            self.teacher_head.load_state_dict(checkpoint["teacher_head"])


__all__ = ["SelfSupervisedConfig", "DINOv2Pretrainer"]
