"""Multi-task learning models for medical image analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MultiTaskOutput:
    cancer_logits: torch.Tensor
    modality_logits: torch.Tensor
    anatomical_logits: torch.Tensor
    quality_logits: torch.Tensor
    tumor_presence_logits: torch.Tensor
    cancer_probs: torch.Tensor | None = None
    modality_probs: torch.Tensor | None = None
    anatomical_probs: torch.Tensor | None = None
    quality_probs: torch.Tensor | None = None
    tumor_presence_probs: torch.Tensor | None = None

    def compute_probs(self) -> "MultiTaskOutput":
        return MultiTaskOutput(
            cancer_logits=self.cancer_logits,
            modality_logits=self.modality_logits,
            anatomical_logits=self.anatomical_logits,
            quality_logits=self.quality_logits,
            tumor_presence_logits=self.tumor_presence_logits,
            cancer_probs=F.softmax(self.cancer_logits, dim=1),
            modality_probs=F.softmax(self.modality_logits, dim=1),
            anatomical_probs=F.softmax(self.anatomical_logits, dim=1),
            quality_probs=F.softmax(self.quality_logits, dim=1),
            tumor_presence_probs=torch.sigmoid(self.tumor_presence_logits),
        )


class MultiTaskMedicalModel(nn.Module):
    def __init__(
        self,
        num_cancer_classes: int = 7,
        num_modality_classes: int = 8,
        num_anatomical_classes: int = 7,
        num_quality_classes: int = 3,
        backbone: str = "efficientnet_b2",
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        from medical.cnn_classifier import _get_backbone_builder
        from medical.network_policy import resolve_pretrained

        builder, feature_dim = _get_backbone_builder(backbone)
        self.backbone = builder(backbone, resolve_pretrained(pretrained, context=f"multitask:{backbone}"))
        self.backbone_name = backbone
        if isinstance(self.backbone, tuple):
            self.backbone = self.backbone[0]
        self.shared_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.cancer_head = nn.Linear(hidden_dim, num_cancer_classes)
        self.modality_head = nn.Linear(hidden_dim, num_modality_classes)
        self.anatomical_head = nn.Linear(hidden_dim, num_anatomical_classes)
        self.quality_head = nn.Linear(hidden_dim, num_quality_classes)
        self.tumor_presence_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> MultiTaskOutput:
        features = self.backbone(x)
        shared = self.shared_head(features)
        return MultiTaskOutput(
            cancer_logits=self.cancer_head(shared),
            modality_logits=self.modality_head(shared),
            anatomical_logits=self.anatomical_head(shared),
            quality_logits=self.quality_head(shared),
            tumor_presence_logits=self.tumor_presence_head(shared),
        ).compute_probs()

    def predict(self, x: torch.Tensor) -> dict[str, Any]:
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        return {
            "cancer": output.cancer_probs,
            "modality": output.modality_probs,
            "anatomical": output.anatomical_probs,
            "quality": output.quality_probs,
            "tumor_presence": output.tumor_presence_probs,
        }


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        cancer_weight: float = 1.0,
        modality_weight: float = 0.5,
        anatomical_weight: float = 0.5,
        quality_weight: float = 0.3,
        tumor_presence_weight: float = 0.8,
    ) -> None:
        super().__init__()
        self.cancer_weight = float(cancer_weight)
        self.modality_weight = float(modality_weight)
        self.anatomical_weight = float(anatomical_weight)
        self.quality_weight = float(quality_weight)
        self.tumor_presence_weight = float(tumor_presence_weight)

    def forward(
        self,
        output: MultiTaskOutput,
        cancer_targets: torch.Tensor,
        modality_targets: torch.Tensor,
        anatomical_targets: torch.Tensor,
        quality_targets: torch.Tensor,
        tumor_presence_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cancer_loss = F.cross_entropy(output.cancer_logits, cancer_targets)
        modality_loss = F.cross_entropy(output.modality_logits, modality_targets)
        anatomical_loss = F.cross_entropy(output.anatomical_logits, anatomical_targets)
        quality_loss = F.cross_entropy(output.quality_logits, quality_targets)
        tp = output.tumor_presence_logits.squeeze(-1)
        tumor_presence_loss = F.binary_cross_entropy_with_logits(tp, tumor_presence_targets.float())
        total = (
            self.cancer_weight * cancer_loss
            + self.modality_weight * modality_loss
            + self.anatomical_weight * anatomical_loss
            + self.quality_weight * quality_loss
            + self.tumor_presence_weight * tumor_presence_loss
        )
        loss_dict = {
            "cancer": float(cancer_loss.item()),
            "modality": float(modality_loss.item()),
            "anatomical": float(anatomical_loss.item()),
            "quality": float(quality_loss.item()),
            "tumor_presence": float(tumor_presence_loss.item()),
            "total": float(total.item()),
        }
        return total, loss_dict


__all__ = ["MultiTaskMedicalModel", "MultiTaskOutput", "MultiTaskLoss"]
