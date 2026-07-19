"""Advanced loss functions for imbalanced medical classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: list[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=None)
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce_loss
        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha, dtype=torch.float32, device=inputs.device)
            alpha_t = alpha_t.gather(0, targets.clamp(min=0, max=len(self.alpha) - 1))
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        num_classes = probs.shape[1]
        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        tp = (probs * targets_onehot).sum(dim=(0, 2, 3))
        fp = (probs * (1 - targets_onehot)).sum(dim=(0, 2, 3))
        fn = ((1 - probs) * targets_onehot).sum(dim=(0, 2, 3))
        tversky_index = tp / (tp + self.alpha * fp + self.beta * fn + 1e-6)
        loss = (1.0 - tversky_index) ** self.gamma
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LDAMLoss(nn.Module):
    def __init__(
        self,
        cls_num_list: list[int],
        max_m: float = 0.5,
        s: float = 30.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not cls_num_list:
            cls_num_list = [1]
        self.cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        self.max_m = float(max_m)
        self.s = float(s)
        self.reduction = reduction
        self.m_list: torch.Tensor | None = None

    def _build_m_list(self, device: torch.device) -> torch.Tensor:
        many_cls = torch.tensor(self.cls_num_list, dtype=torch.float32, device=device)
        max_cls = torch.max(many_cls)
        m_list = self.max_m / torch.sqrt(torch.sqrt(many_cls / max_cls + 1e-6))
        m_list = torch.clamp(m_list, min=0.0, max=self.max_m)
        return m_list

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = inputs.device
        if self.m_list is None or self.m_list.device != device:
            self.m_list = self._build_m_list(device)
        index = torch.zeros_like(inputs, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), True)
        index = index.bool()
        m_list = self.m_list.unsqueeze(0).expand(inputs.size(0), -1)
        inputs_m = inputs - m_list
        inputs = torch.where(index, inputs_m, inputs)
        outputs = inputs * self.s
        loss = F.cross_entropy(outputs, targets, reduction="none")
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ASLLoss(nn.Module):
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs_pos = F.sigmoid(inputs)
        probs_neg = 1.0 - probs_pos
        targets_onehot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        pos_loss = targets_onehot * torch.log(torch.clamp(probs_pos, min=1e-8))
        neg_loss = (1.0 - targets_onehot) * torch.log(torch.clamp(probs_neg, min=1e-8))
        if self.gamma_pos > 0:
            pos_loss = pos_loss * (1.0 - probs_pos) ** self.gamma_pos
        if self.gamma_neg > 0:
            neg_loss = neg_loss * (probs_pos) ** self.gamma_neg
        loss = -(pos_loss + neg_loss)
        if self.clip > 0:
            loss = torch.clamp(loss, min=-self.clip, max=None)
        loss = loss.sum(dim=1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BalancedSoftmaxLoss(nn.Module):
    def __init__(
        self,
        cls_num_list: list[int],
        beta: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not cls_num_list:
            cls_num_list = [1]
        self.cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        self.beta = float(beta)
        self.reduction = reduction
        self.scaling_factor: torch.Tensor | None = None

    def _build_scaling_factor(self, device: torch.device) -> torch.Tensor:
        return (self.cls_num_list ** self.beta).to(device)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = inputs.device
        if self.scaling_factor is None or self.scaling_factor.device != device:
            self.scaling_factor = self._build_scaling_factor(device)
        scaled_inputs = inputs + torch.log(self.scaling_factor.unsqueeze(0) + 1e-6)
        loss = F.cross_entropy(scaled_inputs, targets, reduction="none")
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


__all__ = [
    "FocalLoss",
    "FocalTverskyLoss",
    "LDAMLoss",
    "ASLLoss",
    "BalancedSoftmaxLoss",
]
