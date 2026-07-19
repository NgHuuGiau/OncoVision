"""Uncertainty quantification for medical image classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class UncertaintyResult:
    mean_probability: np.ndarray
    variance: np.ndarray
    entropy: np.ndarray
    mutual_information: np.ndarray
    confidence: float
    predicted_class: int
    predicted_label: str
    calibration_error: float | None = None


class MCDropoutUncertainty:
    def __init__(self, model: nn.Module, num_samples: int = 20, device: str | None = None) -> None:
        self.model = model
        self.num_samples = num_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._enable_dropout()

    def _enable_dropout(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout3d):
                module.train()

    @torch.no_grad()
    def predict(self, x: torch.Tensor, class_labels: list[str] | None = None) -> UncertaintyResult:
        self.model.eval()
        self._enable_dropout()
        x = x.to(self.device)
        probs_list = []
        for _ in range(self.num_samples):
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu())
        all_probs = torch.stack(probs_list, dim=0).numpy()
        mean_probs = all_probs.mean(axis=0)[0]
        variance = all_probs.var(axis=0)[0]
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
        mutual_info = entropy - np.mean([-np.sum(p * np.log(p + 1e-8)) for p in all_probs[:, 0]])
        predicted_class = int(np.argmax(mean_probs))
        confidence = float(mean_probs[predicted_class])
        label = class_labels[predicted_class] if class_labels and predicted_class < len(class_labels) else str(predicted_class)
        return UncertaintyResult(
            mean_probability=mean_probs,
            variance=variance,
            entropy=np.array([entropy]),
            mutual_information=np.array([mutual_info]),
            confidence=confidence,
            predicted_class=predicted_class,
            predicted_label=label,
        )


class DeepEnsembleUncertainty:
    def __init__(self, models: list[nn.Module], class_labels: list[str] | None = None, device: str | None = None) -> None:
        if not models:
            raise ValueError("DeepEnsembleUncertainty requires at least one model.")
        self.models = models
        self.class_labels = class_labels or []
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        for model in self.models:
            model.to(self.device)
            model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> UncertaintyResult:
        x = x.to(self.device)
        probs_list = []
        for model in self.models:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu())
        all_probs = torch.stack(probs_list, dim=0).numpy()
        mean_probs = all_probs.mean(axis=0)[0]
        variance = all_probs.var(axis=0)[0]
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
        mutual_info = entropy - np.mean([-np.sum(p * np.log(p + 1e-8)) for p in all_probs[:, 0]])
        predicted_class = int(np.argmax(mean_probs))
        confidence = float(mean_probs[predicted_class])
        label = self.class_labels[predicted_class] if predicted_class < len(self.class_labels) else str(predicted_class)
        return UncertaintyResult(
            mean_probability=mean_probs,
            variance=variance,
            entropy=np.array([entropy]),
            mutual_information=np.array([mutual_info]),
            confidence=confidence,
            predicted_class=predicted_class,
            predicted_label=label,
        )


class TemperatureScaling:
    def __init__(self, model: nn.Module, device: str | None = None) -> None:
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = nn.Parameter(torch.ones(1, device=self.device))
        self.optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        self.criterion = nn.CrossEntropyLoss()

    def _eval_fn(self) -> torch.Tensor:
        self.optimizer.zero_grad()
        scaled_logits = self.logits / self.temperature.clamp(min=1e-6)
        loss = self.criterion(scaled_logits, self.labels)
        loss.backward()
        return loss

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        self.logits = logits.to(self.device)
        self.labels = labels.to(self.device)
        self.optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        self.optimizer.step(self._eval_fn)
        return float(self.temperature.item())

    @torch.no_grad()
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        scaled = logits / self.temperature.clamp(min=1e-6)
        return torch.softmax(scaled, dim=1)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Delegate ve medical.metrics.compute_calibration_curve de dung MOT nguon
    duy nhat cho logic ECE (tranh lech ket qua giua cac module).
    """
    from medical.metrics import compute_calibration_curve

    result = compute_calibration_curve(labels, probs, n_bins=n_bins)
    return float(result["ece"])


__all__ = [
    "UncertaintyResult",
    "MCDropoutUncertainty",
    "DeepEnsembleUncertainty",
    "TemperatureScaling",
    "compute_ece",
]
