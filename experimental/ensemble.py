"""Ensemble learning for medical image classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


@dataclass(frozen=True)
class EnsembleConfig:
    method: str = "stacking"
    models: list[str] | None = None
    weight_optimization: bool = True
    temperature_scaling: bool = True
    meta_learner: str = "logistic"


class MedicalEnsemble:
    def __init__(
        self,
        models: list[Any] | None = None,
        class_labels: list[str] | None = None,
        config: EnsembleConfig | None = None,
    ) -> None:
        self.models = models or []
        self.class_labels = class_labels or []
        self.config = config or EnsembleConfig()
        self.weights: np.ndarray | None = None
        self.meta_learner: LogisticRegression | None = None

    def add_model(self, model: Any) -> None:
        self.models.append(model)

    def predict_proba(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        if not self.models:
            raise ValueError("Ensemble khong co model nao.")
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        probs_list = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
        probs_list = np.stack(probs_list, axis=0)
        if self.config.method == "averaging":
            return np.mean(probs_list, axis=0)
        if self.config.method == "stacking" and self.meta_learner is not None:
            return self._stacking_predict(probs_list)
        if self.weights is not None:
            return np.tensordot(self.weights, probs_list, axes=([0], [0]))
        return np.mean(probs_list, axis=0)

    def fit(self, X: torch.Tensor | np.ndarray, y: np.ndarray) -> None:
        if len(self.models) < 2 or self.config.method != "stacking":
            if self.config.weight_optimization and len(self.models) > 1:
                self._optimize_weights(X, y)
            return
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        meta_features = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(X)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                meta_features.append(probs)
        meta_X = np.concatenate(meta_features, axis=1)
        self.meta_learner = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
        self.meta_learner.fit(meta_X, y)

    def _stacking_predict(self, probs_list: np.ndarray) -> np.ndarray:
        num_models, num_samples, num_classes = probs_list.shape
        meta_features = probs_list.transpose(1, 0, 2).reshape(num_samples, num_models * num_classes)
        return self.meta_learner.predict_proba(meta_features)

    def _optimize_weights(self, X: torch.Tensor | np.ndarray, y: np.ndarray) -> None:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        probs_list = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(X)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
        probs_array = np.stack(probs_list, axis=0)
        best_loss = float("inf")
        best_weights = np.ones(len(self.models)) / len(self.models)
        for _ in range(200):
            weights = np.random.dirichlet(np.ones(len(self.models)))
            ensemble_probs = np.tensordot(weights, probs_array, axes=([0], [0]))
            ensemble_probs = np.clip(ensemble_probs, 1e-8, 1.0)
            loss = log_loss(y, ensemble_probs)
            if loss < best_loss:
                best_loss = loss
                best_weights = weights
        self.weights = best_weights


__all__ = ["MedicalEnsemble", "EnsembleConfig"]
