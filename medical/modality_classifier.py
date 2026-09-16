from __future__ import annotations

from pathlib import Path

from medical.cnn_classifier import MedicalCNNClassifierWrapper


def load_modality_classifier(path: str | Path, device: str | None = None) -> MedicalCNNClassifierWrapper:
    return MedicalCNNClassifierWrapper.load(path, device=device)


def predict_modality_from_image(wrapper: MedicalCNNClassifierWrapper, source, *, top_k: int = 3):
    return wrapper.predict(source, top_k=top_k)
