from __future__ import annotations

from pathlib import Path

from medical.cnn_classifier import MedicalCNNClassifier, MedicalCNNClassifierWrapper


_MODALITY_LABELS = [
    "ct",
    "mri",
    "xray",
    "ultrasound",
    "mammogram",
    "endoscopy",
    "pet_ct",
    "eus",
]


def build_modality_classifier(num_classes: int = 8) -> MedicalCNNClassifier:
    return MedicalCNNClassifier(
        num_classes=num_classes,
        backbone="resnet18",
        pretrained=True,
        dropout=0.2,
    )


def save_modality_classifier(wrapper: MedicalCNNClassifierWrapper, path: str | Path) -> Path:
    return wrapper.save(path)


def load_modality_classifier(path: str | Path, device: str | None = None) -> MedicalCNNClassifierWrapper:
    return MedicalCNNClassifierWrapper.load(path, device=device)


def predict_modality_from_image(wrapper: MedicalCNNClassifierWrapper, source, *, top_k: int = 3):
    return wrapper.predict(source, top_k=top_k)
