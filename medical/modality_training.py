from __future__ import annotations

from pathlib import Path



def train_modality_classifier(
    dataset_root: str | Path = "dataset/medical",
    output_path: str | Path = "models/pretrained/modality_classifier.pt",
    *,
    image_size: int = 320,
    batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
) -> Path:
    raise NotImplementedError(
        "Modality classifier training requires a labeled dataset where each image "
        "is annotated with its modality (ct, mri, xray, ultrasound, mammogram, "
        "endoscopy, pet_ct, eus). The current medical dataset does not contain "
        "modality labels. To enable this feature, prepare a dataset with "
        "subfolders named after modalities, or provide a CSV mapping images to modalities."
    )
