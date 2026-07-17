from __future__ import annotations

import random
from pathlib import Path

from medical.classifier import iter_medical_image_paths
from medical.cnn_classifier import train_cnn_classifier
from medical.modality_classifier import (
    _MODALITY_LABELS,
    build_modality_classifier,
    save_modality_classifier,
)


def _collect_modality_samples(dataset_root: Path) -> list[tuple[Path, int]]:
    """Quet thu muc dataset_root/<modality>/... lay (anh, chi_so_modality)."""
    samples: list[tuple[Path, int]] = []
    for class_index, modality in enumerate(_MODALITY_LABELS):
        modality_dir = dataset_root / modality
        if not modality_dir.exists():
            continue
        for image_path in iter_medical_image_paths(modality_dir):
            samples.append((image_path, class_index))
    return samples


def _split_train_val(
    samples: list[tuple[Path, int]], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Chia train/val theo ti lệ, giu can bang class (stratified) de danh gia."""
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for item in samples:
        by_class.setdefault(item[1], []).append(item)
    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    for items in by_class.values():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def train_modality_classifier(
    dataset_root: str | Path = "dataset/medical_modality",
    output_path: str | Path = "models/pretrained/modality_classifier.pt",
    *,
    image_size: int = 320,
    batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    pretrained: bool = True,
    verbose: bool = False,
) -> Path:
    root = Path(dataset_root)
    samples = _collect_modality_samples(root)
    if not samples:
        raise FileNotFoundError(
            "Khong tim thay anh modality. Hay tao cau truc dataset/medical_modality/"
            "<modality>/... voi cac thu muc: " + ", ".join(_MODALITY_LABELS)
        )

    train_samples, val_samples = _split_train_val(samples)
    if not val_samples:
        val_samples = None

    existing_labels = {label for _, idx in samples for label in (_MODALITY_LABELS[idx],)}
    class_labels = tuple(label for label in _MODALITY_LABELS if label in existing_labels)
    # Map chi so trong _MODALITY_LABELS goc sang chi so cua class_labels thuc te.
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    remapped_train = [(path, label_to_index[_MODALITY_LABELS[idx]]) for path, idx in train_samples]
    remapped_val = [(path, label_to_index[_MODALITY_LABELS[idx]]) for path, idx in val_samples] if val_samples else None

    model = build_modality_classifier(num_classes=len(class_labels))
    wrapper, _history = train_cnn_classifier(
        remapped_train,
        class_labels=class_labels,
        image_size=image_size,
        backbone=model.backbone_name,
        pretrained=pretrained,
        dropout=0.2,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        val_samples=remapped_val,
        early_stopping_patience=max(3, num_epochs // 3),
        label_smoothing=0.1,
        mixed_precision=True,
        warmup_epochs=max(1, num_epochs // 5),
        progress_tag="modality",
        verbose=verbose,
    )
    save_modality_classifier(wrapper, output_path)
    return Path(output_path)

