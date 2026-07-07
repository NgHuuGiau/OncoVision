from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from medical.cancer_catalog import COMMON_CANCER_TARGETS
from medical.classifier import iter_medical_image_paths
from medical.reporting import build_artifact_stamp
from utils.file_utils import save_yaml


MEDICAL_DATASET_ROOT = Path("dataset/medical")
MEDICAL_CLASS_NAMES = tuple(target.label for target in COMMON_CANCER_TARGETS)


@dataclass(frozen=True)
class MedicalDatasetConfig:
    disease_name: str
    dataset_root: Path
    data_yaml_path: Path
    metadata_dir: Path
    reports_dir: Path
    class_names: tuple[str, ...]
    image_size: int


@dataclass(frozen=True)
class MedicalDatasetSummary:
    dataset_root: Path
    created_directories: list[Path]
    data_yaml_path: Path


def create_default_medical_dataset_config(dataset_root: str | Path = MEDICAL_DATASET_ROOT) -> MedicalDatasetConfig:
    root = Path(dataset_root)
    return MedicalDatasetConfig(
        disease_name="medical_7_cancers",
        dataset_root=root,
        data_yaml_path=root / "data.yaml",
        metadata_dir=root / "metadata",
        reports_dir=root / "reports",
        class_names=MEDICAL_CLASS_NAMES,
        image_size=320,
    )


def create_default_skin_cancer_dataset_config(dataset_root: str | Path = MEDICAL_DATASET_ROOT) -> MedicalDatasetConfig:
    return create_default_medical_dataset_config(dataset_root)


def ensure_medical_dataset_structure(config: MedicalDatasetConfig | None = None) -> MedicalDatasetSummary:
    config = config or create_default_medical_dataset_config()
    created_dirs = [config.dataset_root, config.metadata_dir, config.reports_dir]
    for class_name in config.class_names:
        for split in ("train", "val", "test"):
            created_dirs.append(config.dataset_root / class_name / "processed" / "images" / split)
    for directory in created_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    save_yaml(
        config.data_yaml_path,
        {
            "path": str(config.dataset_root.resolve()),
            "task": "classification",
            "train": str(config.dataset_root.resolve()),
            "val": str(config.dataset_root.resolve()),
            "test": str(config.dataset_root.resolve()),
            "names": {index: name for index, name in enumerate(config.class_names)},
        },
    )
    return MedicalDatasetSummary(
        dataset_root=config.dataset_root,
        created_directories=created_dirs,
        data_yaml_path=config.data_yaml_path,
    )


def iter_medical_class_split_images(dataset_root: str | Path, split: str) -> dict[str, list[Path]]:
    root = Path(dataset_root)
    split_map: dict[str, list[Path]] = {}
    for class_name in MEDICAL_CLASS_NAMES:
        split_dir = root / class_name / "processed" / "images" / split
        split_map[class_name] = list(iter_medical_image_paths(split_dir))
    return split_map


def count_medical_class_split_images(dataset_root: str | Path, split: str) -> dict[str, int]:
    return {class_name: len(paths) for class_name, paths in iter_medical_class_split_images(dataset_root, split).items()}


def normalize_uploaded_image(
    source_path: str | Path,
    target_dir: str | Path,
    *,
    image_size: int = 320,
) -> Path:
    source = Path(source_path)
    destination_dir = Path(target_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        normalized = image.convert("RGB")
        normalized.thumbnail((image_size, image_size))
        background = Image.new("RGB", (image_size, image_size), (0, 0, 0))
        offset = ((image_size - normalized.width) // 2, (image_size - normalized.height) // 2)
        background.paste(normalized, offset)
        target_path = destination_dir / f"{source.stem}_{build_artifact_stamp()}.jpg"
        background.save(target_path, format="JPEG", quality=95)
    return target_path
