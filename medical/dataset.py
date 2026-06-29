from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from medical.reporting import build_artifact_stamp
from utils.file_utils import save_yaml


MEDICAL_DATASET_ROOT = Path("dataset/medical/skin_lesion")
MEDICAL_CANCER_DATASET_ROOT = Path("dataset/medical_cancer")
MEDICAL_ROOT = Path("dataset/medical")
OBJECT_DETECTION_ROOT = Path("dataset/object_detection")


@dataclass(frozen=True)
class MedicalDatasetConfig:
    disease_name: str
    dataset_root: Path
    raw_images_dir: Path
    raw_labels_dir: Path
    processed_images_dir: Path
    processed_labels_dir: Path
    metadata_dir: Path
    reports_dir: Path
    data_yaml_path: Path
    image_size: int
    class_names: dict[int, str]


@dataclass(frozen=True)
class MedicalDatasetSummary:
    dataset_root: Path
    created_directories: list[Path]
    data_yaml_path: Path


def create_default_skin_cancer_dataset_config(dataset_root: str | Path = MEDICAL_DATASET_ROOT) -> MedicalDatasetConfig:
    root = Path(dataset_root)
    return MedicalDatasetConfig(
        disease_name="skin_cancer_screening",
        dataset_root=root,
        raw_images_dir=root / "raw" / "images",
        raw_labels_dir=root / "raw" / "labels",
        processed_images_dir=root / "processed" / "images",
        processed_labels_dir=root / "processed" / "labels",
        metadata_dir=root / "metadata",
        reports_dir=root / "reports",
        data_yaml_path=root / "data.yaml",
        image_size=640,
        class_names={0: "lesion"},
    )


def create_default_medical_cancer_dataset_config(dataset_root: str | Path = MEDICAL_CANCER_DATASET_ROOT) -> MedicalDatasetConfig:
    root = Path(dataset_root)
    return MedicalDatasetConfig(
        disease_name="medical_cancer_screening",
        dataset_root=root,
        raw_images_dir=root / "raw" / "images",
        raw_labels_dir=root / "raw" / "labels",
        processed_images_dir=root / "processed" / "images",
        processed_labels_dir=root / "processed" / "labels",
        metadata_dir=root / "metadata",
        reports_dir=root / "reports",
        data_yaml_path=root / "data.yaml",
        image_size=640,
        class_names={0: "lesion"},
    )


def create_default_medical_dataset_config(dataset_root: str | Path = MEDICAL_ROOT) -> MedicalDatasetConfig:
    root = Path(dataset_root)
    return MedicalDatasetConfig(
        disease_name="medical_ai",
        dataset_root=root,
        raw_images_dir=root / "raw" / "images",
        raw_labels_dir=root / "raw" / "labels",
        processed_images_dir=root / "processed" / "images",
        processed_labels_dir=root / "processed" / "labels",
        metadata_dir=root / "metadata",
        reports_dir=root / "reports",
        data_yaml_path=root / "data.yaml",
        image_size=640,
        class_names={0: "lesion"},
    )


def create_default_object_detection_dataset_config(
    dataset_root: str | Path = OBJECT_DETECTION_ROOT,
) -> MedicalDatasetConfig:
    root = Path(dataset_root)
    return MedicalDatasetConfig(
        disease_name="object_detection",
        dataset_root=root,
        raw_images_dir=root / "raw" / "images",
        raw_labels_dir=root / "raw" / "labels",
        processed_images_dir=root / "processed" / "images",
        processed_labels_dir=root / "processed" / "labels",
        metadata_dir=root / "metadata",
        reports_dir=root / "reports",
        data_yaml_path=root / "data.yaml",
        image_size=640,
        class_names={0: "object"},
    )


def ensure_medical_dataset_structure(config: MedicalDatasetConfig | None = None) -> MedicalDatasetSummary:
    config = config or create_default_skin_cancer_dataset_config()
    created_dirs = [
        config.raw_images_dir,
        config.raw_labels_dir,
        config.processed_images_dir / "train",
        config.processed_images_dir / "val",
        config.processed_images_dir / "test",
        config.processed_labels_dir / "train",
        config.processed_labels_dir / "val",
        config.processed_labels_dir / "test",
        config.metadata_dir,
        config.reports_dir,
    ]
    for directory in created_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    save_yaml(
        config.data_yaml_path,
        {
            "path": str(config.dataset_root.resolve()),
            "train": "processed/images/train",
            "val": "processed/images/val",
            "test": "processed/images/test",
            "names": config.class_names,
        },
    )
    return MedicalDatasetSummary(
        dataset_root=config.dataset_root,
        created_directories=created_dirs,
        data_yaml_path=config.data_yaml_path,
    )


def ensure_medical_cancer_dataset_structure(config: MedicalDatasetConfig | None = None) -> MedicalDatasetSummary:
    config = config or create_default_medical_cancer_dataset_config()
    return ensure_medical_dataset_structure(config)


def ensure_medical_dataset_root_structure(config: MedicalDatasetConfig | None = None) -> MedicalDatasetSummary:
    config = config or create_default_medical_dataset_config()
    return ensure_medical_dataset_structure(config)


def ensure_object_detection_dataset_structure(config: MedicalDatasetConfig | None = None) -> MedicalDatasetSummary:
    config = config or create_default_object_detection_dataset_config()
    return ensure_medical_dataset_structure(config)


def normalize_uploaded_image(
    source_path: str | Path,
    target_dir: str | Path,
    *,
    image_size: int = 640,
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
