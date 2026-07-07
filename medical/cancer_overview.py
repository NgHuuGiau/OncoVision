from __future__ import annotations

from pathlib import Path

from medical.cancer_catalog import COMMON_CANCER_TARGETS
from medical.cancer_dataset_registry import common_cancer_dataset_source_dicts
from medical.classifier import iter_medical_image_paths
from medical.dataset import MEDICAL_DATASET_ROOT


TARGET_KEY_TO_CANCER_TYPE = {
    "liver": "ung_thu_gan",
    "lung": "ung_thu_phoi",
    "breast": "ung_thu_vu",
    "stomach": "ung_thu_da_day",
    "colorectal": "ung_thu_dai_truc_trang",
    "prostate": "ung_thu_tuyen_tien_liet",
    "cervical": "ung_thu_co_tu_cung",
}


def build_cancer_overview(dataset_root: str | Path = MEDICAL_DATASET_ROOT) -> dict[str, object]:
    root = Path(dataset_root)
    sources = common_cancer_dataset_source_dicts()
    grouped_sources: dict[str, list[dict[str, str]]] = {}
    for source in sources:
        grouped_sources.setdefault(source["cancer_type"], []).append(source)

    cancers: list[dict[str, object]] = []
    total_images = 0
    for target in COMMON_CANCER_TARGETS:
        cancer_type = TARGET_KEY_TO_CANCER_TYPE[target.key]
        train_dir = root / target.label / "processed" / "images" / "train"
        val_dir = root / target.label / "processed" / "images" / "val"
        test_dir = root / target.label / "processed" / "images" / "test"
        train_count = sum(1 for _ in iter_medical_image_paths(train_dir))
        val_count = sum(1 for _ in iter_medical_image_paths(val_dir))
        test_count = sum(1 for _ in iter_medical_image_paths(test_dir))
        local_sources = [
            {
                "collection_name": "train",
                "image_count": train_count,
                "collection_root": str(train_dir),
                "failed_count": 0,
            },
            {
                "collection_name": "val",
                "image_count": val_count,
                "collection_root": str(val_dir),
                "failed_count": 0,
            },
            {
                "collection_name": "test",
                "image_count": test_count,
                "collection_root": str(test_dir),
                "failed_count": 0,
            },
        ]
        local_image_count = sum(item["image_count"] for item in local_sources)
        total_images += local_image_count
        local_status = "co_anh_local" if local_image_count > 0 else "chua_co_anh_local"
        cancers.append(
            {
                "cancer_type": cancer_type,
                "target_key": target.key,
                "label": target.label,
                "description": target.description,
                "model_ready": target.model_ready,
                "model_notes": target.notes,
                "local_status": local_status,
                "local_image_count": local_image_count,
                "sources": grouped_sources.get(cancer_type, []),
                "local_sources": local_sources,
            }
        )
    return {
        "sources": sources,
        "download_plan": [],
        "summary": {
            "total_cancer_images": total_images,
            "dataset_root": str(root),
        },
        "cancers": cancers,
    }
