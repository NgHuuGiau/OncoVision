from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from medical.cancer_catalog import list_common_cancer_targets
from medical.cancer_dataset_registry import common_cancer_dataset_source_dicts
from medical.dataset import create_default_skin_cancer_dataset_config
from training.cancer_download_plan import build_download_plan
from training.tcia_downloader import TCIA_DOWNLOAD_ROOT, load_collection_metadata, verify_downloads


DEFAULT_TCIA_COLLECTIONS_FILE = Path("training/tcia_collections_5.json")
SKIN_IMAGE_DIR = create_default_skin_cancer_dataset_config().dataset_root / "raw" / "images"
TARGET_KEY_TO_CANCER_TYPE = {
    "skin": "ung_thu_da",
    "breast": "ung_thu_vu",
    "lung": "ung_thu_phoi",
    "colorectal": "ung_thu_dai_truc_trang",
    "liver": "ung_thu_gan",
    "cervical": "ung_thu_co_tu_cung",
    "prostate": "ung_thu_tuyen_tien_liet",
    "stomach": "ung_thu_da_day",
}


def build_cancer_overview(collections_file: str | Path = DEFAULT_TCIA_COLLECTIONS_FILE) -> dict[str, object]:
    collections_path = Path(collections_file)
    download_plan = build_download_plan()
    sources = common_cancer_dataset_source_dicts()
    collection_type_map = {
        item["collection_name"]: item["cancer_type"] for item in load_collection_metadata(collections_path)
    }
    tcia_report = verify_downloads(collections_path)
    skin_raw_count = len(list(SKIN_IMAGE_DIR.glob("*"))) if SKIN_IMAGE_DIR.exists() else 0
    grouped_sources: dict[str, list[dict[str, str]]] = defaultdict(list)
    grouped_collections: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source in sources:
        grouped_sources[source["cancer_type"]].append(source)
    for item in tcia_report["collections"]:
        cancer_type = collection_type_map.get(item["collection_name"])
        if cancer_type is None:
            continue
        grouped_collections[cancer_type].append(
            {
                "collection_name": item["collection_name"],
                "image_count": item["downloaded_in_collection"],
                "collection_root": item["collection_root"],
                "failed_count": item["failed_count"],
            }
        )

    cancers: list[dict[str, object]] = []
    for target in list_common_cancer_targets():
        cancer_type = TARGET_KEY_TO_CANCER_TYPE[target.key]
        local_sources: list[dict[str, object]] = []
        local_image_count = 0
        if target.key == "skin":
            local_sources.append(
                {
                    "collection_name": "skin_lesion",
                    "image_count": skin_raw_count,
                    "collection_root": str(SKIN_IMAGE_DIR),
                    "failed_count": 0,
                }
            )
            local_image_count += skin_raw_count
        for collection in grouped_collections.get(cancer_type, []):
            local_sources.append(collection)
            local_image_count += int(collection["image_count"])
        if local_image_count > 0:
            local_status = "co_anh_local"
        elif grouped_sources.get(cancer_type):
            local_status = "co_nguon_nhung_chua_tai"
        else:
            local_status = "chua_co_nguon_local"
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
        "download_plan": download_plan,
        "summary": {
            "skin_raw_images": skin_raw_count,
            "tcia_total_images": tcia_report["downloaded_total"],
            "total_cancer_images": skin_raw_count + int(tcia_report["downloaded_total"]),
            "skin_image_dir": str(SKIN_IMAGE_DIR),
            "tcia_root": str(TCIA_DOWNLOAD_ROOT),
            "remaining_to_target": tcia_report["remaining_to_target"],
        },
        "cancers": cancers,
    }
