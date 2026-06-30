from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class CancerDatasetSourceSpec:
    source_id: str
    cancer_type: str
    source_name: str
    status: str
    official_url: str
    notes: str


COMMON_CANCER_DATASET_SOURCES: tuple[CancerDatasetSourceSpec, ...] = (
    CancerDatasetSourceSpec(
        source_id="isic_skin",
        cancer_type="ung_thu_da",
        source_name="ISIC Archive",
        status="ready",
        official_url="https://www.isic-archive.com/",
        notes="Nguồn chính cho ảnh da liễu / skin lesion; có pipeline nhập sẵn qua training/import_skin_lesion_dataset.py.",
    ),
    CancerDatasetSourceSpec(
        source_id="tcia_breast",
        cancer_type="ung_thu_vu",
        source_name="TCIA CBIS-DDSM",
        status="source_confirmed",
        official_url="https://www.cancerimagingarchive.net/browse-collections/",
        notes="Mapping theo collection breast phổ biến trên TCIA; dùng cho mammography/breast imaging. TCGA-BRCA là collection dùng riêng.",
    ),
    CancerDatasetSourceSpec(
        source_id="tcia_lung",
        cancer_type="ung_thu_phoi",
        source_name="TCIA NSCLC-Radiomics",
        status="source_confirmed",
        official_url="https://www.cancerimagingarchive.net/browse-collections/",
        notes="Mapping theo collection phổi phổ biến trên TCIA; dữ liệu DICOM CT/PET/MRI. TCGA-LUAD là collection dùng riêng.",
    ),
    CancerDatasetSourceSpec(
        source_id="tcia_colorectal",
        cancer_type="ung_thu_dai_truc_trang",
        source_name="TCIA TCGA-COADREAD",
        status="source_confirmed",
        official_url="https://www.cancerimagingarchive.net/browse-collections/",
        notes="Mapping theo collection colorectal phổ biến trên TCIA; cần lọc đúng collection trước khi tải. COAD và READ là hai collection riêng, COADREAD là nhóm liên quan.",
    ),
    CancerDatasetSourceSpec(
        source_id="tcia_liver",
        cancer_type="ung_thu_gan",
        source_name="TCIA TCGA-LIHC",
        status="source_confirmed",
        official_url="https://www.cancerimagingarchive.net/browse-collections/",
        notes="Mapping theo liver cancer collection phổ biến trên TCIA.",
    ),
    CancerDatasetSourceSpec(
        source_id="tcia_stomach",
        cancer_type="ung_thu_da_day",
        source_name="TCIA TCGA-STAD",
        status="source_confirmed",
        official_url="https://www.cancerimagingarchive.net/browse-collections/",
        notes="Mapping theo stomach/gastric cancer collection phổ biến trên TCIA.",
    ),
)


def common_cancer_dataset_sources() -> tuple[CancerDatasetSourceSpec, ...]:
    return COMMON_CANCER_DATASET_SOURCES


def common_cancer_dataset_source_dicts() -> list[dict[str, str]]:
    return [asdict(spec) for spec in COMMON_CANCER_DATASET_SOURCES]
