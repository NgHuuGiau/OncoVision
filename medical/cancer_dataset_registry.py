from __future__ import annotations

from dataclasses import asdict, dataclass


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
        source_name="ISIC skin cancer dataset",
        status="ready",
        official_url="https://www.isic-archive.com/",
        notes="Nguon san co cho nhom ung thu da.",
    ),
    CancerDatasetSourceSpec(
        source_id="tcia_liver",
        cancer_type="ung_thu_gan",
        source_name="TCIA liver dataset",
        status="ready",
        official_url="https://www.cancerimagingarchive.net/",
        notes="Nguon tham chieu TCIA cho ung thu gan.",
    ),
    CancerDatasetSourceSpec(
        source_id="tcia_lung",
        cancer_type="ung_thu_phoi",
        source_name="TCIA lung dataset",
        status="ready",
        official_url="https://www.cancerimagingarchive.net/",
        notes="Nguon tham chieu TCIA cho ung thu phoi.",
    ),
    CancerDatasetSourceSpec(
        source_id="tcia_breast",
        cancer_type="ung_thu_vu",
        source_name="TCIA breast dataset",
        status="ready",
        official_url="https://www.cancerimagingarchive.net/",
        notes="Nguon tham chieu TCIA cho ung thu vu.",
    ),
    CancerDatasetSourceSpec(
        source_id="tcia_stomach",
        cancer_type="ung_thu_da_day",
        source_name="TCIA stomach dataset",
        status="ready",
        official_url="https://www.cancerimagingarchive.net/",
        notes="Nguon tham chieu TCIA cho ung thu da day.",
    ),
    CancerDatasetSourceSpec(
        source_id="tcia_colorectal",
        cancer_type="ung_thu_dai_truc_trang",
        source_name="TCIA colorectal dataset",
        status="ready",
        official_url="https://www.cancerimagingarchive.net/",
        notes="Nguon tham chieu TCIA cho ung thu dai truc trang.",
    ),
    CancerDatasetSourceSpec(
        source_id="local_liver",
        cancer_type="ung_thu_gan",
        source_name="dataset/medical/Ung thư gan",
        status="ready",
        official_url="dataset/medical/Ung thư gan",
        notes="Ảnh đã có sẵn trong bộ dữ liệu local.",
    ),
    CancerDatasetSourceSpec(
        source_id="local_lung",
        cancer_type="ung_thu_phoi",
        source_name="dataset/medical/Ung thư phổi",
        status="ready",
        official_url="dataset/medical/Ung thư phổi",
        notes="Ảnh đã có sẵn trong bộ dữ liệu local.",
    ),
    CancerDatasetSourceSpec(
        source_id="local_breast",
        cancer_type="ung_thu_vu",
        source_name="dataset/medical/Ung thư vú",
        status="ready",
        official_url="dataset/medical/Ung thư vú",
        notes="Ảnh đã có sẵn trong bộ dữ liệu local.",
    ),
    CancerDatasetSourceSpec(
        source_id="local_stomach",
        cancer_type="ung_thu_da_day",
        source_name="dataset/medical/Ung thư dạ dày",
        status="ready",
        official_url="dataset/medical/Ung thư dạ dày",
        notes="Ảnh đã có sẵn trong bộ dữ liệu local.",
    ),
    CancerDatasetSourceSpec(
        source_id="local_colorectal",
        cancer_type="ung_thu_dai_truc_trang",
        source_name="dataset/medical/Ung thư đại trực tràng",
        status="ready",
        official_url="dataset/medical/Ung thư đại trực tràng",
        notes="Ảnh đã có sẵn trong bộ dữ liệu local.",
    ),
    CancerDatasetSourceSpec(
        source_id="local_prostate",
        cancer_type="ung_thu_tuyen_tien_liet",
        source_name="dataset/medical/Ung thư tuyến tiền liệt",
        status="ready",
        official_url="dataset/medical/Ung thư tuyến tiền liệt",
        notes="Ảnh đã có sẵn trong bộ dữ liệu local.",
    ),
    CancerDatasetSourceSpec(
        source_id="local_cervical",
        cancer_type="ung_thu_co_tu_cung",
        source_name="dataset/medical/Ung thư cổ tử cung",
        status="ready",
        official_url="dataset/medical/Ung thư cổ tử cung",
        notes="Ảnh đã có sẵn trong bộ dữ liệu local.",
    ),
)


def common_cancer_dataset_source_dicts() -> list[dict[str, str]]:
    return [asdict(spec) for spec in COMMON_CANCER_DATASET_SOURCES]
