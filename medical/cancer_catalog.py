from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CancerScreeningTarget:
    key: str
    label: str
    description: str
    modalities: tuple[str, ...]
    model_ready: bool
    notes: str


COMMON_CANCER_TARGETS: tuple[CancerScreeningTarget, ...] = (
    CancerScreeningTarget(
        key="liver",
        label="Ung thư gan",
        description="Ảnh y khoa dùng để sàng lọc ung thư gan.",
        modalities=("Siêu âm", "CT", "MRI", "PET/CT"),
        model_ready=False,
        notes="Chờ bổ sung model suy luận đã xây dựng bên ngoài.",
    ),
    CancerScreeningTarget(
        key="lung",
        label="Ung thư phổi",
        description="Ảnh y khoa dùng để sàng lọc ung thư phổi.",
        modalities=("X-quang ngực", "CT ngực", "PET/CT"),
        model_ready=False,
        notes="Chờ bổ sung model suy luận đã xây dựng bên ngoài.",
    ),
    CancerScreeningTarget(
        key="breast",
        label="Ung thư vú",
        description="Ảnh y khoa dùng để sàng lọc ung thư vú.",
        modalities=("Mammogram", "Siêu âm vú", "MRI vú"),
        model_ready=False,
        notes="Chờ bổ sung model suy luận đã xây dựng bên ngoài.",
    ),
    CancerScreeningTarget(
        key="stomach",
        label="Ung thư dạ dày",
        description="Ảnh y khoa dùng để sàng lọc ung thư dạ dày.",
        modalities=("Nội soi", "CT", "MRI", "PET", "EUS"),
        model_ready=False,
        notes="Chờ bổ sung model suy luận đã xây dựng bên ngoài.",
    ),
    CancerScreeningTarget(
        key="colorectal",
        label="Ung thư đại trực tràng",
        description="Ảnh y khoa dùng để sàng lọc ung thư đại trực tràng.",
        modalities=("Nội soi đại tràng", "CT ngực-bụng-chậu", "MRI trực tràng", "PET"),
        model_ready=False,
        notes="Chờ bổ sung model suy luận đã xây dựng bên ngoài.",
    ),
    CancerScreeningTarget(
        key="prostate",
        label="Ung thư tuyến tiền liệt",
        description="Ảnh y khoa dùng để sàng lọc ung thư tuyến tiền liệt.",
        modalities=("MRI tuyến tiền liệt", "Siêu âm", "PET/CT"),
        model_ready=False,
        notes="Chờ bổ sung model suy luận đã xây dựng bên ngoài.",
    ),
    CancerScreeningTarget(
        key="cervical",
        label="Ung thư cổ tử cung",
        description="Ảnh y khoa dùng để sàng lọc ung thư cổ tử cung.",
        modalities=("MRI", "CT", "PET/CT"),
        model_ready=False,
        notes="Pap/HPV, soi cổ tử cung và sinh thiết không được hỗ trợ; ảnh y khoa chờ model suy luận phù hợp.",
    ),
    CancerScreeningTarget(
        key="brain",
        label="Ung thư não",
        description="Ảnh y khoa dùng để phân loại các nhóm u não.",
        modalities=("MRI não", "CT sọ não", "PET/CT não"),
        model_ready=True,
        notes="Dữ liệu từ Figshare (708) + Kaggle (13,511) = 14,219 ảnh MRI não.",
    ),
    CancerScreeningTarget(
        key="kidney",
        label="Ung thư thận",
        description="Ảnh y khoa dùng để sàng lọc ung thư thận.",
        modalities=("CT thận", "MRI thận", "Siêu âm thận", "PET/CT thận"),
        model_ready=False,
        notes="Chờ bổ sung model suy luận đã xây dựng bên ngoài.",
    ),
    CancerScreeningTarget(
        key="pancreas",
        label="Ung thư tụy",
        description="Ảnh y khoa dùng để sàng lọc ung thư tụy.",
        modalities=("CT tụy", "MRI tụy", "PET/CT tụy"),
        model_ready=False,
        notes="Chờ bổ sung model suy luận đã xây dựng bên ngoài.",
    ),
    CancerScreeningTarget(
        key="thyroid",
        label="Ung thư tuyến giáp",
        description="Ảnh y khoa dùng để sàng lọc ung thư tuyến giáp.",
        modalities=("Siêu âm tuyến giáp", "CT tuyến giáp", "MRI tuyến giáp", "PET/CT tuyến giáp"),
        model_ready=False,
        notes="Chờ bổ sung model suy luận đã xây dựng bên ngoài.",
    ),
)


def supported_cancer_labels() -> list[str]:
    return [target.label for target in COMMON_CANCER_TARGETS]


def supported_cancer_modalities() -> list[str]:
    modalities: list[str] = []
    for target in COMMON_CANCER_TARGETS:
        for modality in target.modalities:
            if modality not in modalities:
                modalities.append(modality)
    return modalities
