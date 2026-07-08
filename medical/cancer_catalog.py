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
        description="Nhóm dữ liệu hình ảnh cho ung thư gan.",
        modalities=("Siêu âm", "CT", "MRI", "PET/CT"),
        model_ready=True,
        notes="Dữ liệu sẵn trong dataset/medical/Ung thư gan.",
    ),
    CancerScreeningTarget(
        key="lung",
        label="Ung thư phổi",
        description="Nhóm dữ liệu hình ảnh cho ung thư phổi.",
        modalities=("X-quang ngực", "CT ngực", "PET/CT"),
        model_ready=True,
        notes="Dữ liệu sẵn trong dataset/medical/Ung thư phổi.",
    ),
    CancerScreeningTarget(
        key="breast",
        label="Ung thư vú",
        description="Nhóm dữ liệu hình ảnh cho ung thư vú.",
        modalities=("Mammogram", "Siêu âm vú", "MRI vú"),
        model_ready=True,
        notes="Dữ liệu sẵn trong dataset/medical/Ung thư vú.",
    ),
    CancerScreeningTarget(
        key="stomach",
        label="Ung thư dạ dày",
        description="Nhóm dữ liệu hình ảnh cho ung thư dạ dày.",
        modalities=("Nội soi", "CT", "MRI", "PET", "EUS"),
        model_ready=True,
        notes="Dữ liệu sẵn trong dataset/medical/Ung thư dạ dày.",
    ),
    CancerScreeningTarget(
        key="colorectal",
        label="Ung thư đại trực tràng",
        description="Nhóm dữ liệu hình ảnh cho ung thư đại trực tràng.",
        modalities=("Nội soi đại tràng", "CT ngực-bụng-chậu", "MRI trực tràng", "PET"),
        model_ready=True,
        notes="Dữ liệu sẵn trong dataset/medical/Ung thư đại trực tràng.",
    ),
    CancerScreeningTarget(
        key="prostate",
        label="Ung thư tuyến tiền liệt",
        description="Nhóm dữ liệu hình ảnh cho ung thư tuyến tiền liệt.",
        modalities=("MRI tuyến tiền liệt", "Siêu âm", "PET/CT"),
        model_ready=True,
        notes="Dữ liệu sẵn trong dataset/medical/Ung thư tuyến tiền liệt.",
    ),
    CancerScreeningTarget(
        key="cervical",
        label="Ung thư cổ tử cung",
        description="Nhóm dữ liệu hình ảnh cho ung thư cổ tử cung.",
        modalities=("MRI", "CT", "PET/CT"),
        model_ready=True,
        notes="Pap/HPV, soi cổ tử cung và sinh thiết là đầu vào lâm sàng; phần ảnh staging thường dùng MRI, CT, PET/CT.",
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
