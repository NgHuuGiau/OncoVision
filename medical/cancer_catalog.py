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
        key="brain",
        label="Ung thư não",
        description="Ảnh y khoa dùng để phân loại các nhóm u não.",
        modalities=("MRI não", "CT sọ não", "PET/CT não"),
        model_ready=True,
        notes="ConvNeXt-Tiny, 4 loại u não (glioma/meningioma/pituitary/no_tumor).",
    ),
)


def get_cancer_target(key: str) -> CancerScreeningTarget | None:
    normalized = key.strip().lower()
    return next((target for target in COMMON_CANCER_TARGETS if target.key == normalized), None)


def supported_cancer_labels() -> list[str]:
    return [target.label for target in COMMON_CANCER_TARGETS]


def supported_cancer_modalities() -> list[str]:
    modalities: list[str] = []
    for target in COMMON_CANCER_TARGETS:
        for modality in target.modalities:
            if modality not in modalities:
                modalities.append(modality)
    return modalities
