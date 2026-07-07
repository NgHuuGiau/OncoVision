from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CancerScreeningTarget:
    key: str
    label: str
    description: str
    model_ready: bool
    notes: str


COMMON_CANCER_TARGETS: tuple[CancerScreeningTarget, ...] = (
    CancerScreeningTarget(
        key="liver",
        label="Ung thư gan",
        description="Nhóm dữ liệu hình ảnh cho ung thư gan.",
        model_ready=True,
        notes="Dữ liệu sẵn trong dataset/medical/Ung thư gan.",
    ),
    CancerScreeningTarget(
        key="lung",
        label="Ung thư phổi",
        description="Nhóm dữ liệu hình ảnh cho ung thư phổi.",
        model_ready=True,
        notes="Dữ liệu sẵn trong dataset/medical/Ung thư phổi.",
    ),
    CancerScreeningTarget(
        key="breast",
        label="Ung thư vú",
        description="Nhóm dữ liệu hình ảnh cho ung thư vú.",
        model_ready=True,
        notes="Dữ liệu sẵn trong dataset/medical/Ung thư vú.",
    ),
    CancerScreeningTarget(
        key="stomach",
        label="Ung thư dạ dày",
        description="Nhóm dữ liệu hình ảnh cho ung thư dạ dày.",
        model_ready=True,
        notes="Dữ liệu sẵn trong dataset/medical/Ung thư dạ dày.",
    ),
    CancerScreeningTarget(
        key="colorectal",
        label="Ung thư đại trực tràng",
        description="Nhóm dữ liệu hình ảnh cho ung thư đại trực tràng.",
        model_ready=True,
        notes="Dữ liệu sẵn trong dataset/medical/Ung thư đại trực tràng.",
    ),
    CancerScreeningTarget(
        key="prostate",
        label="Ung thư tuyến tiền liệt",
        description="Nhóm dữ liệu hình ảnh cho ung thư tuyến tiền liệt.",
        model_ready=True,
        notes="Dữ liệu sẵn trong dataset/medical/Ung thư tuyến tiền liệt.",
    ),
    CancerScreeningTarget(
        key="cervical",
        label="Ung thư cổ tử cung",
        description="Nhóm dữ liệu hình ảnh cho ung thư cổ tử cung.",
        model_ready=True,
        notes="Dữ liệu sẵn trong dataset/medical/Ung thư cổ tử cung.",
    ),
)


def supported_cancer_labels() -> list[str]:
    return [target.label for target in COMMON_CANCER_TARGETS]
