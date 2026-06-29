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
        key="skin",
        label="Ung thu da",
        description="Sàng lọc tổn thương da và nguy cơ ung thư da.",
        model_ready=True,
        notes="Da co model skin_cancer_screening trong he thong hien tai.",
    ),
    CancerScreeningTarget(
        key="breast",
        label="Ung thu vu",
        description="Sang loc hinh anh lien quan toi ung thu vu.",
        model_ready=False,
        notes="Can bo sung dataset va model rieng.",
    ),
    CancerScreeningTarget(
        key="lung",
        label="Ung thu phoi",
        description="Sang loc hinh anh y khoa phoi/nguc lien quan ung thu phoi.",
        model_ready=False,
        notes="Can model DICOM/X-ray hoac CT chuyen dung.",
    ),
    CancerScreeningTarget(
        key="colorectal",
        label="Ung thu dai truc trang",
        description="Sang loc lien quan dai truc trang va ung thu ruot lon.",
        model_ready=False,
        notes="Can quy trinh rieng va dataset chuyen khoa.",
    ),
    CancerScreeningTarget(
        key="liver",
        label="Ung thu gan",
        description="Sàng lọc liên quan tổn thương gan.",
        model_ready=False,
        notes="Can model cho siêu am/CT/MRI chuyen dung.",
    ),
    CancerScreeningTarget(
        key="cervical",
        label="Ung thu co tu cung",
        description="Sang loc lien quan co tu cung.",
        model_ready=False,
        notes="Can quy trinh papanicolaou/anh soi co tu cung.",
    ),
    CancerScreeningTarget(
        key="prostate",
        label="Ung thu tuyen tien liet",
        description="Sang loc lien quan tuyen tien liet.",
        model_ready=False,
        notes="Cần model và dữ liệu riêng cho imaging/lab data.",
    ),
    CancerScreeningTarget(
        key="stomach",
        label="Ung thu da day",
        description="Sang loc lien quan da day/ong tieu hoa tren.",
        model_ready=False,
        notes="Can model noi soi hoac imaging rieng.",
    ),
)


def list_common_cancer_targets() -> tuple[CancerScreeningTarget, ...]:
    return COMMON_CANCER_TARGETS


def supported_cancer_labels() -> list[str]:
    return [target.label for target in COMMON_CANCER_TARGETS]
