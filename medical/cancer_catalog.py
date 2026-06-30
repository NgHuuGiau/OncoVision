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
        notes="Đã có model skin_cancer_screening trong hệ thống hiện tại.",
    ),
    CancerScreeningTarget(
        key="breast",
        label="Ung thu vu",
        description="Sàng lọc hình ảnh liên quan tới ung thư vú.",
        model_ready=False,
        notes="Cần bổ sung dataset và model riêng.",
    ),
    CancerScreeningTarget(
        key="lung",
        label="Ung thu phoi",
        description="Sàng lọc hình ảnh y khoa phổi/ngực liên quan ung thư phổi.",
        model_ready=False,
        notes="Cần model DICOM/X-ray hoặc CT chuyên dụng.",
    ),
    CancerScreeningTarget(
        key="colorectal",
        label="Ung thu dai truc trang",
        description="Sàng lọc liên quan đại trực tràng và ung thư ruột lớn.",
        model_ready=False,
        notes="Cần quy trình riêng và dataset chuyên khoa.",
    ),
    CancerScreeningTarget(
        key="liver",
        label="Ung thu gan",
        description="Sàng lọc liên quan tổn thương gan.",
        model_ready=False,
        notes="Cần model cho siêu âm/CT/MRI chuyên dụng.",
    ),
    CancerScreeningTarget(
        key="cervical",
        label="Ung thu co tu cung",
        description="Sàng lọc liên quan cổ tử cung.",
        model_ready=False,
        notes="Cần quy trình papanicolaou/ảnh soi cổ tử cung.",
    ),
    CancerScreeningTarget(
        key="prostate",
        label="Ung thu tuyen tien liet",
        description="Sàng lọc liên quan tuyến tiền liệt.",
        model_ready=False,
        notes="Cần model và dữ liệu riêng cho imaging/lab data.",
    ),
    CancerScreeningTarget(
        key="stomach",
        label="Ung thu da day",
        description="Sàng lọc liên quan dạ dày/ống tiêu hóa trên.",
        model_ready=False,
        notes="Cần model nội soi hoặc imaging riêng.",
    ),
)


def list_common_cancer_targets() -> tuple[CancerScreeningTarget, ...]:
    return COMMON_CANCER_TARGETS


def supported_cancer_labels() -> list[str]:
    return [target.label for target in COMMON_CANCER_TARGETS]
