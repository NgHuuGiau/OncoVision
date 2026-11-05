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
        description="Phân tích tổn thương gan trên CT, MRI, siêu âm hoặc PET/CT.",
        modalities=("CT gan", "MRI gan", "Siêu âm gan", "PET/CT"),
        model_ready=True,
        notes="Cần model riêng cho gan; classifier không tạo tọa độ.",
    ),
    CancerScreeningTarget(
        key="lung",
        label="Ung thư phổi",
        description="Phân tích nốt hoặc khối phổi trên X-quang, CT hoặc PET/CT.",
        modalities=("X-quang ngực", "CT ngực", "PET/CT"),
        model_ready=True,
        notes="CT 3D cần model volume; X-quang dùng model 2D riêng.",
    ),
    CancerScreeningTarget(
        key="breast",
        label="Ung thư vú",
        description="Phân tích ảnh mammogram, siêu âm hoặc MRI vú.",
        modalities=("Mammogram", "Siêu âm vú", "MRI vú"),
        model_ready=True,
        notes="Mỗi modality phải nằm trong phạm vi huấn luyện của model.",
    ),
    CancerScreeningTarget(
        key="stomach",
        label="Ung thư dạ dày",
        description="Phân tích ảnh nội soi và ảnh cắt lớp vùng dạ dày.",
        modalities=("Nội soi", "CT dạ dày", "MRI dạ dày", "PET/CT", "EUS"),
        model_ready=True,
        notes="Nội soi 2D và CT/MRI 3D cần model tương ứng.",
    ),
    CancerScreeningTarget(
        key="colorectal",
        label="Ung thư đại trực tràng",
        description="Phân tích nội soi đại tràng, CT, MRI trực tràng hoặc PET/CT.",
        modalities=("Nội soi đại tràng", "CT đại trực tràng", "MRI trực tràng", "PET/CT"),
        model_ready=True,
        notes="Không dùng chung model nội soi cho CT/MRI nếu chưa được huấn luyện đa modality.",
    ),
    CancerScreeningTarget(
        key="prostate",
        label="Ung thư tuyến tiền liệt",
        description="Phân tích MRI, siêu âm hoặc PET/CT tuyến tiền liệt.",
        modalities=("MRI tuyến tiền liệt", "Siêu âm tuyến tiền liệt", "PET/CT"),
        model_ready=True,
        notes="MRI đa chuỗi phải được đăng ký và khai báo đúng trong manifest 3D.",
    ),
    CancerScreeningTarget(
        key="cervical",
        label="Ung thư cổ tử cung",
        description="Phân tích CT, MRI hoặc PET/CT vùng cổ tử cung.",
        modalities=("Ảnh soi cổ tử cung", "CT cổ tử cung", "MRI cổ tử cung", "PET/CT"),
        model_ready=True,
        notes="Ảnh soi cổ tử cung cần model 2D riêng; Pap/HPV và mô bệnh học không đi qua pipeline ảnh.",
    ),
    CancerScreeningTarget(
        key="kidney",
        label="Ung thư thận",
        description="Phân tích CT, MRI, siêu âm hoặc PET/CT thận.",
        modalities=("CT thận", "MRI thận", "Siêu âm thận", "PET/CT thận"),
        model_ready=True,
        notes="CT/MRI 3D cần model volume riêng cho thận.",
    ),
    CancerScreeningTarget(
        key="pancreas",
        label="Ung thư tụy",
        description="Phân tích CT, MRI, PET/CT hoặc EUS tuyến tụy.",
        modalities=("CT tụy", "MRI tụy", "PET/CT tụy", "EUS"),
        model_ready=True,
        notes="Tổn thương tụy nhỏ cần segmentation 3D và dữ liệu lát mỏng phù hợp.",
    ),
    CancerScreeningTarget(
        key="thyroid",
        label="Ung thư tuyến giáp",
        description="Phân tích siêu âm, CT, MRI hoặc PET/CT tuyến giáp.",
        modalities=("Siêu âm tuyến giáp", "CT tuyến giáp", "MRI tuyến giáp", "PET/CT tuyến giáp"),
        model_ready=True,
        notes="Siêu âm 2D cần detector/classifier được huấn luyện riêng.",
    ),
    CancerScreeningTarget(
        key="brain",
        label="Ung thư não",
        description="Ảnh y khoa dùng để phân loại các nhóm u não.",
        modalities=("MRI não", "CT sọ não", "PET/CT não"),
        model_ready=True,
        notes="Model hiện có: ConvNeXt-Tiny, 4 loại u não (glioma/meningioma/pituitary/no_tumor).",
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
