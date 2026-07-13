from __future__ import annotations

from dataclasses import dataclass

# Canonical image-type families (nhóm theo kiểu ảnh, không phải theo từng ung thư).
CT_VOLUME_FAMILY = "ct_volume"
ENDOSCOPY_FAMILY = "endoscopy"
XRAY_MAMMO_FAMILY = "xray_mammo"
CLINICAL_FAMILY = "clinical_non_image"


# Body region -> family cho đầu vào ảnh. Phổi (lung) được tinh chỉnh theo modality
# ở hàm route_input vì nó nằm ở hai nhóm (X-quang vào xray_mammo, CT/MRI vào ct_volume).
# Khóa dùng canonical body region từ dataset.py: liver, lung, breast, stomach,
# colorectal, prostate, cervical.
_BODY_REGION_FAMILY: dict[str, str] = {
    "liver": CT_VOLUME_FAMILY,
    "lung": CT_VOLUME_FAMILY,
    "prostate": CT_VOLUME_FAMILY,
    "cervix": CT_VOLUME_FAMILY,
    "stomach": ENDOSCOPY_FAMILY,
    "colorectal": ENDOSCOPY_FAMILY,
    "breast": XRAY_MAMMO_FAMILY,
}

_VOLUME_MODALITIES = frozenset({"ct", "mri", "pet_ct"})
_PLANAR_MODALITIES = frozenset({"mammogram", "xray", "ultrasound"})

_MULTIMODAL_FAMILIES = frozenset({ENDOSCOPY_FAMILY})

# Các body region cần luồng đa phương thức (ảnh + metadata + mô tả thủ thuật).
_MULTIMODAL_BODY_REGIONS = frozenset({"stomach", "colorectal"})

# Nhóm ung thư có ít ảnh hơn, cần reweight/oversample trước khi kỳ vọng ổn định.
UNDERREPRESENTED_BODY_REGIONS = frozenset({"liver", "cervix"})

# Thành viên mỗi family dùng làm nhãn (label set) khi huấn luyện submodel.
IMAGE_TYPE_FAMILIES: dict[str, dict[str, object]] = {
    CT_VOLUME_FAMILY: {
        "label": "Ảnh cắt lớp (CT/MRI/PET/CT): gan, phổi, tuyến tiền liệt, cổ tử cung",
        "members": ("liver", "lung", "prostate", "cervix"),
        "multimodal": False,
    },
    ENDOSCOPY_FAMILY: {
        "label": "Nội soi: dạ dày, đại trực tràng",
        "members": ("stomach", "colorectal"),
        "multimodal": True,
    },
    XRAY_MAMMO_FAMILY: {
        "label": "X-quang / Mammogram: vú, một phần phổi",
        "members": ("breast", "lung"),
        "multimodal": False,
    },
}


@dataclass(frozen=True)
class InputRoute:
    family: str | None
    body_region: str | None
    modality: str | None
    multimodal: bool = False
    clinical_only: bool = False
    resolved: bool = True

    @property
    def requires_clinical_workflow(self) -> bool:
        return self.clinical_only

    @property
    def requires_multimodal_input(self) -> bool:
        return self.multimodal


def route_input(
    modality: str | None,
    body_region: str | None,
    *,
    allow_clinical_cervical: bool = True,
) -> InputRoute:
    # Cổ tử cung: tách luồng lâm sàng (Pap/HPV/colposcopy) khỏi luồng ảnh.
    if body_region == "cervix" and (modality is None or modality not in _VOLUME_MODALITIES):
        if allow_clinical_cervical and modality not in _PLANAR_MODALITIES:
            return InputRoute(
                family=CLINICAL_FAMILY,
                body_region="cervix",
                modality=modality,
                clinical_only=True,
            )
        return InputRoute(
            family=CLINICAL_FAMILY,
            body_region="cervix",
            modality=modality,
            clinical_only=True,
        )

    if body_region is None:
        return InputRoute(family=None, body_region=None, modality=modality, resolved=False)

    family = _BODY_REGION_FAMILY.get(body_region)
    if family is None:
        return InputRoute(family=None, body_region=body_region, modality=modality, resolved=False)

    # Phổi phân nhánh theo modality: X-quang/mammogram -> xray_mammo, CT/MRI/PET -> ct_volume.
    if body_region == "lung" and modality in _PLANAR_MODALITIES:
        family = XRAY_MAMMO_FAMILY
    elif body_region == "lung" and modality in _VOLUME_MODALITIES:
        family = CT_VOLUME_FAMILY

    multimodal = family in _MULTIMODAL_FAMILIES or body_region in _MULTIMODAL_BODY_REGIONS
    return InputRoute(family=family, body_region=body_region, modality=modality, multimodal=multimodal)


from typing import Any, cast


def family_members(family_key: str) -> tuple[str, ...]:
    config = IMAGE_TYPE_FAMILIES.get(family_key)
    if config is None:
        return ()
    members = cast(list[str], config["members"])
    return tuple(members)


def is_underrepresented(body_region: str) -> bool:
    return body_region in UNDERREPRESENTED_BODY_REGIONS
