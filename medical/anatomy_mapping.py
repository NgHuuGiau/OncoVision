from __future__ import annotations

"""
Cancer type to 3D organ mapping for OncoVision 3D Anatomy Viewer.

Maps each cancer target key to the corresponding organ names in the 3D model.
These organ names must match the object names in the exported .glb files.
"""

from typing import TypedDict


class OrganHighlight(TypedDict):
    """Organ highlight configuration for 3D viewer."""
    name: str
    color: str
    opacity: float


class CancerOrganMapping(TypedDict):
    """Complete mapping for a cancer type."""
    target: str
    label: str
    organs_to_highlight: list[OrganHighlight]


# Vietnamese labels for cancer types
CANCER_LABELS_VI: dict[str, str] = {
    "liver": "Ung thư gan",
    "lung": "Ung thư phổi",
    "breast": "Ung thư vú",
    "stomach": "Ung thư dạ dày",
    "colorectal": "Ung thư đại trực tràng",
    "prostate": "Ung thư tuyến tiền liệt",
    "cervical": "Ung thư cổ tử cung",
    "kidney": "Ung thư thận",
    "pancreas": "Ung thư tụy",
    "thyroid": "Ung thư tuyến giáp",
    "brain": "Ung thư não",
}

# English labels for cancer types
CANCER_LABELS_EN: dict[str, str] = {
    "liver": "Liver Cancer",
    "lung": "Lung Cancer",
    "breast": "Breast Cancer",
    "stomach": "Stomach Cancer",
    "colorectal": "Colorectal Cancer",
    "prostate": "Prostate Cancer",
    "cervical": "Cervical Cancer",
    "kidney": "Kidney Cancer",
    "pancreas": "Pancreas Cancer",
    "thyroid": "Thyroid Cancer",
    "brain": "Brain Cancer",
}

# Organ display names (Vietnamese)
ORGAN_NAMES_VI: dict[str, str] = {
    "Liver": "Gan",
    "Lung_Left": "Phổi trái",
    "Lung_Right": "Phổi phải",
    "Breast_Left": "Vú trái",
    "Breast_Right": "Vú phải",
    "Stomach": "Dạ dày",
    "Colon": "Cổ tràng",
    "Rectum": "Trực tràng",
    "Prostate": "Tuyến tiền liệt",
    "Cervix": "Cổ tử cung",
    "Uterus": "Tử cung",
    "Kidney_Left": "Thận trái",
    "Kidney_Right": "Thận phải",
    "Pancreas": "Tụy",
    "Thyroid": "Tuyến giáp",
    "Brain": "Não",
}

# Organ display names (English)
ORGAN_NAMES_EN: dict[str, str] = {
    "Liver": "Liver",
    "Lung_Left": "Left Lung",
    "Lung_Right": "Right Lung",
    "Breast_Left": "Left Breast",
    "Breast_Right": "Right Breast",
    "Stomach": "Stomach",
    "Colon": "Colon",
    "Rectum": "Rectum",
    "Prostate": "Prostate",
    "Cervix": "Cervix",
    "Uterus": "Uterus",
    "Kidney_Left": "Left Kidney",
    "Kidney_Right": "Right Kidney",
    "Pancreas": "Pancreas",
    "Thyroid": "Thyroid",
    "Brain": "Brain",
}


# Main mapping: cancer_key -> list of organ names in the 3D model
# These names should match the object names in the exported .glb files
CANCER_TO_ORGANS: dict[str, list[str]] = {
    "liver": ["Liver"],
    "lung": ["Lung_Left", "Lung_Right"],
    "breast": ["Breast_Left", "Breast_Right"],
    "stomach": ["Stomach"],
    "colorectal": ["Colon", "Rectum"],
    "prostate": ["Prostate"],
    "cervical": ["Cervix", "Uterus"],
    "kidney": ["Kidney_Left", "Kidney_Right"],
    "pancreas": ["Pancreas"],
    "thyroid": ["Thyroid"],
    "brain": ["Brain"],
}


def get_organ_highlights(cancer_key: str, language: str = "vi") -> list[OrganHighlight]:
    """
    Get organ highlight configuration for a cancer type.

    Args:
        cancer_key: The cancer target key (e.g., "liver", "lung")
        language: Language for organ names ("vi" or "en")

    Returns:
        List of organ highlight configurations
    """
    organs = CANCER_TO_ORGANS.get(cancer_key, [])
    organ_names = ORGAN_NAMES_VI if language == "vi" else ORGAN_NAMES_EN

    highlights = []
    for organ in organs:
        display_name = organ_names.get(organ, organ)
        highlights.append({
            "name": display_name,
            "color": "#FF3333",  # Red highlight for cancer
            "opacity": 0.85,
        })

    return highlights


def get_cancer_label(cancer_key: str, language: str = "vi") -> str:
    """Get cancer label in specified language."""
    labels = CANCER_LABELS_VI if language == "vi" else CANCER_LABELS_EN
    return labels.get(cancer_key, cancer_key)


def get_full_mapping(cancer_key: str, language: str = "vi") -> CancerOrganMapping:
    """
    Get complete mapping for API response.

    Returns:
        Dict with target, label, and organs_to_highlight
    """
    return {
        "target": cancer_key,
        "label": get_cancer_label(cancer_key, language),
        "organs_to_highlight": get_organ_highlights(cancer_key, language),
    }


def get_available_targets() -> list[str]:
    """Get list of available cancer target keys."""
    return list(CANCER_TO_ORGANS.keys())


def validate_target(cancer_key: str) -> bool:
    """Check if a cancer target is valid."""
    return cancer_key in CANCER_TO_ORGANS