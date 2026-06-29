from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "MEDICAL_DISCLAIMER": ("medical.compliance", "MEDICAL_DISCLAIMER"),
    "CancerDatasetSourceSpec": ("medical.cancer_dataset_registry", "CancerDatasetSourceSpec"),
    "MedicalCaseDatabase": ("medical.storage", "MedicalCaseDatabase"),
    "MedicalChatResponse": ("medical.chat_service", "MedicalChatResponse"),
    "MedicalChatService": ("medical.chat_service", "MedicalChatService"),
    "MedicalDatasetConfig": ("medical.dataset", "MedicalDatasetConfig"),
    "MedicalDatasetSummary": ("medical.dataset", "MedicalDatasetSummary"),
    "MedicalImageAnalyzer": ("medical.pipeline", "MedicalImageAnalyzer"),
    "MedicalImageAnalyzerConfig": ("medical.pipeline", "MedicalImageAnalyzerConfig"),
    "MedicalMetrics": ("medical.metrics", "MedicalMetrics"),
    "build_medical_disclaimer": ("medical.compliance", "build_medical_disclaimer"),
    "common_cancer_dataset_source_dicts": ("medical.cancer_dataset_registry", "common_cancer_dataset_source_dicts"),
    "common_cancer_dataset_sources": ("medical.cancer_dataset_registry", "common_cancer_dataset_sources"),
    "compute_medical_metrics": ("medical.metrics", "compute_medical_metrics"),
    "create_default_medical_cancer_dataset_config": ("medical.dataset", "create_default_medical_cancer_dataset_config"),
    "create_default_medical_dataset_config": ("medical.dataset", "create_default_medical_dataset_config"),
    "create_default_object_detection_dataset_config": ("medical.dataset", "create_default_object_detection_dataset_config"),
    "create_default_skin_cancer_dataset_config": ("medical.dataset", "create_default_skin_cancer_dataset_config"),
    "ensure_medical_cancer_dataset_structure": ("medical.dataset", "ensure_medical_cancer_dataset_structure"),
    "ensure_medical_dataset_root_structure": ("medical.dataset", "ensure_medical_dataset_root_structure"),
    "ensure_medical_dataset_structure": ("medical.dataset", "ensure_medical_dataset_structure"),
    "ensure_object_detection_dataset_structure": ("medical.dataset", "ensure_object_detection_dataset_structure"),
    "normalize_uploaded_image": ("medical.dataset", "normalize_uploaded_image"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
