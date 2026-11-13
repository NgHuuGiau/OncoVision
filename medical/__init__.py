"""OncoVision Medical Package.

Core medical image analysis pipeline for cancer screening AI.
Provides inference-only capabilities (no training) for 11+ cancer types.

Public API:
    MedicalImageAnalyzer — Main entry point for image analysis
    MedicalAnalysisResult — Structured analysis result
    DetectionFinding — Single detection with confidence and bbox
    PipelineStageResult — Per-stage pipeline status
    MedicalImageAnalyzerConfig — Configuration dataclass
"""
from __future__ import annotations

from medical.pipeline import (
    DetectionFinding,
    MedicalAnalysisResult,
    MedicalImageAnalyzer,
    MedicalImageAnalyzerConfig,
    PipelineStageResult,
    build_default_medical_analyzer_config,
    validate_medical_analyzer_config,
)

__all__ = [
    "DetectionFinding",
    "MedicalAnalysisResult",
    "MedicalImageAnalyzer",
    "MedicalImageAnalyzerConfig",
    "PipelineStageResult",
    "build_default_medical_analyzer_config",
    "validate_medical_analyzer_config",
]
