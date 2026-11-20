"""Medical Pipeline Package.

Tách từ pipeline.py (2,661 dòng) thành các module nhỏ hơn để dễ bảo trì.
Cấu trúc:
    medical/pipeline/
    ├── __init__.py          # Re-export tất cả public symbols
    ├── helpers.py           # Helper functions
    ├── dataclasses.py       # Data classes & protocols
    ├── detection.py         # Detection helpers
    ├── config.py            # Config builders & validation
    └── analyzer.py          # MedicalImageAnalyzer class
"""
from __future__ import annotations

from medical.pipeline.dataclasses import (
    DetectionFinding,
    DetectorBackend,
    MedicalAnalysisResult,
    MedicalImageAnalyzerConfig,
    PipelineStageResult,
)
from medical.pipeline.helpers import (
    _canonical_body_region_key,
    _sanitize_filename_part,
    _validate_output_path,
)
from medical.pipeline.detection import (
    _coerce_detections,
    _decode_raw_image,
    _map_detections_to_canvas,
)
from medical.pipeline.config import (
    build_default_medical_analyzer_config,
    validate_medical_analyzer_config,
)
from medical.pipeline.analyzer import MedicalImageAnalyzer

# Type alias
from collections.abc import Callable
ProgressCallback = Callable[[str, float], None] | None

__all__ = [
    "DetectionFinding",
    "DetectorBackend",
    "MedicalAnalysisResult",
    "MedicalImageAnalyzerConfig",
    "PipelineStageResult",
    "MedicalImageAnalyzer",
    "ProgressCallback",
    "build_default_medical_analyzer_config",
    "validate_medical_analyzer_config",
    "_sanitize_filename_part",
    "_validate_output_path",
    "_canonical_body_region_key",
    "_coerce_detections",
    "_decode_raw_image",
    "_map_detections_to_canvas",
]
