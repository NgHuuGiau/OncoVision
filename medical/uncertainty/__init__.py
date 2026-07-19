"""Uncertainty quantification for medical image classification."""

from medical.uncertainty.quantification import (
    DeepEnsembleUncertainty,
    MCDropoutUncertainty,
    TemperatureScaling,
    UncertaintyResult,
    compute_ece,
)

__all__ = [
    "UncertaintyResult",
    "MCDropoutUncertainty",
    "DeepEnsembleUncertainty",
    "TemperatureScaling",
    "compute_ece",
]
