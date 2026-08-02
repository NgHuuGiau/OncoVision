"""Uncertainty quantification for medical image classification."""

from medical.uncertainty.quantification import (
    DeepEnsembleUncertainty,
    MCDropoutUncertainty,
    TemperatureScaling,
    UncertaintyResult,
    compute_ece,
)

__all__ = [
    "DeepEnsembleUncertainty",
    "MCDropoutUncertainty",
    "TemperatureScaling",
    "UncertaintyResult",
    "compute_ece",
]
