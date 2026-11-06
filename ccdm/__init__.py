"""
CCDM v3.0: Context-Conditioned Dynamics Model
FiLM-modulated neural architecture for residual dynamics and uncertainty-aware control.
"""

from .models.ccdm import CCDM
from .training.losses import HeteroscedasticNLLLoss
from .utils.normalization import StandardScaler
from .interfaces.dynamics_interface import CCDMInferenceInterface
from .interfaces.spatial_grounding import SystemAgnosticSpatialGrounding
from .interfaces.context_interface import ContextInterface
from .interfaces.controller_adaptation import (
    CCDMControllerAdaptation,
    PhysicalConstraints,
    DynamicNMPCWeights
)

__version__ = "3.0.0"

__all__ = [
    "CCDM",
    "HeteroscedasticNLLLoss",
    "StandardScaler",
    "CCDMInferenceInterface",
    "SystemAgnosticSpatialGrounding",
    "ContextInterface",
    "CCDMControllerAdaptation",
    "PhysicalConstraints",
    "DynamicNMPCWeights"
]

