"""Standalone Spatial Reasoning Layer (Layer 2) with Integrated CCDM Dynamics Bridge."""

from .coordinate_transformer import CoordinateTransformer
from .core import DEFAULT_CONFIG_PATH, SpatialReasoningLayer, SpatialResult, load_config
from .detection_processor import Detection, FrameDetections, ObjectDetectionProcessor, Track
from .feature_encoder import FEATURE_DIM, FEATURE_LAYOUT, FLAG_NAMES, FeatureEncoder
from .spatial_analyzer import IKResult, SpatialAnalysis, SpatialAnalyzer
from .state_manager import MuJoCoStateManager, RobotStateSnapshot
from .multi_agent_interface import (
    MultiAgentInterface,
    MultiAgentTaskPlan,
    TargetSpecification,
    PlaceSpecification
)
from .integrated_pipeline import (
    IntegratedSpatialCCDMPipeline,
    IntegratedPipelineOutput
)

__version__ = "2.0.0"

__all__ = [
    "CoordinateTransformer",
    "DEFAULT_CONFIG_PATH",
    "Detection",
    "FEATURE_DIM",
    "FEATURE_LAYOUT",
    "FLAG_NAMES",
    "FeatureEncoder",
    "FrameDetections",
    "IKResult",
    "MuJoCoStateManager",
    "ObjectDetectionProcessor",
    "RobotStateSnapshot",
    "SpatialAnalysis",
    "SpatialAnalyzer",
    "SpatialReasoningLayer",
    "SpatialResult",
    "Track",
    "MultiAgentInterface",
    "MultiAgentTaskPlan",
    "TargetSpecification",
    "PlaceSpecification",
    "IntegratedSpatialCCDMPipeline",
    "IntegratedPipelineOutput",
    "load_config",
]


