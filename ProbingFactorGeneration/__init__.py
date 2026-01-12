"""
Probing Factor Generation Framework for VQA Data Construction

A modular framework for the first stage of VQA data construction:
small-scale probing + failure → filtering factor
"""

__version__ = "0.1.0"

# Import main components
from .core import (
    ImageLoader, ClaimGenerator, PredefinedClaimGenerator, TemplateClaimGenerator,
    FailureAggregator, FilteringFactorMapper
)
from .models import BaselineModel, JudgeModel
from .io import DataSaver
from .pipeline import ProbingFactorPipeline, create_pipeline

__all__ = [
    # Core modules
    "ImageLoader",
    "ClaimGenerator",
    "PredefinedClaimGenerator",
    "TemplateClaimGenerator",
    "FailureAggregator",
    "FilteringFactorMapper",
    # Models
    "BaselineModel",
    "JudgeModel",
    # IO
    "DataSaver",
    # Pipeline
    "ProbingFactorPipeline",
    "create_pipeline",
]
