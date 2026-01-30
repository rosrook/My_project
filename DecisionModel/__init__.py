"""
DecisionModel: 筛选要素抽象、监督构造、因子预测、管线路由验证的闭环 demo。
"""

from .data import (
    ImageFilterSample,
    FactorPrimitive,
    AbstractionResult,
    LabeledSample,
    PrimitiveSpec,
)
from .filter_factor_abstraction import (
    TextEncoder,
    run_abstraction_pipeline,
)
from .label_construction import build_labeled_samples
from .factor_prediction import FactorPredictionModel, VisionEncoder, multi_label_bce_loss
from .routing_sanity_check import Router, run_sanity_check, SanityCheckReport

__all__ = [
    "ImageFilterSample",
    "FactorPrimitive",
    "AbstractionResult",
    "LabeledSample",
    "PrimitiveSpec",
    "TextEncoder",
    "run_abstraction_pipeline",
    "build_labeled_samples",
    "FactorPredictionModel",
    "VisionEncoder",
    "multi_label_bce_loss",
    "Router",
    "run_sanity_check",
    "SanityCheckReport",
]
