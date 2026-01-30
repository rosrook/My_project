"""
因子预测模型：冻结/部分冻结视觉编码器 + 多标签预测头，学习 image features → factor_id 概率。
"""

from .vision_encoder import VisionEncoder
from .model import FactorPredictionModel
from .loss import multi_label_bce_loss

__all__ = [
    "VisionEncoder",
    "FactorPredictionModel",
    "multi_label_bce_loss",
]
