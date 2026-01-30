"""
轻量级因子预测模型：视觉编码器 + 多标签预测头（线性层或小型 MLP）。
"""

from __future__ import annotations

from typing import Dict, Optional

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

from .vision_encoder import VisionEncoder


class FactorPredictionModel(nn.Module if HAS_TORCH else object):  # type: ignore[misc]
    """
    图像 → 视觉特征 → 多标签预测头 → factor_id 对数概率（logits）。
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        num_factors: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("torch is required for FactorPredictionModel")
        super().__init__()
        self.vision_encoder = vision_encoder
        feat_dim = vision_encoder.feature_dim
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.factor_head = nn.Linear(hidden_dim, num_factors)

    def forward(self, images: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        """
        images: (B, 3, H, W)
        返回 {"factor_logits": (B, num_factors)}
        """
        feats = self.vision_encoder(images)
        h = self.head(feats)
        logits = self.factor_head(h)
        return {"factor_logits": logits}

    def predict_probs(self, images: "torch.Tensor") -> "torch.Tensor":
        """返回 (B, num_factors) 的 sigmoid 概率。"""
        out = self.forward(images)
        return torch.sigmoid(out["factor_logits"])
