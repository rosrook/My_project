"""
多标签损失：标准 BCEWithLogitsLoss，用于 factor 预测。
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


def multi_label_bce_loss(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    reduction: str = "mean",
) -> "torch.Tensor":
    """
    logits: (B, C)
    targets: (B, C) multi-hot，0/1
    返回标量 BCE 损失。
    """
    if not HAS_TORCH:
        raise ImportError("torch is required for multi_label_bce_loss")
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction=reduction
    )
