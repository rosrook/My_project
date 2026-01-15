"""
Decision model definition.

This model predicts:
1) failure reasons (multi-label)
2) filtering factors (multi-label)

The backbone can be replaced by:
- a lightweight CNN
- CLIP encoder + MLP
- any vision feature extractor
"""

from __future__ import annotations

from typing import Dict, Optional

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TORCH = False
    torch = None
    nn = None


class DecisionModel(nn.Module):  # type: ignore[misc]
    """
    Simple multi-head classifier for failure reasons and filtering factors.
    """

    def __init__(
        self,
        image_embed_dim: int,
        num_failure_reasons: int,
        num_factors: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(image_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.failure_head = nn.Linear(hidden_dim, num_failure_reasons)
        self.factor_head = nn.Linear(hidden_dim, num_factors)

    def forward(self, image_embeddings: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        x = self.backbone(image_embeddings)
        x = self.head(x)
        return {
            "failure_logits": self.failure_head(x),
            "factor_logits": self.factor_head(x),
        }


def build_model(
    image_embed_dim: int,
    num_failure_reasons: int,
    num_factors: int,
    hidden_dim: int = 256,
) -> "DecisionModel":
    if not HAS_TORCH:
        raise ImportError("torch is required to build the decision model")
    return DecisionModel(
        image_embed_dim=image_embed_dim,
        num_failure_reasons=num_failure_reasons,
        num_factors=num_factors,
        hidden_dim=hidden_dim,
    )
