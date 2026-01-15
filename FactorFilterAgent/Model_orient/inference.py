"""
Inference utilities for the decision model.

This module focuses on routing:
image -> candidate failure reasons + filtering factors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TORCH = False
    torch = None
    DataLoader = None

from .data_loader import FailureReasonMeta, FilteringFactorMeta
from .decision_model import DecisionModel
from FactorFilterAgent.factor_scoring.vlm_factor_scorer import VLMFactorScorer


def load_model(model_path: str, model: "DecisionModel") -> "DecisionModel":
    if not HAS_TORCH:
        raise ImportError("torch is required for inference")
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_batch(
    model: "DecisionModel",
    image_embeddings: "torch.Tensor",
    failure_reasons: Sequence[FailureReasonMeta],
    factors: Sequence[FilteringFactorMeta],
    threshold: float = 0.5,
) -> List[Dict[str, object]]:
    """
    Convert model outputs to a structured routing response.
    """
    outputs = model(image_embeddings)
    failure_probs = torch.sigmoid(outputs["failure_logits"])
    factor_probs = torch.sigmoid(outputs["factor_logits"])

    results: List[Dict[str, object]] = []
    for i in range(image_embeddings.shape[0]):
        fr_ids = [
            fr.failure_reason_id
            for j, fr in enumerate(failure_reasons)
            if failure_probs[i, j].item() >= threshold
        ]
        factor_ids = [
            fac.factor_id
            for j, fac in enumerate(factors)
            if factor_probs[i, j].item() >= threshold
        ]
        results.append(
            {
                "predicted_failure_reasons": fr_ids,
                "predicted_filtering_factors": factor_ids,
            }
        )
    return results


def run_inference(
    model: "DecisionModel",
    image_embeddings: "torch.Tensor",
    image_ids: Sequence[str],
    failure_reasons: Sequence[FailureReasonMeta],
    factors: Sequence[FilteringFactorMeta],
    threshold: float = 0.5,
) -> List[Dict[str, object]]:
    """
    Run routing inference for a batch of image embeddings.
    """
    batch_results = predict_batch(
        model=model,
        image_embeddings=image_embeddings,
        failure_reasons=failure_reasons,
        factors=factors,
        threshold=threshold,
    )
    outputs: List[Dict[str, object]] = []
    for image_id, payload in zip(image_ids, batch_results):
        outputs.append(
            {
                "image_id": image_id,
                **payload,
            }
        )
    return outputs


async def score_suggested_factors_async(
    image: "Any",
    suggested_factors: List[str],
    scorer: VLMFactorScorer,
) -> Dict[str, Any]:
    """
    Score suggested filtering factors for a single image.
    """
    return await scorer.score_async(image, suggested_factors)
