"""
CLI entry for Model_orient routing inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False
    torch = None

from .data_loader import load_config
from .decision_model import build_model
from .inference import load_model, run_inference


def _load_image_ids(jsonl_path: str) -> List[str]:
    image_ids: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            image_ids.append(item["image_id"])
    return image_ids


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run decision-model routing inference")
    parser.add_argument("--config", required=True, help="config.json path")
    parser.add_argument("--embeddings", required=True, help="torch tensor path (.pt)")
    parser.add_argument("--image_ids", required=True, help="JSONL with image_id per line")
    parser.add_argument("--model", required=True, help="model checkpoint path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not HAS_TORCH:
        raise ImportError("torch is required for Model_orient inference")

    failure_reasons, factors, _ = load_config(args.config)
    image_ids = _load_image_ids(args.image_ids)

    embeddings = torch.load(args.embeddings, map_location="cpu")
    if embeddings.shape[0] != len(image_ids):
        raise ValueError("Embeddings count does not match image_ids count")

    model = build_model(
        image_embed_dim=embeddings.shape[1],
        num_failure_reasons=len(failure_reasons),
        num_factors=len(factors),
    )
    model = load_model(args.model, model)

    results = run_inference(
        model=model,
        image_embeddings=embeddings,
        image_ids=image_ids,
        failure_reasons=failure_reasons,
        factors=factors,
        threshold=args.threshold,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
