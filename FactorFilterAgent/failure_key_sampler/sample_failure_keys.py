"""
Extract failure_breakdown keys from probing نتائج and sample one key per image.

Input: probing_results.json (list of image results)
Output: JSONL with one sampled failure key per image
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image

from FactorFilterAgent.factor_scoring.vlm_factor_scorer import VLMFactorScorer
from FactorFilterAgent.failure_key_sampler.pipeline_output_builder import (
    write_error_output,
    write_error_output_from_failure_root,
)


@dataclass
class SampledFailure:
    image_id: str
    sampled_failure: Optional[str]
    failure_keys: List[str]
    suggested_filtering_factors: List[str]
    image_path: Optional[str] = None


def load_failure_config(config_path: str) -> Dict[str, List[str]]:
    """
    Load failure_id -> suggested_filtering_factors mapping from config.
    """
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, List[str]] = {}
    for item in data.get("failure_reasons", []):
        failure_id = item.get("failure_id")
        factors = item.get("suggested_filtering_factors", []) or []
        if failure_id:
            mapping[failure_id] = list(factors)
    return mapping


def _load_probing_results(input_path: str) -> List[Dict[str, object]]:
    in_path = Path(input_path)
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        results = data.get("results", [])
        if isinstance(results, list):
            return results
        raise ValueError("Invalid probing_results.json: 'results' must be a list.")
    if isinstance(data, list):
        return data
    raise ValueError("Invalid probing_results.json: expected list or dict with 'results'.")


def _resolve_image_path(
    image_id: str,
    item: Dict[str, object],
    image_dir: Optional[str],
) -> Optional[str]:
    candidate_keys = ("image_path", "image_file", "path", "file_path")
    for key in candidate_keys:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value

    source_metadata = item.get("source_metadata")
    if isinstance(source_metadata, dict):
        for key in candidate_keys:
            value = source_metadata.get(key)
            if isinstance(value, str) and value:
                return value

    if not image_dir:
        return None

    base_dir = Path(image_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_id_path = Path(image_id)
    if image_id_path.suffix:
        candidate = base_dir / image_id
        if candidate.exists():
            return str(candidate)
    else:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = base_dir / f"{image_id}{ext}"
            if candidate.exists():
                return str(candidate)

    return None


def sample_failure_keys(
    input_path: str,
    output_path: str,
    failure_config_path: str,
    random_seed: Optional[int] = None,
) -> List[SampledFailure]:
    """
    Load probing_results.json and sample one failure key per image.

    If a sample has >1 failure keys, randomly choose one.
    If a sample has 1 failure key, choose it.
    If no failure keys, sampled_failure is None.
    """

    if random_seed is not None:
        random.seed(random_seed)

    out_path = Path(output_path)
    failure_factor_map = load_failure_config(failure_config_path)

    data = _load_probing_results(input_path)

    results: List[SampledFailure] = []
    for item in data:
        image_id = item.get("image_id", "")
        image_path = _resolve_image_path(image_id, item, None)
        failure_breakdown = (
            item.get("aggregated_failures", {}).get("failure_breakdown", {}) or {}
        )
        keys = [
            k
            for k in failure_breakdown.keys()
            if k not in {"null", "model_limitation"}
        ]
        if len(keys) == 0:
            sampled = None
        elif len(keys) == 1:
            sampled = keys[0]
        else:
            sampled = random.choice(keys)

        suggested_factors = failure_factor_map.get(sampled, []) if sampled else []

        results.append(
            SampledFailure(
                image_id=image_id,
                sampled_failure=sampled,
                failure_keys=keys,
                suggested_filtering_factors=suggested_factors,
                image_path=image_path,
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "image_id": r.image_id,
                        "sampled_failure": r.sampled_failure,
                        "failure_keys": r.failure_keys,
                        "suggested_filtering_factors": r.suggested_filtering_factors,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return results


async def score_sampled_failures_async(
    samples: List[SampledFailure],
    image_provider: Callable[[str], Image.Image],
    scorer: VLMFactorScorer,
) -> List[Dict[str, object]]:
    """
    Score each sampled failure using a VLM.

    image_provider: function that returns PIL.Image given image_id.
    """
    outputs: List[Dict[str, object]] = []
    for sample in samples:
        if not sample.sampled_failure:
            outputs.append(
                {
                    "image_id": sample.image_id,
                    "sampled_failure": None,
                    "score": 0.0,
                    "explanation": "No failure key",
                }
            )
            continue
        image = image_provider(sample.image_id)
        score_result = await scorer.score_async(image, sample.suggested_filtering_factors)
        outputs.append(
            {
                "image_id": sample.image_id,
                "sampled_failure": sample.sampled_failure,
                "score": score_result.get("score", 0.0),
                "explanation": score_result.get("explanation", ""),
                "details": score_result,
            }
        )
    return outputs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample failure_breakdown keys from probing_results.json"
    )
    parser.add_argument("--input", default=None, help="Path to probing_results.json")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument(
        "--failure_config",
        default=None,
        help="Path to failure_config.example.json (or compatible config)",
    )
    parser.add_argument(
        "--failure_root",
        default=None,
        help="Root dir containing rank*/failures folders (optional)",
    )
    parser.add_argument(
        "--pipeline_config",
        default=None,
        help="Path to pipeline mapping config (failure_id -> pipeline_type/name)",
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help="Directory containing images (used to build base64 jpg field)",
    )
    parser.add_argument(
        "--error_output",
        default=None,
        help="Optional output JSON file for error cases with base64 images",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for sample_index in error output",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.failure_root:
        if not args.error_output:
            raise ValueError("--error_output is required when using --failure_root")
        if not args.pipeline_config:
            raise ValueError("--pipeline_config is required when using --failure_root")
        write_error_output_from_failure_root(
            failure_root=args.failure_root,
            pipeline_config_path=args.pipeline_config,
            error_output_path=args.error_output,
            random_seed=args.seed,
            start_index=args.start_index,
        )
        return

    if not args.input or not args.output or not args.failure_config:
        raise ValueError("--input, --output, and --failure_config are required")

    samples = sample_failure_keys(
        args.input,
        args.output,
        failure_config_path=args.failure_config,
        random_seed=args.seed,
    )

    if args.error_output and args.pipeline_config:
        write_error_output(
            samples,
            pipeline_config_path=args.pipeline_config,
            error_output_path=args.error_output,
            image_dir=args.image_dir,
            start_index=args.start_index,
        )


if __name__ == "__main__":
    main()
