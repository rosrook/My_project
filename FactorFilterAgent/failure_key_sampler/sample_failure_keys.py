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
from typing import Dict, List, Optional


@dataclass
class SampledFailure:
    image_id: str
    sampled_failure: Optional[str]
    failure_keys: List[str]
    suggested_filtering_factors: List[str]


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

    in_path = Path(input_path)
    out_path = Path(output_path)
    failure_factor_map = load_failure_config(failure_config_path)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: List[SampledFailure] = []
    for item in data:
        image_id = item.get("image_id", "")
        failure_breakdown = (
            item.get("aggregated_failures", {}).get("failure_breakdown", {}) or {}
        )
        keys = list(failure_breakdown.keys())
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample failure_breakdown keys from probing_results.json"
    )
    parser.add_argument("--input", required=True, help="Path to probing_results.json")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--failure_config",
        required=True,
        help="Path to failure_config.example.json (or compatible config)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    sample_failure_keys(
        args.input,
        args.output,
        failure_config_path=args.failure_config,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
