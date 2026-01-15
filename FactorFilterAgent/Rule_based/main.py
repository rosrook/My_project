"""
CLI entry for Rule-based routing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .rule_based_filter import (
    FailureReason,
    FilteringFactor,
    run_rule_based_filtering,
)


def _load_failure_config(config_path: str) -> List[FailureReason]:
    """
    Build FailureReason objects from failure_config.example.json.
    """
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    failure_reasons: List[FailureReason] = []
    for item in data.get("failure_reasons", []):
        failure_id = item.get("failure_id")
        description = item.get("description", "")
        factors = []
        for idx, text in enumerate(item.get("suggested_filtering_factors", []) or []):
            factor_id = f"{failure_id}__{idx}"
            factors.append(FilteringFactor(factor_id=factor_id, description=text))
        if failure_id:
            failure_reasons.append(
                FailureReason(
                    failure_reason_id=failure_id,
                    description=description,
                    core_filtering_factors=factors,
                )
            )
    return failure_reasons


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run rule-based routing")
    parser.add_argument("--parquet_dir", required=True, help="Parquet directory")
    parser.add_argument("--failure_config", required=True, help="Failure config JSON")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--parquet_sample_size", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--include_metadata", action="store_true")
    parser.add_argument("--top_k", type=int, default=1)
    args = parser.parse_args()

    failure_reasons = _load_failure_config(args.failure_config)
    results = run_rule_based_filtering(
        parquet_dir=args.parquet_dir,
        failure_reasons=failure_reasons,
        sample_size=args.sample_size,
        parquet_sample_size=args.parquet_sample_size,
        random_seed=args.random_seed,
        include_metadata=args.include_metadata,
        top_k=args.top_k,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "image_id": r.image_id,
                        "image_path": r.image_path,
                        "matched_failure_reasons": [
                            {
                                "failure_reason_id": m.failure_reason_id,
                                "passed_factors": [p.description for p in m.passed_factors],
                            }
                            for m in r.matched_failure_reasons
                        ],
                        "structured_payload": r.structured_payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
