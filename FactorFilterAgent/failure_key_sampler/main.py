"""
CLI entry for failure_key_sampler.
"""

from __future__ import annotations

from .sample_failure_keys import sample_failure_keys


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
