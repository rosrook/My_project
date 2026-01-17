"""
CLI entry for failure_key_sampler.
"""

from __future__ import annotations

from .sample_failure_keys import sample_failure_keys
from .pipeline_output_builder import write_error_output


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
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

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
        )


if __name__ == "__main__":
    main()
