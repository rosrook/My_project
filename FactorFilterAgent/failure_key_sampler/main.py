"""
CLI entry for failure_key_sampler.
"""

from __future__ import annotations

from .sample_failure_keys import sample_failure_keys
from .pipeline_output_builder import (
    write_error_output,
    write_error_output_from_failure_root,
)


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
        )


if __name__ == "__main__":
    main()
