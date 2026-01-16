"""
Analyze failure collection output in rank*/failures folders.

Usage:
  python examples/analyze_failure_collection.py --output_dir ./output
"""

import argparse
import json
from pathlib import Path

from ProbingFactorGeneration.io import analyze_failure_collection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze failure collection output and summarize stats."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory containing rank*/failures or a failures folder.",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save stats as JSON.",
    )
    args = parser.parse_args()

    stats = analyze_failure_collection(Path(args.output_dir))
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
