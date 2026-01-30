#!/usr/bin/env python3
"""
将 ProbingFactorGeneration 的 prefill 输出目录转为 DecisionModel demo_data 格式。

输入：prefill 根目录，例如 My_project/data/output_1_17_prefill
  - 结构：rank_0/failures/part_4_0_result.json、rank_0/failures/part_4_0.jpg
  - 每个 *_result.json 需包含字段 "suggested_filtering_factors"（字符串列表）

输出：DecisionModel demo_data 样例格式
  - annotations.jsonl：每行 {"image_path", "image_id", "filter_factor_texts"}
  - filter_factors.json：{"filter_factors_schema_version": "v1.0", "filter_factors": [...]}

用法:
  python -m DecisionModel.tools.convert_prefill_to_demo /path/to/output_1_17_prefill
  python -m DecisionModel.tools.convert_prefill_to_demo /path/to/prefill --output-dir /path/to/DecisionModel/demo_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_result_jsons(prefill_root: Path) -> list[tuple[Path, Path]]:
    """返回 [(result_json_path, jpg_path), ...]，仅当同名 .jpg 存在时包含。"""
    prefill_root = prefill_root.resolve()
    if not prefill_root.is_dir():
        return []

    pairs: list[tuple[Path, Path]] = []
    for rank_dir in sorted(prefill_root.iterdir()):
        if not rank_dir.is_dir() or not rank_dir.name.startswith("rank_"):
            continue
        failures_dir = rank_dir / "failures"
        if not failures_dir.is_dir():
            continue
        for jpath in failures_dir.glob("*_result.json"):
            image_base = jpath.stem.removesuffix("_result")
            jpg_path = failures_dir / f"{image_base}.jpg"
            if jpg_path.exists():
                pairs.append((jpath, jpg_path))
    return pairs


def load_factors(result_path: Path) -> list[str]:
    """从 *_result.json 读取 suggested_filtering_factors。"""
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    factors = data.get("suggested_filtering_factors")
    if factors is None:
        return []
    if isinstance(factors, list):
        return [str(x).strip() for x in factors if x]
    return []


def run(
    prefill_root: str | Path,
    output_dir: str | Path,
    annotations_name: str = "annotations.jsonl",
    factors_name: str = "filter_factors.json",
) -> tuple[int, Path, Path]:
    """
    扫描 prefill_root 下 rank_*/failures/*_result.json，生成 annotations.jsonl 与 filter_factors.json。
    返回 (条目数, annotations_path, filter_factors_path)。
    """
    prefill_root = Path(prefill_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_result_jsons(prefill_root)
    if not pairs:
        return 0, output_dir / annotations_name, output_dir / factors_name

    all_factors: set[str] = set()
    annotations: list[dict] = []

    for idx, (result_path, jpg_path) in enumerate(pairs):
        factors = load_factors(result_path)
        all_factors.update(factors)
        image_id = result_path.stem.removesuffix("_result")
        annotations.append({
            "image_path": str(jpg_path.resolve()),
            "image_id": image_id,
            "filter_factor_texts": factors,
        })

    annotations_path = output_dir / annotations_name
    with open(annotations_path, "w", encoding="utf-8") as f:
        for rec in annotations:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    factors_path = output_dir / factors_name
    factors_body = {
        "filter_factors_schema_version": "v1.0",
        "description": f"Converted from {prefill_root.name}; {len(annotations)} images.",
        "filter_factors": sorted(all_factors),
    }
    with open(factors_path, "w", encoding="utf-8") as f:
        json.dump(factors_body, f, indent=4, ensure_ascii=False)

    return len(annotations), annotations_path, factors_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert prefill output (rank_*/failures/*_result.json) to DecisionModel demo_data format."
    )
    parser.add_argument(
        "prefill_root",
        type=Path,
        help="Root dir, e.g. My_project/data/output_1_17_prefill",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: DecisionModel/demo_data under project root)",
    )
    parser.add_argument(
        "--annotations-name",
        default="annotations.jsonl",
        help="Output annotations filename (default: annotations.jsonl)",
    )
    parser.add_argument(
        "--factors-name",
        default="filter_factors.json",
        help="Output filter_factors filename (default: filter_factors.json)",
    )
    args = parser.parse_args()

    prefill_root = args.prefill_root.resolve()
    if not prefill_root.exists() or not prefill_root.is_dir():
        print(f"Error: prefill_root not a directory: {prefill_root}", file=sys.stderr)
        return 1

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        # Default: DecisionModel/demo_data under project root (script under DecisionModel/tools/)
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / "DecisionModel" / "demo_data"

    n, ann_path, fac_path = run(
        prefill_root,
        output_dir,
        annotations_name=args.annotations_name,
        factors_name=args.factors_name,
    )
    print(f"Wrote {n} records to {ann_path}")
    with open(fac_path, "r", encoding="utf-8") as f:
        n_factors = len(json.load(f)["filter_factors"])
    print(f"Wrote {n_factors} factors to {fac_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
