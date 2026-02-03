#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug runner for QA_Generator Step 3 (only).

功能：
- 读取 Step 2 的输出 JSON（FactorFilterAgent 的 error_output）
- 在其中「均匀采样」指定数量的样本（默认 10 条）
- 对每条样本单独跑一遍 QA_Generator 的 pipeline（Step 3）
- 为每条样本创建独立子目录，保存：
  - 该样本的单条输入 JSON
  - 完整的 Step 3 输出目录（questions/answers/最终结果等）
  - 详细日志（--log-file）
  - 问题生成调试信息（--debug-questions）

用法示例：

    OPENAI_API_KEY="EMPTY" OPENAI_BASE_URL="http://10.158.159.139:8000/v1" MODEL_NAME="Qwen3-VL-235B-A22B-Instruct" \\
      python run_step3_debug_samples.py \\
        --input-json /home/zhuxuzhou/My_project/FactorFilterAgent/failure_key_sampler/img_with_pipeline_type_and_prefill/third_more_30k_expanded_error_cases.json \\
        --output-root debug_step3_samples \\
        --num-samples 10 \\
        --concurrency 5 \\
        --request-delay 0.1 \\
        --batch-size 1000 \\
        --enable-validation-exemptions

说明：
- 本脚本只负责 Step 3，不会触发 Step 1 / Step 2
- 会调用 `python QA_Generator/pipeline/pipeline.py`，参数基本沿用 `run_full_pipeline.sh` 的 Step 3
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent


def uniform_sample_indices(total: int, k: int) -> List[int]:
    """在 [0, total-1] 范围内均匀采样 k 个索引（不放回），尽量覆盖全局。"""
    if total <= k:
        return list(range(total))
    # 均匀间隔采样：i 从 0..k-1，把区间 [0, total-1] 等分为 k-1 段
    return [round(i * (total - 1) / (k - 1)) for i in range(k)]


def run_for_one_sample(
    record: Dict[str, Any],
    idx: int,
    sample_idx: int,
    args: argparse.Namespace,
    project_root: Path,
) -> None:
    """为单条样本创建子目录并调用 pipeline.py 跑 Step 3。"""
    output_root: Path = args.output_root
    # 用 sample 序号 + 原始 index + 可选 id 组织目录名，便于定位
    rec_id = record.get("id") or record.get("sample_index") or record.get("source_a_id") or idx
    sample_dir = output_root / f"sample_{sample_idx:02d}_idx_{idx}_id_{rec_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 写入单条输入 JSON（pipeline.py 期望的是一个列表）
    input_path = sample_dir / "input_record.json"
    with input_path.open("w", encoding="utf-8") as f:
        json.dump([record], f, ensure_ascii=False, indent=2)

    # 准备 Step 3 命令行参数
    step3_args: List[str] = [
        sys.executable,
        str(project_root / "QA_Generator" / "pipeline" / "pipeline.py"),
        str(input_path),
        "--concurrency",
        str(args.concurrency),
        "--request-delay",
        str(args.request_delay),
        "--batch-size",
        str(args.batch_size),
    ]

    if args.enable_validation_exemptions:
        step3_args.append("--enable-validation-exemptions")
    if args.no_async:
        step3_args.append("--no-async")
    if args.no_intermediate:
        step3_args.append("--no-intermediate")
    if args.pipelines:
        step3_args.append("--pipelines")
        step3_args.extend(args.pipelines)
    if args.question_config:
        step3_args.extend(["--question-config", args.question_config])
    if args.answer_config:
        step3_args.extend(["--answer-config", args.answer_config])

    # 为每个样本单独日志文件
    log_file = sample_dir / "step3.log"
    step3_args.extend(["--log-file", str(log_file)])
    # 问题生成调试信息输出目录
    debug_q_dir = sample_dir / "debug" / "questions"
    step3_args.append("--debug-questions")
    step3_args.extend(["--debug-question-dir", str(debug_q_dir)])

    # 环境变量：沿用当前进程已有的 OPENAI_API_KEY/OPENAI_BASE_URL/MODEL_NAME 等
    env = os.environ.copy()
    if args.qa_debug:
        env["QA_DEBUG"] = "1"

    print(f"\n=== Running sample {sample_idx} (global index {idx}, id={rec_id}) ===")
    print("Working dir:", sample_dir)
    print("Command:", " ".join(step3_args))

    # 在 sample_dir 下运行，让 pipeline 的默认 output_dir 落在该目录下
    result = subprocess.run(
        step3_args,
        cwd=str(sample_dir),
        env=env,
    )
    if result.returncode != 0:
        print(f"[ERROR] sample {sample_idx} failed with exit code {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug-only runner for QA_Generator Step 3 (sampled records).")
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Step 2 的输出 JSON（FactorFilterAgent 的 error_output）",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="debug_step3_samples",
        help="调试输出根目录（每条样本一个子目录，默认: ./debug_step3_samples）",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="均匀采样的样本数（默认: 10）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="异步并发请求数（传给 QA_Generator/pipeline.py，默认: 5）",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.1,
        help="请求间隔秒数（传给 QA_Generator/pipeline.py，默认: 0.1）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Step3 批处理大小参数（此处每条样本只有 1 条记录，但沿用接口，默认: 1000）",
    )
    parser.add_argument(
        "--enable-validation-exemptions",
        action="store_true",
        help="开启验证豁免（等价于 run_full_pipeline.sh 的 ENABLE_VALIDATION_EXEMPTIONS=true）",
    )
    parser.add_argument(
        "--no-async",
        action="store_true",
        help="禁用异步并行处理（传给 --no-async）",
    )
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="不保存中间结果（questions/answers 文件）",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        nargs="+",
        default=None,
        help="仅运行指定的 pipeline 名（可选）",
    )
    parser.add_argument(
        "--question-config",
        type=str,
        default=None,
        help="问题配置文件路径（可选，传给 --question-config）",
    )
    parser.add_argument(
        "--answer-config",
        type=str,
        default=None,
        help="答案配置文件路径（可选，传给 --answer-config）",
    )
    parser.add_argument(
        "--qa-debug",
        action="store_true",
        help="设置 QA_DEBUG=1，开启 QA_Generator 内部 DEBUG 日志",
    )

    args = parser.parse_args()

    input_path = Path(args.input_json).expanduser().resolve()
    if not input_path.exists():
        print(f"[ERROR] input json not found: {input_path}")
        return 1

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    args.output_root = output_root

    # 加载 Step2 输出
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"[ERROR] input json must be a list, got {type(data)}")
        return 1

    total = len(data)
    if total == 0:
        print("[ERROR] input json is empty (0 records)")
        return 1

    k = max(1, min(args.num_samples, total))
    indices = uniform_sample_indices(total, k)

    print(f"Total records: {total}, sampling {k} records at indices: {indices}")

    for sample_idx, idx in enumerate(indices, start=1):
        record = data[idx]
        run_for_one_sample(
            record=record,
            idx=idx,
            sample_idx=sample_idx,
            args=args,
            project_root=PROJECT_ROOT,
        )

    print("\n✅ Step 3 debug run completed.")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

