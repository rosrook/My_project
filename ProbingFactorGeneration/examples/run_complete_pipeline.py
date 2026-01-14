#!/usr/bin/env python3
"""
完整 Pipeline 运行示例

使用说明:
1. 确保已安装所有依赖: pip install -r requirements.txt
2. 配置环境变量（如需要）:
   - SERVICE_NAME, ENV, API_KEY, USE_LB_CLIENT (for judge model)
3. 运行脚本:
   python examples/run_complete_pipeline.py

或使用命令行参数:
python examples/run_complete_pipeline.py \
    --parquet_dir /mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/ \
    --sample_size 10 \
    --output_dir ./output \
    --baseline_model_path /path/to/llava/model \
    --judge_model_name /workspace/Qwen3-VL-235B-A22B-Instruct
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directories to path
# When running from My_project/, we need My_project/ in the path
# so Python can find the ProbingFactorGeneration package
current_dir = Path(__file__).resolve().parent
probing_root = current_dir.parent  # ProbingFactorGeneration directory
project_root = probing_root.parent  # My_project directory

# Add My_project to path (this allows "from ProbingFactorGeneration import ...")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also add ProbingFactorGeneration to path as fallback
if str(probing_root) not in sys.path:
    sys.path.insert(0, str(probing_root))

# Store probing_root for resolving config file paths
PROBING_ROOT = probing_root

# Now import from ProbingFactorGeneration
from ProbingFactorGeneration.core import ImageLoader, TemplateClaimGenerator, FailureAggregator, FilteringFactorMapper
from ProbingFactorGeneration.models import BaselineModel, JudgeModel
from ProbingFactorGeneration.io import DataSaver
from ProbingFactorGeneration.pipeline import ProbingFactorPipeline


async def run_pipeline(
    parquet_dir: str,
    sample_size: int = 10,
    output_dir: str = "./output",
    baseline_model_path: str = None,
    judge_model_name: str = None,
    claim_template_config: str = "configs/claim_template.example_v1_1.json",
    use_local_baseline: bool = False,
    random_seed: int = 42,
    parquet_sample_size: Optional[int] = None,
    include_source_metadata: bool = False
):
    """
    运行完整的 pipeline。
    
    Args:
        parquet_dir: Parquet 文件目录路径
        sample_size: 采样图像数量
        output_dir: 输出目录
        baseline_model_path: Baseline 模型路径（本地 LLaVA 模型）
        judge_model_name: Judge 模型名称（Qwen 模型名称）
        claim_template_config: Claim template 配置文件路径
        use_local_baseline: 是否使用本地 baseline 模型
        random_seed: 随机种子
    """
    # Auto-determine parquet_sample_size if not provided
    if parquet_sample_size is None and sample_size <= 50:
        parquet_sample_size = min(3, max(1, sample_size // 20 + 1))
    
    image_loader = ImageLoader(
        parquet_dir=parquet_dir,
        sample_size=sample_size,
        parquet_sample_size=parquet_sample_size,
        random_seed=random_seed,
        lazy_load=True
    )
    
    image_paths = image_loader.get_all_image_paths()
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    # Resolve config path
    claim_template_path = Path(claim_template_config)
    if not claim_template_path.is_absolute():
        claim_template_path = PROBING_ROOT / claim_template_config
    
    if not claim_template_path.exists():
        print(f"Claim template config not found: {claim_template_path}")
        return
    
    template_generator = TemplateClaimGenerator(config_path=str(claim_template_path))
    
    # Initialize models with optimized settings
    if use_local_baseline and baseline_model_path:
        baseline_model = BaselineModel(
            model_path=baseline_model_path,
            use_local_model=True,
            gpu_id=0,
            max_concurrent=1,
            request_delay=0.0
        )
    else:
        baseline_model = BaselineModel(
            model_name="gemini-pro-vision",
            max_concurrent=10,
            request_delay=0.0
        )
    
    if judge_model_name:
        judge_model = JudgeModel(
            model_name=judge_model_name,
            max_concurrent=10,
            use_lb_client=True,
            request_delay=0.0
        )
    else:
        judge_model = JudgeModel(
            model_name="gemini-pro-vision",
            max_concurrent=10,
            request_delay=0.0
        )
    
    failure_aggregator = FailureAggregator()
    filtering_factor_mapper = FilteringFactorMapper()
    data_saver = DataSaver(output_dir=output_dir)
    
    pipeline = ProbingFactorPipeline(
        image_loader=image_loader,
        claim_generator=template_generator,
        baseline_model=baseline_model,
        judge_model=judge_model,
        failure_aggregator=failure_aggregator,
        filtering_factor_mapper=filtering_factor_mapper,
        data_saver=data_saver,
        include_source_metadata=include_source_metadata
    )
    
    try:
        async with baseline_model, judge_model:
            results = await pipeline.process_batch_with_templates_async(image_paths)
            
            # Save first image's jpg and result for debugging
            if results:
                first_image_path = image_paths[0]
                first_image = image_loader.load(first_image_path)
                first_image_id = results[0]['image_id']
                
                # Save first image as jpg
                first_image_jpg_path = Path(output_dir) / f"{first_image_id}.jpg"
                first_image.save(first_image_jpg_path, 'JPEG', quality=95)
                
                # Save first result
                import json
                first_result_path = Path(output_dir) / f"{first_image_id}_result.json"
                with open(first_result_path, 'w', encoding='utf-8') as f:
                    json.dump(results[0], f, indent=2, ensure_ascii=False)
            
            # Save all results
            output_path = pipeline.data_saver.save_results(
                results,
                "probing_results",
                "json"
            )
            print(f"Results saved to: {output_path}")
            
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Run complete Probing Factor Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default="/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/",
        help="Directory containing parquet files"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="Number of images to sample from parquet files"
    )
    
    parser.add_argument(
        "--parquet_sample_size",
        type=int,
        default=None,
        help="Number of parquet files to randomly sample before reading (None = use all files, recommended: 1-5 for small sample_size)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--baseline_model_path",
        type=str,
        default=None,
        help="Path to local LLaVA baseline model (if using local model)"
    )
    
    parser.add_argument(
        "--judge_model_name",
        type=str,
        default=None,
        help="Judge model name (e.g., /workspace/Qwen3-VL-235B-A22B-Instruct)"
    )
    
    parser.add_argument(
        "--claim_template_config",
        type=str,
        default="configs/claim_template.example_v1_1.json",
        help="Path to claim template configuration file"
    )
    
    parser.add_argument(
        "--use_local_baseline",
        action="store_true",
        help="Use local baseline model (requires --baseline_model_path)"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    
    parser.add_argument(
        "--include_source_metadata",
        action="store_true",
        help="Include source metadata (e.g., conversations) in results (for reference only)"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.use_local_baseline and not args.baseline_model_path:
        parser.error("--use_local_baseline requires --baseline_model_path")
    
    # 运行 pipeline
    asyncio.run(run_pipeline(
        parquet_dir=args.parquet_dir,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        baseline_model_path=args.baseline_model_path,
        judge_model_name=args.judge_model_name,
        claim_template_config=args.claim_template_config,
        use_local_baseline=args.use_local_baseline,
        random_seed=args.random_seed,
        parquet_sample_size=args.parquet_sample_size,
        include_source_metadata=args.include_source_metadata
    ))


if __name__ == "__main__":
    main()
