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
import hashlib
import random
from pathlib import Path
from typing import Optional

try:
    from tqdm.asyncio import tqdm as atqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    atqdm = None

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
import time


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
    
    # Estimate average claims per image (sample from config if possible)
    # Default to 10, will be refined if template generator provides this info
    avg_claims_per_image = 10
    try:
        import json
        with open(claim_template_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
            if isinstance(template_data, dict):
                templates = template_data.get("templates", [])
                if templates:
                    avg_claims_per_image = len(templates)
    except:
        pass  # Use default if cannot estimate
    
    # Auto-optimize concurrency based on data size
    from ProbingFactorGeneration.config import calculate_optimal_concurrency, estimate_processing_time
    
    data_size = len(image_paths)
    baseline_optimal_concurrency = calculate_optimal_concurrency(
        data_size=data_size,
        is_local_model=use_local_baseline,
        avg_claims_per_image=avg_claims_per_image
    )
    judge_optimal_concurrency = calculate_optimal_concurrency(
        data_size=data_size,
        is_local_model=False,  # Judge is typically API
        avg_claims_per_image=avg_claims_per_image
    )
    
    # Initialize models with auto-optimized settings (pass None to enable auto-optimization)
    if use_local_baseline and baseline_model_path:
        baseline_model = BaselineModel(
            model_path=baseline_model_path,
            use_local_model=True,
            gpu_id=0,
            max_concurrent=None,  # Auto-optimize
            request_delay=0.0
        )
    else:
        baseline_model = BaselineModel(
            model_name="gemini-pro-vision",
            max_concurrent=None,  # Auto-optimize
            request_delay=0.0
        )
    
    # Apply optimization
    baseline_model.optimize_concurrency_for_data_size(data_size, avg_claims_per_image)
    
    if judge_model_name:
        judge_model = JudgeModel(
            model_name=judge_model_name,
            max_concurrent=None,  # Auto-optimize
            use_lb_client=True,
            request_delay=0.0
        )
    else:
        judge_model = JudgeModel(
            model_name="gemini-pro-vision",
            max_concurrent=None,  # Auto-optimize
            request_delay=0.0
        )
    
    # Apply optimization
    judge_model.optimize_concurrency_for_data_size(data_size, avg_claims_per_image)
    
    # Estimate processing time
    time_estimate = estimate_processing_time(
        num_images=data_size,
        avg_claims_per_image=avg_claims_per_image,
        baseline_max_concurrent=baseline_model.max_concurrent,
        judge_max_concurrent=judge_model.max_concurrent,
        is_local_baseline=use_local_baseline,
        is_local_judge=False
    )
    
    print(f"\n{'='*80}")
    print(f"Processing Configuration")
    print(f"{'='*80}")
    print(f"Total images: {data_size}")
    print(f"Average claims per image: {avg_claims_per_image}")
    print(f"Baseline model concurrency: {baseline_model.max_concurrent}")
    print(f"Judge model concurrency: {judge_model.max_concurrent}")
    print(f"\nTime Estimate:")
    print(f"  Baseline stage: {time_estimate['baseline_time_minutes']:.1f} minutes ({time_estimate['baseline_time_seconds']:.0f}s)")
    print(f"  Judge stage: {time_estimate['judge_time_minutes']:.1f} minutes ({time_estimate['judge_time_seconds']:.0f}s)")
    print(f"  Total estimated time: {time_estimate['total_time_hours']:.2f} hours ({time_estimate['total_time_minutes']:.1f} minutes)")
    print(f"  Estimated throughput: {time_estimate['estimated_throughput']:.1f} images/hour")
    print(f"{'='*80}\n")
    
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
            results = await pipeline.process_batch_with_templates_async(image_paths, show_progress=True)
            
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


async def run_pipeline_with_failure_sampling(
    parquet_dir: str,
    target_failure_count: int,
    batch_size: int = 50,
    output_dir: str = "./output",
    baseline_model_path: str = None,
    judge_model_name: str = None,
    claim_template_config: str = "configs/claim_template.example_v1_1.json",
    use_local_baseline: bool = False,
    random_seed: int = 42,
    include_source_metadata: bool = False,
    max_empty_batches: int = None,
    parquet_sample_size: Optional[int] = None
):
    """
    运行 pipeline，持续采样直到收集到指定数量的有failure的图片。
    
    Args:
        parquet_dir: Parquet 文件目录路径
        target_failure_count: 目标的有failure的图片数量
        batch_size: 每次处理的批次大小
        output_dir: 输出目录
        baseline_model_path: Baseline 模型路径（本地 LLaVA 模型）
        judge_model_name: Judge 模型名称（Qwen 模型名称）
        claim_template_config: Claim template 配置文件路径
        use_local_baseline: 是否使用本地 baseline 模型
        random_seed: 随机种子
        include_source_metadata: 是否包含源元数据
        max_empty_batches: 最大连续空batch数量，如果经过n个batch仍没有找到错误案例，则终止程序
        parquet_sample_size: Parquet 文件采样数量（None = 使用全部 parquet 文件）
    """
    # Detect torchrun/distributed context (no hard dependency on torch)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # Manually bind GPU if torchrun is used and torch is available
    if world_size > 1:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
        except Exception:
            pass
    
    # Use per-rank output dir to avoid write conflicts under torchrun
    output_dir_path = Path(output_dir)
    if world_size > 1:
        output_dir_path = output_dir_path / f"rank_{rank}"
    
    # Initialize ImageLoader (load all parquet files for sampling)
    # In distributed mode, sample after sharding to avoid empty shards.
    per_rank_parquet_sample = None
    if world_size <= 1:
        per_rank_parquet_sample = parquet_sample_size
    image_loader = ImageLoader(
        parquet_dir=parquet_dir,
        sample_size=None,  # Don't limit initial sampling
        parquet_sample_size=per_rank_parquet_sample,
        random_seed=random_seed,
        lazy_load=True
    )
    
    # Resolve config path
    claim_template_path = Path(claim_template_config)
    if not claim_template_path.is_absolute():
        claim_template_path = PROBING_ROOT / claim_template_config
    
    if not claim_template_path.exists():
        print(f"Claim template config not found: {claim_template_path}")
        return
    
    template_generator = TemplateClaimGenerator(config_path=str(claim_template_path))
    
    # Estimate average claims per image
    avg_claims_per_image = 10
    try:
        import json
        with open(claim_template_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
            if isinstance(template_data, dict):
                templates = template_data.get("templates", [])
                if templates:
                    avg_claims_per_image = len(templates)
    except:
        pass
    
    # Initialize models with auto-optimized settings
    if use_local_baseline and baseline_model_path:
        baseline_model = BaselineModel(
            model_path=baseline_model_path,
            use_local_model=True,
            gpu_id=local_rank if world_size > 1 else 0,
            max_concurrent=None,
            request_delay=0.0
        )
    else:
        baseline_model = BaselineModel(
            model_name="gemini-pro-vision",
            max_concurrent=None,
            request_delay=0.0
        )
    
    baseline_model.optimize_concurrency_for_data_size(batch_size, avg_claims_per_image)
    
    if judge_model_name:
        judge_model = JudgeModel(
            model_name=judge_model_name,
            max_concurrent=None,
            use_lb_client=True,
            request_delay=0.0
        )
    else:
        judge_model = JudgeModel(
            model_name="gemini-pro-vision",
            max_concurrent=None,
            request_delay=0.0
        )
    
    judge_model.optimize_concurrency_for_data_size(batch_size, avg_claims_per_image)
    
    failure_aggregator = FailureAggregator()
    filtering_factor_mapper = FilteringFactorMapper()
    data_saver = DataSaver(output_dir=output_dir_path)
    
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
    
    # Build deterministic shard of parquet files, then incrementally load images
    if world_size > 1:
        parquet_files = image_loader._discover_parquet_files()
    else:
        parquet_files = image_loader.get_parquet_files()
    if not parquet_files:
        print("No parquet files found!")
        return
    
    if world_size > 1:
        def _stable_hash(value: str) -> int:
            return int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)
        parquet_files = [
            p for p in parquet_files
            if _stable_hash(str(p)) % world_size == rank
        ]
        if parquet_sample_size is not None and parquet_sample_size < len(parquet_files):
            parquet_files = random.sample(parquet_files, parquet_sample_size)
            print(f"Sampled {parquet_sample_size} parquet files from shard {rank}/{world_size}")
        print(f"Shard {rank}/{world_size}: {len(parquet_files)} parquet files")
    
    rng = random.Random(random_seed + rank * 1000)
    rng.shuffle(parquet_files)
    parquet_index = 0
    
    processed_paths = set()
    all_image_paths = []
    queued_paths = set()
    
    def _enqueue_records(records):
        for record in records:
            if 'image_path' in record:
                path = record['image_path']
            elif 'image_bytes' in record:
                image_id = record.get('image_id', f"image_{len(queued_paths)}")
                path = f"<bytes:{image_id}>"
            else:
                continue
            if path not in processed_paths and path not in queued_paths:
                queued_paths.add(path)
                all_image_paths.append(path)
    
    def _load_next_parquet_batch() -> int:
        nonlocal parquet_index
        if parquet_index >= len(parquet_files):
            return 0
        if parquet_sample_size is None:
            batch_files = parquet_files[parquet_index:]
        else:
            batch_files = parquet_files[parquet_index:parquet_index + parquet_sample_size]
        parquet_index += len(batch_files)
        records = image_loader.load_parquet_files(batch_files)
        before = len(all_image_paths)
        _enqueue_records(records)
        return len(all_image_paths) - before
    
    # Initial load (may be incremental if parquet_sample_size is set)
    while not all_image_paths:
        added = _load_next_parquet_batch()
        if added <= 0:
            break
    
    if not all_image_paths:
        print("No images found!")
        return
    
    # Shuffle once, then iterate sequentially to avoid repeated random sampling
    rng.shuffle(all_image_paths)
    shard_index = 0
    
    # Track processed images and collected failures
    failure_results = []
    total_processed = 0
    batch_num = 0
    consecutive_empty_batches = 0  # Track consecutive batches without failures
    first_result = None  # Store first processed result (regardless of failure status)
    first_result_path = None  # Store path of first image
    first_result_saved = False  # Flag to track if first result has been saved
    crash_protection_triggered = False  # Flag to indicate if crash protection was triggered
    
    def _unique_path(base_path: Path) -> Path:
        if not base_path.exists():
            return base_path
        suffix = 1
        while True:
            candidate = base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")
            if not candidate.exists():
                return candidate
            suffix += 1
    
    def _save_failure_case(result: dict, image_path: str):
        failure_dir = output_dir_path / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        image_id = result.get('image_id') or image_loader.get_image_id(image_path)
        
        # Save JSON result immediately
        import json
        json_path = _unique_path(failure_dir / f"{image_id}_result.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"\n✗ Error: Could not save failure JSON for {image_id}: {e}")
            return
        
        # Try to save image
        try:
            image = image_loader.load(image_path)
            image_path_out = _unique_path(failure_dir / f"{image_id}.jpg")
            image.save(image_path_out, 'JPEG', quality=95)
        except Exception as e:
            print(f"\n⚠ Warning: Could not save failure image for {image_id}: {e}")
    
    print(f"\n{'='*80}")
    print(f"Failure-based Sampling Pipeline")
    print(f"{'='*80}")
    print(f"Target failure count: {target_failure_count}")
    print(f"Batch size: {batch_size}")
    if max_empty_batches is not None:
        print(f"Max empty batches: {max_empty_batches}")
    if world_size > 1:
        print(f"Distributed mode: rank {rank}/{world_size} (local_rank={local_rank})")
    print(f"{'='*80}\n")
    
    try:
        async with baseline_model, judge_model:
            start_time = time.time()
            
            # Create progress bar for failure collection
            if HAS_TQDM:
                from tqdm import tqdm
                failure_pbar = tqdm(
                    total=target_failure_count,
                    desc="Collecting failures",
                    unit="failure",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} failures [{elapsed}<{remaining}, {rate_fmt}]'
                )
            else:
                failure_pbar = None
            
            try:
                while len(failure_results) < target_failure_count:
                    batch_num += 1
                    
                    # Sample a batch of images without replacement (from shuffled shard)
                    if shard_index >= len(all_image_paths):
                        # Try loading more parquet files (incremental sampling)
                        while parquet_index < len(parquet_files):
                            added = _load_next_parquet_batch()
                            if added > 0:
                                # Shuffle only the newly added slice
                                start = len(all_image_paths) - added
                                new_slice = all_image_paths[start:]
                                rng.shuffle(new_slice)
                                all_image_paths[start:] = new_slice
                                break
                        if shard_index >= len(all_image_paths):
                            batch_paths = []
                        else:
                            batch_paths = all_image_paths[shard_index:shard_index + batch_size]
                            shard_index += len(batch_paths)
                    else:
                        batch_paths = all_image_paths[shard_index:shard_index + batch_size]
                        shard_index += len(batch_paths)
                    
                    if not batch_paths:
                        if failure_pbar:
                            failure_pbar.close()
                        print(f"\nNo more images available. Collected {len(failure_results)}/{target_failure_count} failure images.")
                        print(f"Total processed: {total_processed} images")
                        print(f"Processed paths count: {len(processed_paths)}")
                        break
                    
                    # Update progress bar description
                    if failure_pbar:
                        failure_pbar.set_description(f"Batch {batch_num}: Processing {len(batch_paths)} images")
                        failure_pbar.set_postfix({
                            'collected': len(failure_results),
                            'processed': total_processed,
                            'batch': batch_num
                        })
                    else:
                        print(f"\nBatch {batch_num}: Processing {len(batch_paths)} images...")
                        print(f"  Already collected: {len(failure_results)}/{target_failure_count} failure images")
                        print(f"  Total processed: {total_processed}")
                    
                    # Process batch (with progress bar)
                    batch_results = await pipeline.process_batch_with_templates_async(batch_paths, show_progress=True)
                    
                    # Filter results with failures
                    batch_failures = []
                    for idx, result in enumerate(batch_results):
                        total_processed += 1
                        image_id = result.get('image_id', '')
                        image_path = batch_paths[idx] if idx < len(batch_paths) else None
                        
                        # Save first result immediately after processing (if not already saved)
                        if first_result is None and image_path:
                            first_result = result
                            first_result_path = image_path
                            
                            # Save first image's jpg and result immediately (regardless of failure status)
                            if not first_result_saved:
                                import json
                                first_image_id = result.get('image_id', 'first_image')
                                first_result_json_path = output_dir_path / f"{first_image_id}_result.json"
                                
                                # Always save JSON result (even if image save fails)
                                try:
                                    with open(first_result_json_path, 'w', encoding='utf-8') as f:
                                        json.dump(result, f, indent=2, ensure_ascii=False)
                                    json_saved = True
                                except Exception as e:
                                    json_saved = False
                                    print(f"\n✗ Error: Could not save first result JSON: {e}")
                                
                                # Try to save image (may fail for bytes format)
                                image_saved = False
                                first_image_jpg_path = None
                                if json_saved:
                                    try:
                                        first_image = image_loader.load(image_path)
                                        first_image_jpg_path = output_dir_path / f"{first_image_id}.jpg"
                                        first_image.save(first_image_jpg_path, 'JPEG', quality=95)
                                        image_saved = True
                                    except Exception as e:
                                        print(f"\n⚠ Warning: Could not save first processed image (image data may not be available): {e}")
                                        if failure_pbar:
                                            failure_pbar.write(f"  Warning: Could not save first image: {e}")
                                
                                if json_saved:
                                    first_result_saved = True
                                    print(f"\n✓ First processed result saved immediately:")
                                    if image_saved:
                                        print(f"  Image: {first_image_jpg_path}")
                                    else:
                                        print(f"  Image: Not saved (image data not available)")
                                    print(f"  Result: {first_result_json_path}")
                                    if failure_pbar:
                                        if image_saved:
                                            failure_pbar.write(f"\n✓ First processed image saved: {first_image_id}.jpg and {first_image_id}_result.json")
                                        else:
                                            failure_pbar.write(f"\n✓ First processed result saved: {first_image_id}_result.json (image not saved)")
                        
                        # Check if this image has failures
                        aggregated_failures = result.get('aggregated_failures', {})
                        failed_claims = aggregated_failures.get('failed_claims', 0)
                        
                        if failed_claims > 0:
                            batch_failures.append(result)
                            failure_results.append(result)
                            
                            # Save failure case immediately
                            if image_path:
                                _save_failure_case(result, image_path)
                            
                            # Update progress bar
                            if failure_pbar:
                                failure_pbar.update(1)
                        
                        # Mark as processed
                        if image_path:
                            processed_paths.add(image_path)
                    
                    # Update consecutive empty batches counter
                    if len(batch_failures) == 0:
                        consecutive_empty_batches += 1
                    else:
                        consecutive_empty_batches = 0  # Reset counter when failures found
                    
                    if not failure_pbar:
                        print(f"  Found {len(batch_failures)} images with failures in this batch")
                        print(f"  Progress: {len(failure_results)}/{target_failure_count} failure images collected")
                        if max_empty_batches is not None:
                            print(f"  Consecutive empty batches: {consecutive_empty_batches}/{max_empty_batches}")
                    
                    # Check crash protection: max_empty_batches reached
                    if max_empty_batches is not None and consecutive_empty_batches >= max_empty_batches:
                        crash_protection_triggered = True
                        if failure_pbar:
                            failure_pbar.close()
                        print(f"\n⚠️  Crash protection triggered: {consecutive_empty_batches} consecutive batches without failures.")
                        print(f"   Terminating program. Collected {len(failure_results)}/{target_failure_count} failure images.")
                        
                        # First result should already be saved (if it was processed)
                        if first_result_saved:
                            print(f"   First processed case was already saved.")
                        elif first_result is not None and first_result_path:
                            # Save first result if not already saved (shouldn't happen, but just in case)
                            try:
                                import json
                                first_image = image_loader.load(first_result_path)
                                first_image_id = first_result.get('image_id', 'first_image')
                                first_image_jpg_path = output_dir_path / f"{first_image_id}.jpg"
                                first_image.save(first_image_jpg_path, 'JPEG', quality=95)
                                
                                first_result_json_path = output_dir_path / f"{first_image_id}_result.json"
                                with open(first_result_json_path, 'w', encoding='utf-8') as f:
                                    json.dump(first_result, f, indent=2, ensure_ascii=False)
                                print(f"   Saved first processed case: {first_image_id}.jpg and {first_image_id}_result.json")
                            except Exception as e:
                                print(f"   Warning: Could not save first processed case: {e}")
                        
                        break
                    
                    # Check if we've reached the target
                    if len(failure_results) >= target_failure_count:
                        if failure_pbar:
                            failure_pbar.close()
                        print(f"\n✓ Reached target! Collected {len(failure_results)} failure images.")
                        break
            finally:
                if failure_pbar:
                    failure_pbar.close()
            
            elapsed_time = time.time() - start_time
            
            # Save all failure results (only if crash protection was not triggered)
            if crash_protection_triggered:
                print(f"\n{'='*80}")
                print(f"Sampling Terminated (Crash Protection)")
                print(f"{'='*80}")
                print(f"Total images processed: {total_processed}")
                print(f"Consecutive empty batches: {consecutive_empty_batches}")
                print(f"Only first processed case was saved.")
                print(f"Time elapsed: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f}s)")
                print(f"{'='*80}\n")
            elif failure_results:
                output_path = pipeline.data_saver.save_results(
                    failure_results,
                    "probing_results_failures",
                    "json"
                )
                print(f"\n{'='*80}")
                print(f"Sampling Complete")
                print(f"{'='*80}")
                print(f"Total images processed: {total_processed}")
                print(f"Failure images collected: {len(failure_results)}")
                print(f"Success rate: {len(failure_results)/total_processed*100:.2f}%" if total_processed > 0 else "N/A")
                print(f"Time elapsed: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f}s)")
                print(f"Results saved to: {output_path}")
                print(f"{'='*80}\n")
            else:
                print(f"\nNo failure images found after processing {total_processed} images.")
            
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
    
    parser.add_argument(
        "--target_failure_count",
        type=int,
        default=None,
        help="Target number of images with failures to collect. If set, will continuously sample and process until this many failure images are collected. (New feature, does not override --sample_size)"
    )
    
    parser.add_argument(
        "--failure_batch_size",
        type=int,
        default=50,
        help="Batch size for failure-based sampling (used with --target_failure_count)"
    )
    
    parser.add_argument(
        "--max_empty_batches",
        type=int,
        default=None,
        help="Maximum number of consecutive batches without failures before termination. If set, program will terminate after N batches without finding any failures, and only save the first processed case."
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.use_local_baseline and not args.baseline_model_path:
        parser.error("--use_local_baseline requires --baseline_model_path")
    
    # 选择运行模式
    if args.target_failure_count is not None:
        # 运行 failure-based sampling 模式
        asyncio.run(run_pipeline_with_failure_sampling(
            parquet_dir=args.parquet_dir,
            target_failure_count=args.target_failure_count,
            batch_size=args.failure_batch_size,
            output_dir=args.output_dir,
            baseline_model_path=args.baseline_model_path,
            judge_model_name=args.judge_model_name,
            claim_template_config=args.claim_template_config,
            use_local_baseline=args.use_local_baseline,
            random_seed=args.random_seed,
            include_source_metadata=args.include_source_metadata,
            max_empty_batches=args.max_empty_batches,
            parquet_sample_size=args.parquet_sample_size
        ))
    else:
        # 运行原有的固定采样模式
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
