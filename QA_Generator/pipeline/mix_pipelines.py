#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline数据调配脚本

根据指定的比例混合不同pipeline类型的数据，生成最终的VQA数据集。

用法:
    python QA_Generator/pipeline/mix_pipelines.py \\
        --input-dir <输入目录> \\
        --ratios object_absence:0.3 object_counting:0.2 question:0.5 \\
        --output <输出文件.json>

    # 从meta.json自动读取pipeline文件
    python QA_Generator/pipeline/mix_pipelines.py \\
        --meta <meta.json> \\
        --ratios object_absence:0.3 object_counting:0.2 question:0.5 \\
        --output <输出文件.json>

    # 使用配置文件
    python QA_Generator/pipeline/mix_pipelines.py \\
        --config <config.json> \\
        --output <输出文件.json>
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "data" in data:
        return data["data"]
    else:
        raise ValueError(f"无法解析JSON文件格式: {file_path}")


def parse_ratios(ratios_str: List[str]) -> Dict[str, float]:
    """
    解析比例字符串列表
    
    格式: ["pipeline1:0.3", "pipeline2:0.2", ...]
    """
    ratios = {}
    for ratio_str in ratios_str:
        if ':' not in ratio_str:
            raise ValueError(f"比例格式错误，应为 'pipeline:ratio': {ratio_str}")
        parts = ratio_str.split(':', 1)
        pipeline_name = parts[0].strip()
        try:
            ratio = float(parts[1].strip())
            if ratio < 0:
                raise ValueError(f"比例必须 >= 0: {ratio}")
            ratios[pipeline_name] = ratio
        except ValueError as e:
            raise ValueError(f"无法解析比例值: {parts[1]}") from e
    return ratios


def normalize_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
    """归一化比例，使总和为1.0"""
    total = sum(ratios.values())
    if total == 0:
        raise ValueError("所有比例的总和不能为0")
    return {k: v / total for k, v in ratios.items()}


def sample_by_ratio(
    pipeline_data: Dict[str, List[Dict[str, Any]]],
    ratios: Dict[str, float],
    total_samples: Optional[int] = None,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    按比例采样数据
    
    Args:
        pipeline_data: {pipeline_name: [data_items]}
        ratios: {pipeline_name: ratio}
        total_samples: 总样本数（如果为None，则使用所有可用数据）
        seed: 随机种子
    
    Returns:
        采样后的数据列表
    """
    if seed is not None:
        random.seed(seed)
    
    # 归一化比例
    normalized_ratios = normalize_ratios(ratios)
    
    # 计算每个pipeline的样本数
    pipeline_counts = {}
    available_pipelines = []
    
    for pipeline_name, ratio in normalized_ratios.items():
        if pipeline_name not in pipeline_data:
            print(f"[WARNING] Pipeline '{pipeline_name}' 在输入数据中不存在，跳过")
            continue
        
        available_pipelines.append(pipeline_name)
        data_count = len(pipeline_data[pipeline_name])
        
        if total_samples is None:
            # 使用所有可用数据
            pipeline_counts[pipeline_name] = data_count
        else:
            # 按比例计算样本数
            pipeline_counts[pipeline_name] = max(1, int(total_samples * ratio))
            # 不能超过可用数据量
            pipeline_counts[pipeline_name] = min(
                pipeline_counts[pipeline_name],
                data_count
            )
    
    if not available_pipelines:
        raise ValueError("没有可用的pipeline数据")
    
    # 如果指定了总样本数，重新归一化以确保总和等于total_samples
    if total_samples is not None:
        current_total = sum(pipeline_counts.values())
        if current_total != total_samples:
            # 调整比例，优先保证比例关系
            scale = total_samples / current_total
            for pipeline_name in available_pipelines:
                pipeline_counts[pipeline_name] = max(
                    1, int(pipeline_counts[pipeline_name] * scale)
                )
            # 如果还有差异，从最大的pipeline调整
            current_total = sum(pipeline_counts.values())
            if current_total != total_samples:
                diff = total_samples - current_total
                if diff > 0:
                    # 增加最大的pipeline
                    max_pipeline = max(available_pipelines, 
                                     key=lambda p: pipeline_counts[p])
                    pipeline_counts[max_pipeline] += diff
                else:
                    # 减少最大的pipeline
                    max_pipeline = max(available_pipelines, 
                                     key=lambda p: pipeline_counts[p])
                    pipeline_counts[max_pipeline] += diff  # diff是负数
    
    # 采样数据
    sampled_data = []
    for pipeline_name in available_pipelines:
        data_list = pipeline_data[pipeline_name]
        count = pipeline_counts[pipeline_name]
        
        if count >= len(data_list):
            # 使用所有数据
            sampled = data_list.copy()
        else:
            # 随机采样
            sampled = random.sample(data_list, count)
        
        sampled_data.extend(sampled)
        print(f"  - {pipeline_name}: {len(sampled)}/{len(data_list)} 样本")
    
    # 打乱顺序
    random.shuffle(sampled_data)
    
    return sampled_data


def load_pipeline_files_from_dir(input_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """从目录中加载所有pipeline分类文件"""
    pipeline_data = {}
    
    # 查找所有 vqa_dataset_successful_*_*.json 文件（排除 all_ 文件）
    pattern = "vqa_dataset_successful_*_*.json"
    for file_path in input_dir.glob(pattern):
        if "all_" in file_path.name:
            continue
        
        # 从文件名提取pipeline名称
        # 格式: vqa_dataset_successful_{pipeline_name}_{timestamp}.json
        # 例如: vqa_dataset_successful_object_absence_20240101_120000.json
        stem = file_path.stem  # 去掉 .json 扩展名
        # 移除前缀 "vqa_dataset_successful_"
        prefix = "vqa_dataset_successful_"
        if not stem.startswith(prefix):
            print(f"[WARNING] 文件名格式不符合预期，跳过: {file_path.name}")
            continue
        
        remaining = stem[len(prefix):]
        # 分割剩余部分，最后一个部分通常是时间戳
        parts = remaining.split("_")
        
        # 尝试从数据中获取pipeline_name（更可靠）
        try:
            data = load_json_file(file_path)
            if data and isinstance(data, list) and len(data) > 0:
                # 从第一条数据中获取pipeline_name
                first_item = data[0]
                pipeline_name = first_item.get("pipeline_name")
                if pipeline_name:
                    # 使用数据中的pipeline_name
                    pass
                else:
                    # 回退到文件名解析
                    # 假设最后一部分是时间戳，前面的是pipeline_name
                    if len(parts) >= 2:
                        pipeline_name = "_".join(parts[:-1])
                    else:
                        pipeline_name = parts[0] if parts else "unknown"
            else:
                # 空数据，使用文件名解析
                if len(parts) >= 2:
                    pipeline_name = "_".join(parts[:-1])
                else:
                    pipeline_name = parts[0] if parts else "unknown"
        except Exception as e:
            print(f"[WARNING] 读取文件失败，使用文件名解析: {file_path.name}, 错误: {e}")
            # 回退到文件名解析
            if len(parts) >= 2:
                pipeline_name = "_".join(parts[:-1])
            else:
                pipeline_name = parts[0] if parts else "unknown"
            data = []
        
        if pipeline_name in pipeline_data:
            pipeline_data[pipeline_name].extend(data)
        else:
            pipeline_data[pipeline_name] = data
        
        print(f"加载 Pipeline '{pipeline_name}': {len(data)} 样本 ({file_path.name})")
    
    return pipeline_data


def load_pipeline_files_from_meta(meta_file: Path) -> Dict[str, List[Dict[str, Any]]]:
    """从meta.json加载pipeline文件路径"""
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    pipeline_output_files = meta.get("pipeline_output_files", {})
    if not pipeline_output_files:
        raise ValueError(f"meta.json 中未找到 pipeline_output_files 字段")
    
    pipeline_data = {}
    for pipeline_name, file_path_str in pipeline_output_files.items():
        file_path = Path(file_path_str)
        if not file_path.is_absolute():
            # 相对于meta.json所在目录
            file_path = meta_file.parent / file_path
        
        if not file_path.exists():
            print(f"[WARNING] Pipeline文件不存在: {file_path}")
            continue
        
        data = load_json_file(file_path)
        pipeline_data[pipeline_name] = data
        print(f"加载 Pipeline '{pipeline_name}': {len(data)} 样本 ({file_path.name})")
    
    return pipeline_data


def main():
    parser = argparse.ArgumentParser(
        description='按比例混合不同pipeline类型的数据，生成VQA数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从目录加载，指定比例
  python QA_Generator/pipeline/mix_pipelines.py \\
      --input-dir ./output \\
      --ratios object_absence:0.3 object_counting:0.2 question:0.5 \\
      --output mixed_dataset.json

  # 从meta.json加载
  python QA_Generator/pipeline/mix_pipelines.py \\
      --meta ./output/meta.json \\
      --ratios object_absence:0.3 object_counting:0.2 question:0.5 \\
      --output mixed_dataset.json

  # 指定总样本数
  python QA_Generator/pipeline/mix_pipelines.py \\
      --input-dir ./output \\
      --ratios object_absence:0.3 object_counting:0.2 question:0.5 \\
      --total-samples 1000 \\
      --output mixed_dataset.json

  # 使用配置文件
  python QA_Generator/pipeline/mix_pipelines.py \\
      --config mix_config.json \\
      --output mixed_dataset.json
        """
    )
    
    # 输入源（三选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-dir',
        type=str,
        help='输入目录（包含按pipeline分类的JSON文件）'
    )
    input_group.add_argument(
        '--meta',
        type=str,
        help='meta.json文件路径（从中读取pipeline文件路径）'
    )
    input_group.add_argument(
        '--config',
        type=str,
        help='配置文件路径（JSON格式，包含ratios和input_dir/meta）'
    )
    
    # 比例设置
    parser.add_argument(
        '--ratios',
        nargs='+',
        help='Pipeline比例列表，格式: pipeline1:ratio1 pipeline2:ratio2 ...'
    )
    
    parser.add_argument(
        '--total-samples',
        type=int,
        default=None,
        help='总样本数（如果指定，将按比例采样；否则使用所有可用数据）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出JSON文件路径'
    )
    
    args = parser.parse_args()
    
    # 加载配置（如果使用配置文件）
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if 'ratios' in config:
            ratios_str = [f"{k}:{v}" for k, v in config['ratios'].items()]
            args.ratios = ratios_str
        
        if 'input_dir' in config:
            args.input_dir = config['input_dir']
        elif 'meta' in config:
            args.meta = config['meta']
        
        if 'total_samples' in config:
            args.total_samples = config.get('total_samples')
        
        if 'seed' in config:
            args.seed = config.get('seed')
    
    # 解析比例
    if not args.ratios:
        raise ValueError("必须指定 --ratios 或使用包含ratios的配置文件")
    
    ratios = parse_ratios(args.ratios)
    print(f"比例设置: {ratios}")
    
    # 加载数据
    print("\n加载数据...")
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise ValueError(f"输入目录不存在: {input_dir}")
        pipeline_data = load_pipeline_files_from_dir(input_dir)
    elif args.meta:
        meta_file = Path(args.meta)
        if not meta_file.exists():
            raise ValueError(f"meta.json文件不存在: {meta_file}")
        pipeline_data = load_pipeline_files_from_meta(meta_file)
    else:
        raise ValueError("必须指定 --input-dir 或 --meta")
    
    if not pipeline_data:
        raise ValueError("未找到任何pipeline数据")
    
    # 显示可用pipeline
    print(f"\n可用Pipeline: {list(pipeline_data.keys())}")
    print(f"各Pipeline样本数:")
    for pipeline_name, data in pipeline_data.items():
        print(f"  - {pipeline_name}: {len(data)} 样本")
    
    # 采样数据
    print(f"\n按比例采样数据...")
    if args.total_samples:
        print(f"目标总样本数: {args.total_samples}")
    else:
        print(f"使用所有可用数据")
    
    sampled_data = sample_by_ratio(
        pipeline_data,
        ratios,
        total_samples=args.total_samples,
        seed=args.seed
    )
    
    print(f"\n采样结果: 共 {len(sampled_data)} 样本")
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 混合数据已保存: {output_path}")
    
    # 统计信息
    pipeline_counts = defaultdict(int)
    for item in sampled_data:
        pipeline_name = item.get("pipeline_name", "unknown")
        pipeline_counts[pipeline_name] += 1
    
    print(f"\n最终数据分布:")
    for pipeline_name, count in sorted(pipeline_counts.items()):
        ratio = count / len(sampled_data) if sampled_data else 0
        print(f"  - {pipeline_name}: {count} 样本 ({ratio:.1%})")


if __name__ == "__main__":
    main()
