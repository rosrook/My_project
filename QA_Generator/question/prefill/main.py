#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VQA问题生成系统主程序（预填充对象版本）
支持通过claim或target object直接指定目标对象
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QA_Generator.question.prefill.vqa_generator import VQAGeneratorPrefill


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='VQA问题生成系统（预填充对象版本） - 基于配置文件的声明式问题生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件，使用所有pipeline
  python QA_Generator/question/prefill/main.py input.json output.json
  
  # 只使用特定的pipeline
  python QA_Generator/question/prefill/main.py input.json output.json --pipelines question object_counting
  
  # 限制处理样本数（用于测试）
  python QA_Generator/question/prefill/main.py input.json output.json -n 100

输入数据格式要求:
  输入JSON文件中的每条记录必须包含"prefill"字段，格式如下：
  
  方式1: 使用claim
  {
    "source_a": {...},
    "prefill": {
      "claim": "图片中有一个红色的小汽车停在路边"
    }
  }
  
  方式2: 使用target_object
  {
    "source_a": {...},
    "prefill": {
      "target_object": "car",
      "target_object_info": {  // 可选
        "name": "car",
        "category": "vehicle"
      }
    }
  }
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='输入JSON文件路径（必须包含prefill字段：claim或target_object）'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='输出JSON文件路径（生成的VQA问题）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（默认: QA_Generator/question/config/question_config.json）'
    )
    parser.add_argument(
        '--pipelines',
        type=str,
        nargs='+',
        default=None,
        help='要使用的pipeline列表（默认: 使用所有pipeline）'
    )
    parser.add_argument(
        '-n', '--max-samples',
        type=int,
        default=None,
        help='最大处理样本数（默认: 全部）'
    )
    parser.add_argument(
        '--enable-validation-exemptions',
        action='store_true',
        help='开启指定pipeline的验证豁免（question/visual_recognition/caption/text_association）'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    # 如果没有指定配置文件，使用默认路径
    if args.config:
        config_path = Path(args.config)
    else:
        # 默认配置文件路径（相对于项目根目录）
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "QA_Generator" / "question" / "config" / "question_config.json"
    
    if not input_file.exists():
        print(f"[ERROR] 输入文件不存在: {input_file}")
        return
    
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        return
    
    try:
        # 初始化生成器
        generator = VQAGeneratorPrefill(
            config_path=config_path,
            enable_validation_exemptions=args.enable_validation_exemptions,
        )
        
        # 处理数据文件
        generator.process_data_file(
            input_file=input_file,
            output_file=output_file,
            pipeline_names=args.pipelines,
            max_samples=args.max_samples
        )
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
