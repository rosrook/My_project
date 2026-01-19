#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VQA答案生成系统主程序
"""
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QA_Generator.answer.answer_generator import AnswerGenerator
from QA_Generator.answer.validator import AnswerValidator


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='VQA答案生成系统 - 根据问题和图片生成答案',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理问题文件，生成答案
  python QA_Generator/answer/main.py questions.json answers.json
  
  # 指定配置文件
  python QA_Generator/answer/main.py questions.json answers.json --config QA_Generator/answer/answer_config.json
  
  # 限制处理样本数（用于测试）
  python QA_Generator/answer/main.py questions.json answers.json -n 100
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='输入JSON文件路径（问题生成系统的输出）'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='输出JSON文件路径（生成的答案）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（默认: QA_Generator/answer/answer_config.json）'
    )
    parser.add_argument(
        '-n', '--max-samples',
        type=int,
        default=None,
        help='最大处理样本数（默认: 全部）'
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
        config_path = project_root / "QA_Generator" / "answer" / "answer_config.json"
    
    if not input_file.exists():
        print(f"[ERROR] 输入文件不存在: {input_file}")
        return
    
    try:
        # 初始化生成器和校验器
        generator = AnswerGenerator(config_path=config_path if config_path.exists() else None)
        validator = AnswerValidator()
        
        # 处理数据文件
        process_answer_file(
            generator=generator,
            validator=validator,
            input_file=input_file,
            output_file=output_file,
            max_samples=args.max_samples
        )
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()


def process_answer_file(
    generator: AnswerGenerator,
    validator: AnswerValidator,
    input_file: Path,
    output_file: Path,
    max_samples: Optional[int] = None
) -> None:
    """
    处理答案生成文件
    
    Args:
        generator: 答案生成器实例
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        max_samples: 最大处理样本数
    """
    print(f"[INFO] 读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"输入文件应该包含一个数组，但得到: {type(data)}")
    
    print(f"[INFO] 总记录数: {len(data)}")
    
    if max_samples:
        data = data[:max_samples]
        print(f"[INFO] 限制处理前 {max_samples} 条记录")
    
    # 处理每条记录
    results = []
    errors = []
    total_processed = 0
    total_success = 0
    total_failed = 0
    
    for idx, record in enumerate(data, 1):
        total_processed += 1
        
        try:
            # 提取必要信息
            question = record.get("question")
            question_type = record.get("question_type")
            image_base64 = record.get("image_base64")
            
            if not question:
                errors.append({
                    "index": idx,
                    "id": record.get("id"),
                    "error": "缺少question字段"
                })
                total_failed += 1
                print(f"[WARNING] 记录 {idx} 缺少question字段，跳过")
                continue
            
            if not question_type:
                errors.append({
                    "index": idx,
                    "id": record.get("id"),
                    "error": "缺少question_type字段"
                })
                total_failed += 1
                print(f"[WARNING] 记录 {idx} 缺少question_type字段，跳过")
                continue
            
            if not image_base64:
                errors.append({
                    "index": idx,
                    "id": record.get("id"),
                    "error": "缺少image_base64字段"
                })
                total_failed += 1
                print(f"[WARNING] 记录 {idx} 缺少image_base64字段，跳过")
                continue
            
            # 生成答案
            pipeline_info = {
                "pipeline_name": record.get("pipeline_name"),
                "pipeline_intent": record.get("pipeline_intent"),
                "answer_type": record.get("answer_type")
            }
            
            answer_result = generator.generate_answer(
                question=question,
                image_base64=image_base64,
                question_type=question_type,
                pipeline_info=pipeline_info
            )
            
            if not answer_result or answer_result.get("answer") is None:
                errors.append({
                    "index": idx,
                    "id": record.get("id"),
                    "error": "答案生成失败"
                })
                total_failed += 1
                print(f"[WARNING] 记录 {idx} 答案生成失败")
                continue
            
            # 校验和修复
            validated_result, validation_report = validator.validate_and_fix(
                result=answer_result,
                image_base64=image_base64
            )
            
            # 如果校验失败，记录警告但继续处理
            if not validation_report.get("validation_passed", True):
                print(f"[WARNING] 记录 {idx} 校验未完全通过，但已尝试修复")
            
            # 构建输出结果
            result = {
                # 原始信息
                "question": question,
                "question_type": question_type,
                "image_base64": image_base64,
                
                # 答案信息（使用校验后的结果）
                "answer": validated_result.get("answer", answer_result.get("answer")),
                "explanation": validated_result.get("explanation", answer_result.get("explanation", "")),
                
                # 完整问题（选择题包含选项）
                "full_question": validated_result.get("full_question", answer_result.get("full_question", question)),
                
                # 选择题特有字段
                "options": validated_result.get("options", answer_result.get("options")),
                "correct_option": validated_result.get("correct_option", answer_result.get("correct_option")),
                
                # 校验信息
                "validation_report": validation_report,
                
                # 原始元数据
                "pipeline_name": record.get("pipeline_name"),
                "pipeline_intent": record.get("pipeline_intent"),
                "answer_type": record.get("answer_type"),
                "sample_index": record.get("sample_index"),
                "id": record.get("id"),
                "source_a_id": record.get("source_a_id"),
                "timestamp": record.get("timestamp", ""),
                "generated_at": datetime.now().isoformat()
            }
            
            results.append(result)
            total_success += 1
            
            # 进度报告
            if total_processed % 10 == 0:
                print(f"[进度] 已处理: {total_processed}, 成功: {total_success}, 失败: {total_failed}")
                
        except Exception as e:
            errors.append({
                "index": idx,
                "id": record.get("id"),
                "error": str(e)
            })
            total_failed += 1
            print(f"[ERROR] 处理记录 {idx} 时出错: {e}")
    
    # 保存结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存错误信息
    if errors:
        error_file = output_file.parent / f"{output_file.stem}_errors.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"  错误信息已保存到: {error_file}")
    
    print(f"\n[完成] 处理完成！")
    print(f"  总处理: {total_processed}")
    print(f"  成功生成: {total_success}")
    print(f"  失败: {total_failed}")
    print(f"  结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

