#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Judge 模型测试文件
测试 JudgeModel 的各种功能调用

使用方法:
    python test_judge_model.py
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image

# 设置环境变量（如果需要）
import os
# os.environ["OPENAI_API_KEY"] = "your_api_key"
# os.environ["OPENAI_BASE_URL"] = "http://your-api-url/v1"
# os.environ["MODEL_NAME"] = "your-model-name"

from ProbingFactorGeneration.models import JudgeModel


async def test_basic_qa(image_path: str, question: str, model_name: Optional[str] = None):
    """
    测试1: 基本的问答功能
    """
    print("\n" + "=" * 80)
    print("测试1: 基本问答功能")
    print("=" * 80)
    
    img_path = Path(image_path)
    if not img_path.is_file():
        print(f"❌ 图片文件不存在: {img_path}")
        return
    
    image = Image.open(str(img_path)).convert("RGB")
    judge = JudgeModel(model_name=model_name)
    
    async with judge:
        image_base64 = judge._image_to_base64(image)
        
        prompt = (
            "You are a helpful visual question answering model.\n"
            "Given the image and the question, answer concisely in the same language as the question.\n\n"
            f"Question: {question}"
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ]
        
        response = await judge._call_model_async(messages)
        answer = response.choices[0].message.content
        
        print(f"问题: {question}")
        print(f"答案: {answer}")
        print("✅ 测试1完成")


async def test_prefill_slots(
    image_path: str,
    claim_template: Dict[str, Any],
    prefill_slots: List[str],
    model_name: Optional[str] = None
):
    """
    测试2: 预填充槽位功能（prefill_template_slots_async）
    这是 judge 模型用于预填充 claim template 中指定槽位的功能
    """
    print("\n" + "=" * 80)
    print("测试2: 预填充槽位功能")
    print("=" * 80)
    
    img_path = Path(image_path)
    if not img_path.is_file():
        print(f"❌ 图片文件不存在: {img_path}")
        return
    
    image = Image.open(str(img_path)).convert("RGB")
    judge = JudgeModel(model_name=model_name)
    
    async with judge:
        result = await judge.prefill_template_slots_async(
            image=image,
            claim_template=claim_template,
            prefill_slots=prefill_slots
        )
        
        print(f"Claim Template: {claim_template.get('claim_template', '')}")
        print(f"需要预填充的槽位: {prefill_slots}")
        print(f"填充结果:")
        print(f"  - is_relevant: {result.get('is_relevant')}")
        print(f"  - filled_values: {json.dumps(result.get('filled_values', {}), indent=2, ensure_ascii=False)}")
        if result.get('metadata'):
            print(f"  - metadata: {json.dumps(result.get('metadata', {}), indent=2, ensure_ascii=False)}")
        print("✅ 测试2完成")
        return result


async def test_verify_completion(
    image_path: str,
    claim_template: Dict[str, Any],
    completion: Dict[str, Any],
    model_name: Optional[str] = None
):
    """
    测试3: 验证完成结果功能（verify_completion_async）
    这是 judge 模型用于验证 baseline 模型完成的 claim 是否正确
    """
    print("\n" + "=" * 80)
    print("测试3: 验证完成结果功能")
    print("=" * 80)
    
    img_path = Path(image_path)
    if not img_path.is_file():
        print(f"❌ 图片文件不存在: {img_path}")
        return
    
    image = Image.open(str(img_path)).convert("RGB")
    judge = JudgeModel(model_name=model_name)
    
    async with judge:
        verification = await judge.verify_completion_async(
            image=image,
            claim_template=claim_template,
            completion=completion
        )
        
        print(f"Claim Template: {claim_template.get('claim_template', '')}")
        print(f"Completed Claim: {completion.get('completed_claim', '')}")
        print(f"验证结果:")
        print(f"  - is_correct: {verification.get('is_correct')}")
        print(f"  - claim_is_valid: {verification.get('claim_is_valid')}")
        print(f"  - explanation_is_reasonable: {verification.get('explanation_is_reasonable')}")
        print(f"  - failure_reason: {verification.get('failure_reason')}")
        print(f"  - judge_explanation: {verification.get('judge_explanation', '')[:200]}...")
        if verification.get('metadata', {}).get('precheck_result'):
            precheck = verification['metadata']['precheck_result']
            print(f"  - precheck_result:")
            print(f"      template_is_answerable: {precheck.get('template_is_answerable')}")
            print(f"      confidence: {precheck.get('confidence')}")
        print("✅ 测试3完成")
        return verification


async def test_batch_prefill(
    image_paths: List[str],
    claim_templates: List[Dict[str, Any]],
    prefill_slots_list: List[List[str]],
    model_name: Optional[str] = None
):
    """
    测试4: 批量预填充槽位功能（prefill_template_slots_batch_async）
    """
    print("\n" + "=" * 80)
    print("测试4: 批量预填充槽位功能")
    print("=" * 80)
    
    images = []
    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        if not img_path.is_file():
            print(f"❌ 图片文件不存在: {img_path}")
            continue
        images.append(Image.open(str(img_path)).convert("RGB"))
    
    if not images:
        print("❌ 没有有效的图片")
        return
    
    judge = JudgeModel(model_name=model_name, max_concurrent=3)
    
    async with judge:
        results = await judge.prefill_template_slots_batch_async(
            images=images,
            claim_templates=claim_templates[:len(images)],
            prefill_slots_list=prefill_slots_list[:len(images)]
        )
        
        print(f"批量处理 {len(results)} 个模板")
        for i, result in enumerate(results):
            print(f"\n结果 {i+1}:")
            print(f"  - is_relevant: {result.get('is_relevant')}")
            print(f"  - filled_values: {json.dumps(result.get('filled_values', {}), indent=2, ensure_ascii=False)}")
        print("✅ 测试4完成")
        return results


async def test_batch_verify(
    image_paths: List[str],
    claim_templates: List[Dict[str, Any]],
    completions: List[Dict[str, Any]],
    model_name: Optional[str] = None
):
    """
    测试5: 批量验证完成结果功能（verify_completion_batch_async）
    """
    print("\n" + "=" * 80)
    print("测试5: 批量验证完成结果功能")
    print("=" * 80)
    
    images = []
    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        if not img_path.is_file():
            print(f"❌ 图片文件不存在: {img_path}")
            continue
        images.append(Image.open(str(img_path)).convert("RGB"))
    
    if not images:
        print("❌ 没有有效的图片")
        return
    
    judge = JudgeModel(model_name=model_name, max_concurrent=3)
    
    async with judge:
        verifications = await judge.verify_completion_batch_async(
            images=images,
            claim_templates=claim_templates[:len(images)],
            completions=completions[:len(images)]
        )
        
        print(f"批量验证 {len(verifications)} 个完成结果")
        for i, verification in enumerate(verifications):
            print(f"\n验证结果 {i+1}:")
            print(f"  - is_correct: {verification.get('is_correct')}")
            print(f"  - failure_reason: {verification.get('failure_reason')}")
            print(f"  - judge_explanation: {verification.get('judge_explanation', '')[:150]}...")
        print("✅ 测试5完成")
        return verifications


async def main():
    """
    主测试函数
    请根据实际情况修改以下参数
    """
    # ========== 配置参数 ==========
    # 图片路径（请修改为实际路径）
    IMAGE_PATH = "/path/to/your/test_image.jpg"
    
    # 模型名称（None 表示使用默认配置）
    MODEL_NAME = None  # 例如: "Qwen3-VL-235B-A22B-Instruct" 或从环境变量读取
    
    # 如果环境变量已设置，可以从环境变量读取
    MODEL_NAME = os.getenv("MODEL_NAME") or MODEL_NAME
    
    # ========== 测试1: 基本问答 ==========
    print("\n开始测试 Judge 模型...")
    
    # 如果图片文件存在，运行测试1
    if Path(IMAGE_PATH).is_file():
        await test_basic_qa(
            image_path=IMAGE_PATH,
            question="这张图片中有什么物体？",
            model_name=MODEL_NAME
        )
    else:
        print(f"\n⚠️  跳过测试1: 图片文件不存在 ({IMAGE_PATH})")
        print("   请修改 IMAGE_PATH 为实际的图片路径")
    
    # ========== 测试2: 预填充槽位 ==========
    if Path(IMAGE_PATH).is_file():
        claim_template_example = {
            "claim_template": "The [OBJECT_A] is [RELATIVE_DIRECTION] the [OBJECT_B].",
            "claim_id": "spatial_relative_position",
            "slots": {
                "OBJECT_A": {
                    "type": "object",
                    "description": "First object in the spatial relationship",
                    "selection_criteria": "Select a visually salient object"
                },
                "OBJECT_B": {
                    "type": "object",
                    "description": "Second object in the spatial relationship",
                    "selection_criteria": "Select a visually salient object"
                },
                "RELATIVE_DIRECTION": {
                    "type": "categorical_value",
                    "description": "Spatial relationship direction",
                    "values": ["to the left of", "to the right of", "above", "below", "in front of", "behind"]
                }
            },
            "metadata": {
                "name": "Relative Spatial Relationship"
            }
        }
        
        await test_prefill_slots(
            image_path=IMAGE_PATH,
            claim_template=claim_template_example,
            prefill_slots=["OBJECT_A", "OBJECT_B"],
            model_name=MODEL_NAME
        )
    else:
        print(f"\n⚠️  跳过测试2: 图片文件不存在 ({IMAGE_PATH})")
    
    # ========== 测试3: 验证完成结果 ==========
    if Path(IMAGE_PATH).is_file():
        claim_template_example = {
            "claim_template": "The [OBJECT_A] is [RELATIVE_DIRECTION] the [OBJECT_B].",
            "claim_id": "spatial_relative_position",
            "slots": {
                "RELATIVE_DIRECTION": {
                    "type": "categorical_value",
                    "values": ["to the left of", "to the right of", "above", "below"]
                }
            }
        }
        
        completion_example = {
            "completed_claim": "The dog is to the left of the hurdle.",
            "explanation": "The dog is positioned to the left side of the hurdle in the image.",
            "is_related": True,
            "claim_id": "spatial_relative_position",
            "content_type": "spatial",
            "metadata": {
                "original_template": "The [OBJECT_A] is [RELATIVE_DIRECTION] the [OBJECT_B].",
                "placeholders": ["OBJECT_A", "RELATIVE_DIRECTION", "OBJECT_B"]
            }
        }
        
        await test_verify_completion(
            image_path=IMAGE_PATH,
            claim_template=claim_template_example,
            completion=completion_example,
            model_name=MODEL_NAME
        )
    else:
        print(f"\n⚠️  跳过测试3: 图片文件不存在 ({IMAGE_PATH})")
    
    # ========== 测试4和5: 批量测试 ==========
    # 如果需要批量测试，可以取消下面的注释
    # IMAGE_PATHS = [IMAGE_PATH] * 3  # 使用同一张图片测试3次
    # await test_batch_prefill(...)
    # await test_batch_verify(...)
    
    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    # 运行所有测试
    asyncio.run(main())
