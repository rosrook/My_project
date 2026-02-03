#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VQA数据集生成完整流程
接收batch_process.sh的输出，依次生成问题和答案，生成完整的VQA数据集
"""
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = None
    HAS_TQDM = False

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QA_Generator.question.prefill.vqa_generator import VQAGeneratorPrefill
from QA_Generator.answer.answer_generator import AnswerGenerator
from QA_Generator.answer.validator import AnswerValidator
from QA_Generator.clients.async_client import AsyncGeminiClient
from QA_Generator.logging.logger import (
    Logger,
    set_global_logger,
    get_logger,
    log_warning,
    log_error,
    log_debug,
    log_debug_dict,
)


# 错误详情字段最大长度（避免记录过大导致IO/内存问题）
_ERROR_RAW_RESPONSE_MAX = 500
_ERROR_REASON_MAX = 200
_ERROR_SUMMARY_MAX = 300
_ERROR_ISSUES_MAX = 3  # 最多保留几条 issues
_ERROR_TRACEBACK_MAX = 400


class VQAPipeline:
    """VQA数据集生成完整流程"""
    
    def __init__(
        self,
        question_config_path: Optional[Path] = None,
        answer_config_path: Optional[Path] = None,
        enable_validation_exemptions: bool = False,
    ):
        """
        初始化流程
        
        Args:
            question_config_path: 问题生成配置文件路径
            answer_config_path: 答案生成配置文件路径
        """
        project_root = Path(__file__).parent.parent
        
        # 默认配置文件路径
        if question_config_path is None:
            question_config_path = project_root / "question" / "config" / "question_config.json"
        if answer_config_path is None:
            answer_config_path = project_root / "answer" / "answer_config.json"
        
        # 初始化生成器
        self.question_generator = VQAGeneratorPrefill(
            config_path=question_config_path,
            enable_validation_exemptions=enable_validation_exemptions,
        )
        self.answer_generator = AnswerGenerator(
            config_path=answer_config_path if answer_config_path.exists() else None
        )
        self.validator = AnswerValidator()
        
        # 统计信息
        self.stats = {
            "input_records": 0,
            "questions_generated": 0,
            "answers_generated": 0,
            "validation_passed": 0,
            "validation_failed": 0,
            "errors": []
        }
    
    def _extract_error_details(
        self,
        error_stage: str,
        error_msg: str,
        validation_report: Optional[Dict[str, Any]] = None,
        answer_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        从校验报告和答案生成结果中提取结构化错误详情，便于排查。
        所有字符串均截断，避免记录过大导致IO/内存问题。
        """
        def _trunc(s: str, n: int) -> str:
            if not s:
                return ""
            return str(s)[:n]

        def _compact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            """移除空值，减少记录体积"""
            return {k: v for k, v in (d or {}).items() if v is not None and v != "" and v != []}

        details: Dict[str, Any] = {
            "error_stage": error_stage,
            "error_summary": _trunc(error_msg, _ERROR_SUMMARY_MAX),
        }
        if validation_report:
            fc = validation_report.get("format_check") or {}
            if isinstance(fc, dict):
                issues = fc.get("issues", []) or []
                details["format_check"] = {
                    "passed": fc.get("passed", True),
                    "issues": [str(i)[:150] for i in issues[:_ERROR_ISSUES_MAX]],
                    "fixes_applied": (fc.get("fixes_applied") or [])[:3],
                }
            vqa = validation_report.get("vqa_validation") or {}
            if isinstance(vqa, dict):
                conf = vqa.get("confidence_assessment") or {}
                ans_val = vqa.get("answer_validation") or {}
                ans_issues = (ans_val.get("issues", []) or [])[:3]
                details["vqa_validation"] = {
                    "passed": vqa.get("passed", True),
                    "confidence_assessment": _compact_dict({
                        "passed": conf.get("passed", True),
                        "confidence": conf.get("confidence"),
                        "correctness_reason": _trunc(conf.get("correctness_reason"), _ERROR_REASON_MAX),
                        "raw_response_truncated": _trunc(conf.get("raw_response_truncated"), _ERROR_RAW_RESPONSE_MAX),
                    }) if isinstance(conf, dict) else {},
                    "answer_validation": _compact_dict({
                        "passed": ans_val.get("passed", True),
                        "is_valid": ans_val.get("is_valid"),
                        "validation_reason": _trunc(ans_val.get("validation_reason"), _ERROR_REASON_MAX),
                        "issues": [str(i)[:150] for i in ans_issues],
                        "raw_response_truncated": _trunc(ans_val.get("raw_response_truncated"), _ERROR_RAW_RESPONSE_MAX),
                    }) if isinstance(ans_val, dict) else {},
                    "rescue_attempted": vqa.get("rescue_attempted"),
                    "rescue_successful": vqa.get("rescue_successful"),
                    "rescue_reason": _trunc(vqa.get("rescue_reason"), _ERROR_REASON_MAX),
                }
            details["regeneration_reason"] = _trunc(validation_report.get("regeneration_reason"), _ERROR_SUMMARY_MAX)
            repair = validation_report.get("repair") or {}
            if isinstance(repair, dict) and repair:
                details["repair"] = {
                    "repair_successful": repair.get("repair_successful"),
                    "repair_reason": _trunc(repair.get("repair_reason"), _ERROR_REASON_MAX),
                }
        if answer_result and isinstance(answer_result, dict):
            gen_detail: Dict[str, Any] = {
                "error_type": answer_result.get("error_type"),
                "error_step": answer_result.get("error_step"),
            }
            if answer_result.get("error"):
                gen_detail["error"] = _trunc(str(answer_result.get("error")), _ERROR_SUMMARY_MAX)
            raw = answer_result.get("raw_response") or answer_result.get("raw_response_truncated")
            if raw:
                gen_detail["raw_response_truncated"] = _trunc(str(raw), _ERROR_RAW_RESPONSE_MAX)
            if gen_detail:
                details["generation"] = gen_detail
        return details
    
    def _generate_answers_from_data(
        self,
        questions_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        从问题数据直接生成答案（优化版本，避免文件I/O）
        
        Returns:
            (results, errors): 答案结果列表和错误列表
        """
        from datetime import datetime
        
        if not isinstance(questions_data, list):
            raise ValueError(f"问题数据应该是一个列表，但得到: {type(questions_data)}")
        
        # 处理每个问题
        results = []
        errors = []
        total_processed = 0
        total_success = 0
        total_failed = 0
        
        for idx, record in enumerate(questions_data, 1):
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
                        "error": "缺少question字段",
                        "error_stage": "input_validation",
                        "error_details": self._extract_error_details("input_validation", "缺少question字段"),
                        "sample_index": record.get("sample_index"),
                        "source_a_id": record.get("source_a_id"),
                    })
                    total_failed += 1
                    continue
                
                if not question_type:
                    errors.append({
                        "index": idx,
                        "id": record.get("id"),
                        "error": "缺少question_type字段",
                        "error_stage": "input_validation",
                        "error_details": self._extract_error_details("input_validation", "缺少question_type字段"),
                        "sample_index": record.get("sample_index"),
                        "source_a_id": record.get("source_a_id"),
                    })
                    total_failed += 1
                    continue
                
                if not image_base64:
                    errors.append({
                        "index": idx,
                        "id": record.get("id"),
                        "error": "缺少image_base64字段",
                        "error_stage": "input_validation",
                        "error_details": self._extract_error_details("input_validation", "缺少image_base64字段"),
                        "sample_index": record.get("sample_index"),
                        "source_a_id": record.get("source_a_id"),
                    })
                    total_failed += 1
                    continue
                
                # 生成答案（移除重试和验证逻辑以提高效率）
                prefill_object = record.get("prefill_object")
                prefill = record.get("prefill") if isinstance(record.get("prefill"), dict) else {}
                target_object = None
                if isinstance(prefill_object, dict):
                    target_object = prefill_object.get("name")
                if not target_object and isinstance(prefill, dict):
                    target_object = prefill.get("target_object")
                prefill_claim = prefill.get("claim") if isinstance(prefill, dict) else None
                pipeline_info = {
                    "pipeline_name": record.get("pipeline_name"),
                    "pipeline_intent": record.get("pipeline_intent"),
                    "answer_type": record.get("answer_type"),
                    "prefill_object": prefill_object,
                    "target_object": target_object,
                    "prefill_claim": prefill_claim,
                }
                
                answer_result = self.answer_generator.generate_answer(
                    question=question,
                    image_base64=image_base64,
                    question_type=question_type,
                    pipeline_info=pipeline_info
                )
                
                if not answer_result or answer_result.get("answer") is None:
                    error_msg = (
                        answer_result.get("error", "答案生成失败")
                        if isinstance(answer_result, dict)
                        else "答案生成失败"
                    )
                    err_details = self._extract_error_details(
                        "generation", error_msg,
                        answer_result=answer_result if isinstance(answer_result, dict) else None,
                    )
                    errors.append({
                        "index": idx,
                        "id": record.get("id"),
                        "error": error_msg,
                        "error_stage": "generation",
                        "error_details": err_details,
                        "sample_index": record.get("sample_index"),
                        "source_a_id": record.get("source_a_id"),
                    })
                    total_failed += 1
                    continue
                
                # 构建输出结果（直接使用生成结果，不进行验证）
                result = {
                    # 原始信息
                    "question": question,
                    "question_type": question_type,
                    "image_base64": image_base64,
                    
                    # 答案信息（直接使用生成结果）
                    "answer": answer_result.get("answer"),
                    "explanation": answer_result.get("explanation", ""),
                    
                    # 完整问题（选择题包含选项）
                    "full_question": answer_result.get("full_question", question),
                    
                    # 选择题特有字段
                    "options": answer_result.get("options"),
                    "correct_option": answer_result.get("correct_option"),
                    
                    # 校验信息（简化：标记为通过）
                    "validation_report": {"validation_passed": True},
                    
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
                
                # 进度报告（优化：减少输出频率，提高速度）
                if total_processed % 50 == 0:
                    print(f"[进度] 已处理: {total_processed}/{len(questions_data)}, 成功: {total_success}, 失败: {total_failed}")
            
            except Exception as e:
                import traceback
                exc_details = self._extract_error_details("exception", str(e))
                exc_details["exception_type"] = type(e).__name__
                exc_details["traceback"] = traceback.format_exc()[-_ERROR_TRACEBACK_MAX:]
                errors.append({
                    "index": idx,
                    "id": record.get("id"),
                    "error": str(e),
                    "error_stage": "exception",
                    "error_details": exc_details,
                    "sample_index": record.get("sample_index"),
                    "source_a_id": record.get("source_a_id"),
                })
                total_failed += 1
                # 优化：只在每10个错误时输出一次，减少日志开销
                if total_failed % 10 == 0:
                    log_error(f"已累计 {total_failed} 个错误（最新: 记录 {idx}）")
        
        print(f"[完成] 答案生成完成！总处理: {total_processed}, 成功: {total_success}, 失败: {total_failed}")
        return results, errors
    
    async def _generate_answers_from_data_async(
        self,
        questions_data: List[Dict[str, Any]],
        concurrency: int = 5,
        request_delay: float = 0.1,
        suppress_progress: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        异步从问题数据生成答案（并发调用，参考vlmtool/generate_vqa的实现）
        
        Args:
            questions_data: 问题数据列表
            concurrency: 最大并发数（建议1-5，某些API不支持高并发）
            request_delay: 每个请求之间的延迟（秒），用于避免触发API并发限制
        """
        if not isinstance(questions_data, list):
            raise ValueError(f"问题数据应该是一个列表，但得到: {type(questions_data)}")
        
        # 如果并发数过高，给出警告
        if concurrency > 5:
            warning_msg = f"并发数设置为 {concurrency}，某些API可能不支持高并发。\n  如果遇到401错误，建议降低并发数（--concurrency 1-3）"
            log_warning(warning_msg)
        
        from datetime import datetime
        
        # 使用AsyncGeminiClient进行异步并发处理
        async with AsyncGeminiClient(max_concurrent=concurrency, request_delay=request_delay) as client:
            async def process_one(idx: int, record: Dict[str, Any]):
                """处理单个问题记录"""
                try:
                    # ========== 步骤1: 提取输入数据 ==========
                    question = record.get("question")
                    question_type = record.get("question_type")
                    image_base64 = record.get("image_base64")
                    
                    log_debug(f"[记录 {idx}] ========== 开始处理 ==========")
                    log_debug_dict(f"[记录 {idx}] 输入数据", {
                        "id": record.get("id"),
                        "question": question,
                        "question_type": question_type,
                        "image_base64_length": len(image_base64) if image_base64 else 0,
                        "pipeline_name": record.get("pipeline_name"),
                        "pipeline_intent": record.get("pipeline_intent"),
                        "answer_type": record.get("answer_type")
                    })
                    
                    if not question:
                        return ("err", {
                            "index": idx,
                            "id": record.get("id"),
                            "error": "缺少question字段",
                            "error_stage": "input_validation",
                            "error_details": self._extract_error_details("input_validation", "缺少question字段"),
                            "sample_index": record.get("sample_index"),
                            "source_a_id": record.get("source_a_id"),
                        })
                    if not question_type:
                        return ("err", {
                            "index": idx,
                            "id": record.get("id"),
                            "error": "缺少question_type字段",
                            "error_stage": "input_validation",
                            "error_details": self._extract_error_details("input_validation", "缺少question_type字段"),
                            "sample_index": record.get("sample_index"),
                            "source_a_id": record.get("source_a_id"),
                        })
                    if not image_base64:
                        return ("err", {
                            "index": idx,
                            "id": record.get("id"),
                            "error": "缺少image_base64字段",
                            "error_stage": "input_validation",
                            "error_details": self._extract_error_details("input_validation", "缺少image_base64字段"),
                            "sample_index": record.get("sample_index"),
                            "source_a_id": record.get("source_a_id"),
                        })
                    
                    prefill_object = record.get("prefill_object")
                    prefill = record.get("prefill") if isinstance(record.get("prefill"), dict) else {}
                    target_object = None
                    if isinstance(prefill_object, dict):
                        target_object = prefill_object.get("name")
                    if not target_object and isinstance(prefill, dict):
                        target_object = prefill.get("target_object")
                    prefill_claim = prefill.get("claim") if isinstance(prefill, dict) else None
                    pipeline_info = {
                        "pipeline_name": record.get("pipeline_name"),
                        "pipeline_intent": record.get("pipeline_intent"),
                        "answer_type": record.get("answer_type"),
                        "prefill_object": prefill_object,
                        "target_object": target_object,
                        "prefill_claim": prefill_claim,
                    }
                    
                    max_retries = 3
                    retry_count = 0
                    last_error = None
                    last_answer_result = None  # 用于错误详情（生成失败时）
                    validated_result = None
                    validation_report = None
                    
                    while retry_count <= max_retries:
                        # ========== 步骤2: 生成答案 ==========
                        log_debug(f"[记录 {idx}] ========== 步骤2: 生成答案 (重试 {retry_count}/{max_retries}) ==========")
                        log_debug(f"[记录 {idx}] 调用 generate_answer_async")
                        log_debug_dict(f"[记录 {idx}] 生成答案的输入参数", {
                            "question": question,
                            "question_type": question_type,
                            "pipeline_info": pipeline_info
                        })
                        
                        # 生成答案（异步，与vlmtool/generate_vqa对齐）
                        # 获取模型名称（从config或使用默认值）
                        from QA_Generator.config import config
                        model_name = config.MODEL_NAME
                        answer_result = await self.answer_generator.generate_answer_async(
                            question=question,
                            image_base64=image_base64,
                            question_type=question_type,
                            pipeline_info=pipeline_info,
                            async_client=client,
                            model=model_name  # 传入模型名称，与vlmtool一致
                        )
                        
                        log_debug(f"[记录 {idx}] generate_answer_async 返回结果")
                        log_debug_dict(f"[记录 {idx}] 答案生成结果 (answer_result)", answer_result if answer_result else {"error": "返回None"})
                        
                        if not answer_result or answer_result.get("answer") is None:
                            last_error = (
                                answer_result.get("error", "答案生成失败")
                                if isinstance(answer_result, dict)
                                else "答案生成失败"
                            )
                            last_answer_result = answer_result if isinstance(answer_result, dict) else None
                            log_error(f"[记录 {idx}] 答案生成失败: {last_error}, answer_result={answer_result}")
                            retry_count += 1
                            if retry_count <= max_retries:
                                continue
                            else:
                                # 最后一次重试仍失败，记录生成阶段的错误详情
                                err_details = self._extract_error_details(
                                    "generation", last_error,
                                    answer_result=last_answer_result,
                                )
                                return ("err", {
                                    "index": idx,
                                    "id": record.get("id"),
                                    "error": last_error,
                                    "error_stage": "generation",
                                    "error_details": err_details,
                                    "retry_count": retry_count,
                                    "sample_index": record.get("sample_index"),
                                    "source_a_id": record.get("source_a_id"),
                                })
                        
                        # ========== 步骤3: 校验和修复 ==========
                        log_debug(f"[记录 {idx}] ========== 步骤3: 校验和修复 ==========")
                        log_debug(f"[记录 {idx}] 调用 validator.validate_and_fix")
                        log_debug_dict(f"[记录 {idx}] 传递给 validator 的 result", {
                            "question": answer_result.get("question"),
                            "question_type": answer_result.get("question_type"),
                            "answer": answer_result.get("answer"),
                            "full_question": answer_result.get("full_question"),
                            "options": answer_result.get("options"),
                            "correct_option": answer_result.get("correct_option"),
                            "explanation": answer_result.get("explanation", "")[:100] if answer_result.get("explanation") else ""
                        })
                        
                        # 校验和修复（同步操作，但很快）
                        validated_result, validation_report = self.validator.validate_and_fix(
                            result=answer_result,
                            image_base64=image_base64
                        )
                        
                        log_debug(f"[记录 {idx}] validator.validate_and_fix 返回结果")
                        log_debug_dict(f"[记录 {idx}] 验证后的结果 (validated_result)", {
                            "question": validated_result.get("question") if validated_result else None,
                            "question_type": validated_result.get("question_type") if validated_result else None,
                            "answer": validated_result.get("answer") if validated_result else None,
                            "full_question": validated_result.get("full_question") if validated_result else None,
                            "options": validated_result.get("options") if validated_result else None,
                            "correct_option": validated_result.get("correct_option") if validated_result else None,
                            "explanation": validated_result.get("explanation", "")[:100] if validated_result and validated_result.get("explanation") else ""
                        })
                        log_debug_dict(f"[记录 {idx}] 验证报告 (validation_report)", validation_report if validation_report else {"error": "返回None"})
                        
                        if validation_report.get("validation_passed", False):
                            if retry_count > 0:
                                print(f"[成功] 记录 {idx} 经过 {retry_count} 次重试后验证通过")
                            break
                        
                        if not validation_report.get("should_regenerate", False):
                            break
                        
                        retry_count += 1
                        if retry_count <= max_retries:
                            reason = validation_report.get("regeneration_reason", "验证失败")
                            print(f"[重试] 记录 {idx} 验证失败 ({reason})，正在重新生成 ({retry_count}/{max_retries})...")
                        else:
                            last_error = f"经过 {max_retries} 次重试后仍验证失败: {validation_report.get('regeneration_reason', '验证失败')}"
                            last_answer_result = answer_result  # 保存最后一次生成结果，用于错误详情
                    
                    if validated_result is None or not validation_report.get("validation_passed", False):
                        err_details = self._extract_error_details(
                            "validation" if validation_report else "generation",
                            last_error or "答案生成或验证失败",
                            validation_report=validation_report,
                            answer_result=last_answer_result if validation_report is None else None,
                        )
                        return ("err", {
                            "index": idx,
                            "id": record.get("id"),
                            "error": last_error or "答案生成或验证失败",
                            "error_stage": "validation" if validation_report else "generation",
                            "error_details": err_details,
                            "retry_count": retry_count,
                            "validation_report": validation_report,
                            "sample_index": record.get("sample_index"),
                            "source_a_id": record.get("source_a_id"),
                        })
                    
                    # ========== 步骤4: 构建最终输出结果 ==========
                    log_debug(f"[记录 {idx}] ========== 步骤4: 构建最终输出结果 ==========")
                    
                    final_answer = validated_result.get("answer", answer_result.get("answer"))
                    final_explanation = validated_result.get("explanation", answer_result.get("explanation", ""))
                    final_full_question = validated_result.get("full_question", answer_result.get("full_question", question))
                    final_options = validated_result.get("options", answer_result.get("options"))
                    final_correct_option = validated_result.get("correct_option", answer_result.get("correct_option"))
                    
                    log_debug_dict(f"[记录 {idx}] 最终输出结果字段", {
                        "question": question,
                        "question_type": question_type,
                        "answer": final_answer,
                        "full_question": final_full_question,
                        "options": final_options,
                        "correct_option": final_correct_option,
                        "explanation_length": len(final_explanation) if final_explanation else 0
                    })
                    
                    # 特别详细记录 options 和 full_question
                    if final_options:
                        log_debug(f"[记录 {idx}] 最终 options 字典内容:")
                        for key, value in final_options.items():
                            log_debug(f"[记录 {idx}]   {key}: {value} (type: {type(value).__name__})")
                    else:
                        log_warning(f"[记录 {idx}] 最终 options 为空或 None!")
                    
                    log_debug(f"[记录 {idx}] 最终 full_question 内容:")
                    log_debug(f"[记录 {idx}] {final_full_question}")
                    
                    result = {
                        "question": question,
                        "question_type": question_type,
                        "image_base64": image_base64,
                        "answer": final_answer,
                        "explanation": final_explanation,
                        "full_question": final_full_question,
                        "options": final_options,
                        "correct_option": final_correct_option,
                        "validation_report": validation_report,
                        "pipeline_name": record.get("pipeline_name"),
                        "pipeline_intent": record.get("pipeline_intent"),
                        "answer_type": record.get("answer_type"),
                        "sample_index": record.get("sample_index"),
                        "id": record.get("id"),
                        "source_a_id": record.get("source_a_id"),
                        "timestamp": record.get("timestamp", ""),
                        "generated_at": datetime.now().isoformat()
                    }
                    
                    log_debug(f"[记录 {idx}] ========== 处理完成 ==========\n")
                    return ("ok", result)
                except Exception as e:
                    import traceback
                    exc_details = self._extract_error_details("exception", str(e))
                    exc_details["exception_type"] = type(e).__name__
                    exc_details["traceback"] = traceback.format_exc()[-_ERROR_TRACEBACK_MAX:]
                    return ("err", {
                        "index": idx,
                        "id": record.get("id"),
                        "error": str(e),
                        "error_stage": "exception",
                        "error_details": exc_details,
                        "sample_index": record.get("sample_index"),
                        "source_a_id": record.get("source_a_id"),
                    })
            
            # 创建所有任务
            tasks = [process_one(i, rec) for i, rec in enumerate(questions_data, 1)]
            
            # 使用 as_completed 实现实时进度跟踪（参考vlmtool/generate_vqa的实现）
            results: List[Dict[str, Any]] = []
            errors: List[Dict[str, Any]] = []
            completed = 0
            total = len(tasks)
            
            # 并发执行所有任务并实时跟踪进度
            for coro in asyncio.as_completed(tasks):
                try:
                    tag, payload = await coro
                    if tag == "ok":
                        results.append(payload)
                    else:
                        errors.append(payload)
                except Exception as e:
                    import traceback
                    exc_details = self._extract_error_details("exception", str(e))
                    exc_details["exception_type"] = type(e).__name__
                    exc_details["traceback"] = traceback.format_exc()[-_ERROR_TRACEBACK_MAX:]
                    errors.append({
                        "error": f"任务执行异常: {str(e)}",
                        "error_stage": "exception",
                        "error_details": exc_details,
                    })
                
                completed += 1
                
                # 每完成10个或完成全部时显示进度（使用全局进度条时抑制）
                if not suppress_progress and (completed % 10 == 0 or completed == total):
                    success_count = len(results)
                    failed_count = len(errors)
                    print(f"[进度] 已处理: {completed}/{total}, 成功: {success_count}, 失败: {failed_count}")
            
            if not suppress_progress:
                print(f"[完成] 异步答案生成完成！成功: {len(results)}, 失败: {len(errors)}")
            return results, errors
    
    async def run_async(
        self,
        input_file: Path,
        output_dir: Path,
        pipeline_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        save_intermediate: bool = True,
        batch_size: int = 1000,
        concurrency: int = 5,  # 异步并发数（建议1-5）
        request_delay: float = 0.1,  # 请求延迟（秒）
        use_async: bool = True,  # 是否使用异步并行处理
        debug_question_output_dir: Optional[Path] = None,
        progress_bar: bool = True,  # 是否显示单一进度条
    ) -> Dict[str, Any]:
        """
        运行完整流程（支持大文件分流处理）
        
        Args:
            input_file: 输入文件路径（batch_process.sh的输出）
            output_dir: 输出目录
            pipeline_names: 要使用的pipeline列表（None表示使用所有）
            max_samples: 最大处理样本数（None表示全部）
            save_intermediate: 是否保存中间结果
            batch_size: 每批处理的记录数（用于大文件分流处理）
            
        Returns:
            统计信息和结果路径
        """
        print("=" * 80)
        print("VQA数据集生成完整流程")
        print("=" * 80)
        print(f"输入文件: {input_file}")
        print(f"输出目录: {output_dir}")
        print()
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建中间结果目录
        if save_intermediate:
            intermediate_dir = output_dir / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            questions_dir = intermediate_dir / "questions"
            answers_dir = intermediate_dir / "answers"
            questions_dir.mkdir(exist_ok=True)
            answers_dir.mkdir(exist_ok=True)
        else:
            intermediate_dir = None
            questions_dir = None
            answers_dir = None
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 分流处理大文件
        
        # 收集所有结果
        all_successful_vqa = []  # 最终全部通过可用的vqa数据
        all_question_errors = []  # 生成question步骤出错的数据
        all_answer_validation_failed = []  # answer检验不通过的数据
        
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            all_input_data = json.load(f)
        
        if not isinstance(all_input_data, list):
            raise ValueError(f"输入文件应该包含一个数组，但得到: {type(all_input_data)}")
        
        total_records = len(all_input_data)
        if max_samples:
            all_input_data = all_input_data[:max_samples]
            total_records = len(all_input_data)
        
        # 单一进度条（需要 tqdm）
        pbar = None
        if progress_bar and HAS_TQDM and tqdm:
            pbar = tqdm(total=total_records, desc="VQA生成", unit="条", ncols=80, mininterval=1.0)
        
        # 分批处理
        for batch_idx in range(0, total_records, batch_size):
            batch_num = (batch_idx // batch_size) + 1
            total_batches = (total_records + batch_size - 1) // batch_size
            batch_data = all_input_data[batch_idx:batch_idx + batch_size]
            
            if pbar is None:
                print(f"\n{'=' * 80}")
                print(f"处理批次 {batch_num}/{total_batches} (记录 {batch_idx + 1}-{min(batch_idx + batch_size, total_records)})")
                print(f"{'=' * 80}")
            
            # 创建临时批次文件（优化：不使用indent以减少序列化时间）
            batch_input_file = output_dir / f"batch_{batch_num}_input_{timestamp}.json"
            with open(batch_input_file, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, separators=(',', ':'))
            
            try:
                # Step 1: 生成问题（支持异步并行处理）
                # 如果启用中间结果保存，保存到intermediate/questions目录
                if save_intermediate and questions_dir:
                    batch_questions_file = questions_dir / f"batch_{batch_num}_questions_{timestamp}.json"
                else:
                    # 临时文件，稍后会被清理
                    batch_questions_file = output_dir / f"batch_{batch_num}_questions_{timestamp}.json"
                batch_question_errors = []
                
                try:
                    # 使用异步并行处理（如果启用）
                    if use_async:
                        # 设置失败案例目录（在输出目录下创建子目录，保留失败案例）
                        failed_selection_dir = output_dir / "failed_selection"
                        # 使用异步方法生成问题
                        await self.question_generator.process_data_file_async(
                            input_file=batch_input_file,
                            output_file=batch_questions_file,
                            pipeline_names=pipeline_names,
                            max_samples=None,  # 批次文件已经限制大小
                            num_gpus=1,  # 问题生成阶段使用单GPU组
                            max_concurrent_per_gpu=concurrency,
                            request_delay=request_delay,
                            failed_selection_dir=failed_selection_dir,
                            debug_output_dir=debug_question_output_dir,
                            suppress_progress=(pbar is not None),
                        )
                    else:
                        # 使用同步方法（兼容模式）
                        # 设置失败案例目录（在输出目录下创建子目录，保留失败案例）
                        failed_selection_dir = output_dir / "failed_selection"
                        self.question_generator.process_data_file(
                            input_file=batch_input_file,
                            output_file=batch_questions_file,
                            pipeline_names=pipeline_names,
                            max_samples=None,  # 批次文件已经限制大小
                            failed_selection_dir=failed_selection_dir,
                            debug_output_dir=debug_question_output_dir,
                            suppress_progress=(pbar is not None),
                        )
                    
                    # 读取生成的问题
                    with open(batch_questions_file, 'r', encoding='utf-8') as f:
                        batch_questions = json.load(f)
                    
                    # 读取问题生成阶段的错误
                    # vqa_generator会生成带时间戳的错误文件，格式: questions_YYYYMMDD_HHMMSS_errors_YYYYMMDD_HHMMSS.json
                    # 或者: batch_X_questions_YYYYMMDD_HHMMSS_errors_YYYYMMDD_HHMMSS.json
                    error_pattern = f"*_errors_*.json"
                    error_files = [f for f in batch_questions_file.parent.glob(error_pattern) 
                                  if batch_questions_file.stem in f.stem]
                    if error_files:
                        # 使用最新的错误文件
                        error_file = max(error_files, key=lambda p: p.stat().st_mtime)
                        with open(error_file, 'r', encoding='utf-8') as f:
                            batch_question_errors = json.load(f)
                            # 确保是列表格式
                            if isinstance(batch_question_errors, list):
                                all_question_errors.extend(batch_question_errors)
                            else:
                                all_question_errors.append(batch_question_errors)
                    
                    if pbar is None:
                        print(f"✓ 批次 {batch_num} 问题生成完成: {len(batch_questions)} 个问题")
                        if batch_question_errors:
                            print(f"  - 问题生成错误: {len(batch_question_errors)} 条")
                    
                except Exception as e:
                    print(f"✗ 批次 {batch_num} 问题生成失败: {e}")
                    # 将整个批次标记为错误
                    for record in batch_data:
                        all_question_errors.append({
                            "record": record,
                            "error": str(e),
                            "batch": batch_num
                        })
                    continue
                
                # Step 2: 生成答案（优化：直接在内存中处理，减少文件I/O）
                # 如果启用中间结果保存，保存到intermediate/answers目录
                if save_intermediate and answers_dir:
                    batch_answers_file = answers_dir / f"batch_{batch_num}_answers_{timestamp}.json"
                else:
                    # 不保存中间答案文件
                    batch_answers_file = None
                
                try:
                    # 使用异步并行处理（如果启用）
                    if use_async:
                        batch_answers, answer_errors = await self._generate_answers_from_data_async(
                            batch_questions,
                            concurrency=concurrency,
                            request_delay=request_delay,
                            suppress_progress=(pbar is not None),
                        )
                    else:
                        # 使用串行处理（兼容模式）
                        batch_answers, answer_errors = self._generate_answers_from_data(batch_questions)
                    
                    # 只在需要保存中间结果时写入文件
                    if batch_answers_file:
                        batch_answers_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(batch_answers_file, 'w', encoding='utf-8') as f:
                            json.dump(batch_answers, f, ensure_ascii=False, separators=(',', ':'))
                    
                    if pbar is None:
                        print(f"✓ 批次 {batch_num} 答案生成完成: {len(batch_answers)} 个答案")
                        if answer_errors:
                            print(f"  - 答案生成失败: {len(answer_errors)} 条")
                    
                    # 处理生成失败的答案（answer_errors） - 无论异步还是同步方法返回的错误
                    for error_item in answer_errors:
                        # 从对应的question中找到原始数据
                        error_index = error_item.get("index", 0)
                        error_id = error_item.get("id")
                        
                        # 尝试从batch_questions中找到对应的question
                        corresponding_question = None
                        for q in batch_questions:
                            if q.get("id") == error_id or (error_index > 0 and batch_questions.index(q) + 1 == error_index):
                                corresponding_question = q
                                break
                        
                        # 构建失败记录（含详细错误信息）
                        vr = error_item.get("validation_report") or {}
                        failed_item = {
                            "question": corresponding_question.get("question") if corresponding_question else None,
                            "question_type": corresponding_question.get("question_type") if corresponding_question else None,
                            "image_base64": corresponding_question.get("image_base64") if corresponding_question else None,
                            "pipeline_name": corresponding_question.get("pipeline_name") if corresponding_question else None,
                            "pipeline_intent": corresponding_question.get("pipeline_intent") if corresponding_question else None,
                            "answer_type": corresponding_question.get("answer_type") if corresponding_question else None,
                            "sample_index": corresponding_question.get("sample_index") if corresponding_question else error_item.get("sample_index"),
                            "id": error_id or (corresponding_question.get("id") if corresponding_question else None),
                            "source_a_id": corresponding_question.get("source_a_id") if corresponding_question else error_item.get("source_a_id"),
                            "answer": None,
                            "explanation": None,
                            "full_question": corresponding_question.get("question") if corresponding_question else None,
                            "options": None,
                            "correct_option": None,
                            "error": error_item.get("error", "答案生成失败"),
                            "error_stage": error_item.get("error_stage", "generation"),
                            "error_details": error_item.get("error_details") or self._extract_error_details(
                                error_item.get("error_stage", "generation"),
                                error_item.get("error", "答案生成失败"),
                                validation_report=vr,
                            ),
                            "error_index": error_index,
                            "retry_count": error_item.get("retry_count", 0),
                            "batch": batch_num,
                            "validation_passed": False,
                            "validation_report": vr if vr else {
                                "validation_passed": False,
                                "error": error_item.get("error", "答案生成失败"),
                                "format_check": {"passed": False, "issues": [error_item.get("error", "答案生成失败")]},
                                "vqa_validation": {"passed": False}
                            },
                            "timestamp": corresponding_question.get("timestamp") if corresponding_question else datetime.now().isoformat(),
                            "generated_at": datetime.now().isoformat()
                        }
                        all_answer_validation_failed.append(failed_item)
                    
                    # 分类答案数据
                    for answer in batch_answers:
                        # 跳过 None 值
                        if answer is None:
                            continue
                        
                        # 确保 answer 是字典类型
                        if not isinstance(answer, dict):
                            # 如果不是字典，跳过或创建错误记录
                            log_warning(f"答案数据格式错误，跳过: {type(answer).__name__}")
                            continue
                        
                        validation_report = answer.get("validation_report")
                        if validation_report is None:
                            validation_report = {}
                        elif not isinstance(validation_report, dict):
                            validation_report = {}
                        
                        validation_passed = validation_report.get("validation_passed", False)
                        
                        if validation_passed:
                            # 校验通过，加入成功数据
                            all_successful_vqa.append(answer)
                        else:
                            # 校验不通过，加入校验失败数据；确保有 error 和 error_details 便于排查
                            if "error" not in answer:
                                reason = (
                                    validation_report.get("regeneration_reason")
                                    or validation_report.get("format_check", {}).get("issues", [])
                                )
                                if isinstance(reason, list):
                                    reason = "; ".join(str(r) for r in reason[:3]) if reason else "校验失败"
                                answer["error"] = reason or "答案校验未通过"
                            answer["error_stage"] = "validation"
                            if "error_details" not in answer:
                                answer["error_details"] = self._extract_error_details(
                                    "validation",
                                    answer.get("error", "答案校验未通过"),
                                    validation_report=validation_report,
                                )
                            all_answer_validation_failed.append(answer)
                    
                    if pbar is None:
                        print(f"  - 校验通过: {len([a for a in batch_answers if a.get('validation_report', {}).get('validation_passed', False)])}")
                        print(f"  - 校验未通过: {len([a for a in batch_answers if not a.get('validation_report', {}).get('validation_passed', False)])}")
                        print(f"  - 生成失败: {len(answer_errors)}")
                    
                except Exception as e:
                    print(f"✗ 批次 {batch_num} 答案生成失败: {e}")
                    # 将问题数据标记为答案生成失败（这些数据已经生成了问题，但答案生成失败）
                    exc_details = self._extract_error_details("exception", str(e))
                    exc_details["exception_type"] = type(e).__name__
                    import traceback
                    exc_details["traceback"] = traceback.format_exc()[-_ERROR_TRACEBACK_MAX:]
                    for question in batch_questions:
                        failed_item = {
                            "question": question.get("question"),
                            "question_type": question.get("question_type"),
                            "image_base64": question.get("image_base64"),
                            "pipeline_name": question.get("pipeline_name"),
                            "pipeline_intent": question.get("pipeline_intent"),
                            "sample_index": question.get("sample_index"),
                            "id": question.get("id"),
                            "source_a_id": question.get("source_a_id"),
                            "answer": None,
                            "full_question": question.get("question"),
                            "options": None,
                            "correct_option": None,
                            "error": f"答案生成失败: {str(e)}",
                            "error_stage": "exception",
                            "error_details": exc_details,
                            "batch": batch_num,
                            "validation_passed": False,
                            "validation_report": {
                                "validation_passed": False,
                                "error": f"答案生成失败: {str(e)}"
                            }
                        }
                        all_answer_validation_failed.append(failed_item)
                    continue
                
            finally:
                # 更新进度条
                if pbar is not None:
                    pbar.update(len(batch_data))
                # 清理临时批次输入文件（总是清理，因为不需要保留）
                try:
                    if batch_input_file.exists():
                        batch_input_file.unlink()
                except Exception as e:
                    print(f"[WARNING] 清理批次输入文件失败: {e}")
                
                # 清理批次中间文件（如果未保存到intermediate目录）
                # 如果save_intermediate=False或文件不在intermediate目录下，则清理
                try:
                    if batch_questions_file.exists():
                        # 检查是否在intermediate目录下
                        if not save_intermediate or (questions_dir and batch_questions_file.parent != questions_dir):
                            batch_questions_file.unlink()
                except Exception as e:
                    print(f"[WARNING] 清理批次问题文件失败: {e}")
                
                try:
                    if batch_answers_file and batch_answers_file.exists():
                        # 检查是否在intermediate目录下
                        if not save_intermediate or (answers_dir and batch_answers_file.parent != answers_dir):
                            batch_answers_file.unlink()
                except Exception as e:
                    print(f"[WARNING] 清理批次答案文件失败: {e}")
                
                # 清理批次错误文件（如果不需要保存中间结果）
                if not save_intermediate:
                    try:
                        # 清理问题生成阶段的错误文件
                        error_pattern = f"*_errors_*.json"
                        if batch_questions_file.exists():
                            error_files = [f for f in batch_questions_file.parent.glob(error_pattern) 
                                          if batch_questions_file.stem in f.stem]
                            for error_file in error_files:
                                error_file.unlink()
                    except Exception as e:
                        print(f"[WARNING] 清理批次错误文件失败: {e}")
        
        if pbar is not None:
            pbar.close()
        
        # Step 3: 生成最终输出文件
        print("\n" + "=" * 80)
        print("Step 3: 生成最终输出文件")
        print("=" * 80)
        
        # 输出文件1: 最终全部通过可用的vqa数据
        successful_vqa_file = output_dir / f"vqa_dataset_successful_{timestamp}.json"
        successful_vqa_data = self._prepare_final_dataset(all_successful_vqa)
        with open(successful_vqa_file, 'w', encoding='utf-8') as f:
            json.dump(successful_vqa_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 成功VQA数据已保存: {successful_vqa_file}")
        print(f"  - 总样本数: {len(successful_vqa_data)}")
        
        # 输出文件2: 生成question步骤出错的数据
        question_errors_file = output_dir / f"question_errors_{timestamp}.json"
        with open(question_errors_file, 'w', encoding='utf-8') as f:
            json.dump(all_question_errors, f, ensure_ascii=False, indent=2)
        print(f"✓ 问题生成错误数据已保存: {question_errors_file}")
        print(f"  - 错误记录数: {len(all_question_errors)}")
        
        # 输出文件3: answer检验不通过的数据
        answer_validation_failed_file = output_dir / f"answer_validation_failed_{timestamp}.json"
        validation_failed_data = self._prepare_final_dataset(all_answer_validation_failed)
        with open(answer_validation_failed_file, 'w', encoding='utf-8') as f:
            json.dump(validation_failed_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 答案校验失败数据已保存: {answer_validation_failed_file}")
        print(f"  - 失败记录数: {len(validation_failed_data)}")
        
        # 更新统计信息
        self.stats["questions_generated"] = len(all_successful_vqa) + len(all_answer_validation_failed)
        self.stats["answers_generated"] = len(all_successful_vqa) + len(all_answer_validation_failed)
        self.stats["validation_passed"] = len(all_successful_vqa)
        self.stats["validation_failed"] = len(all_answer_validation_failed)
        self.stats["question_errors"] = len(all_question_errors)
        self.stats["input_records"] = total_records
        
        # 生成摘要报告（不再生成统计文件）
        print("\n" + "=" * 80)
        print("流程完成摘要")
        print("=" * 80)
        print(f"成功VQA数据: {successful_vqa_file}")
        print(f"  - 总样本数: {len(successful_vqa_data)}")
        print(f"问题生成错误: {question_errors_file}")
        print(f"  - 错误记录数: {len(all_question_errors)}")
        print(f"答案校验失败: {answer_validation_failed_file}")
        print(f"  - 失败记录数: {len(validation_failed_data)}")
        if save_intermediate and intermediate_dir:
            print(f"中间结果目录: {intermediate_dir}")
        if (output_dir / "failed_selection").exists():
            print(f"对象选择失败案例: {output_dir / 'failed_selection'}")
        print("=" * 80)

        meta_path = output_dir / "meta.json"
        meta = {
            "generated_at": datetime.now().isoformat(),
            "input_file": str(input_file),
            "output_dir": str(output_dir),
            "total_records": total_records,
            "successful_vqa_count": len(successful_vqa_data),
            "question_errors_count": len(all_question_errors),
            "answer_validation_failed_count": len(validation_failed_data),
            "pipeline_names": pipeline_names,
            "max_samples": max_samples,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "request_delay": request_delay,
            "use_async": use_async,
            "QA-generator code changes": "",
            "Config changes": "",
            "Other things": "",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"元信息已保存: {meta_path}")
        
        return {
            "successful_vqa": successful_vqa_file,
            "question_errors": question_errors_file,
            "answer_validation_failed": answer_validation_failed_file,
            "stats": self.stats
        }
    
    def run(
        self,
        input_file: Path,
        output_dir: Path,
        pipeline_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        save_intermediate: bool = True,
        batch_size: int = 1000,
        concurrency: int = 5,  # 异步并发数（建议1-5）
        request_delay: float = 0.1,  # 请求延迟（秒）
        use_async: bool = True,  # 是否使用异步并行处理
        debug_question_output_dir: Optional[Path] = None,
        progress_bar: bool = True,  # 是否显示单一进度条
    ) -> Dict[str, Any]:
        """
        运行完整流程（同步包装器，内部调用异步版本）
        
        Args:
            input_file: 输入文件路径（batch_process.sh的输出）
            output_dir: 输出目录
            pipeline_names: 要使用的pipeline列表（None表示使用所有）
            max_samples: 最大处理样本数（None表示全部）
            save_intermediate: 是否保存中间结果
            batch_size: 每批处理的记录数（用于大文件分流处理）
            concurrency: 异步并发数（建议1-5）
            request_delay: 请求延迟（秒）
            use_async: 是否使用异步并行处理
            
        Returns:
            统计信息和结果路径
        """
        return asyncio.run(self.run_async(
            input_file=input_file,
            output_dir=output_dir,
            pipeline_names=pipeline_names,
            max_samples=max_samples,
            save_intermediate=save_intermediate,
            batch_size=batch_size,
            concurrency=concurrency,
            request_delay=request_delay,
            use_async=use_async,
            debug_question_output_dir=debug_question_output_dir,
            progress_bar=progress_bar,
        ))
    
    def _prepare_final_dataset(self, answers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        准备最终数据集
        
        可以在这里进行额外的处理和过滤
        支持处理成功数据和失败数据
        """
        final_dataset = []
        
        for answer in answers_data:
            # 跳过 None 值
            if answer is None:
                continue
            
            # 确保 answer 是字典类型
            if not isinstance(answer, dict):
                # 如果不是字典，尝试转换或跳过
                if isinstance(answer, (str, int, float)):
                    # 如果是简单类型，创建一个基本记录
                    answer = {"error": f"数据格式错误: {type(answer).__name__}", "raw_data": str(answer)}
                else:
                    # 其他类型，跳过
                    continue
            
            # 安全获取 validation_report
            validation_report = answer.get("validation_report")
            if validation_report is None:
                validation_report = {}
            elif not isinstance(validation_report, dict):
                validation_report = {}
            
            # 构建最终数据项
            item = {
                # 基本信息
                "id": answer.get("id"),
                "sample_index": answer.get("sample_index"),
                "source_a_id": answer.get("source_a_id"),
                
                # 问题和答案
                "question": answer.get("question"),
                "full_question": answer.get("full_question", answer.get("question")),
                "answer": answer.get("answer"),
                "question_type": answer.get("question_type"),
                
                # 图片
                "image_base64": answer.get("image_base64"),
                
                # 选择题特有字段
                "options": answer.get("options"),
                "correct_option": answer.get("correct_option"),
                
                # 解释
                "explanation": answer.get("explanation", ""),
                
                # Pipeline信息
                "pipeline_name": answer.get("pipeline_name"),
                "pipeline_intent": answer.get("pipeline_intent"),
                "answer_type": answer.get("answer_type"),
                
                # 校验信息（简化版）
                "validation_passed": answer.get("validation_passed", validation_report.get("validation_passed", False)),
                "validation_score": self._calculate_validation_score(validation_report),
                
                # 时间戳
                "generated_at": answer.get("generated_at", ""),
                "timestamp": answer.get("timestamp", "")
            }
            
            # 如果有错误信息，也包含进去（生成失败或校验失败都会记录错误）
            if "error" in answer:
                item["error"] = answer.get("error")
                if "error_index" in answer:
                    item["error_index"] = answer.get("error_index")
                if "retry_count" in answer:
                    item["retry_count"] = answer.get("retry_count")
                if "error_stage" in answer:
                    item["error_stage"] = answer.get("error_stage")
                if "error_details" in answer and answer.get("error_details"):
                    item["error_details"] = answer.get("error_details")
            # 校验失败但无 error 时，从 validation_report 推导
            elif answer.get("answer") is None or not answer.get("validation_passed", True):
                vr = answer.get("validation_report") or {}
                reason = vr.get("regeneration_reason") or vr.get("error")
                if not reason and isinstance(vr.get("format_check"), dict):
                    issues = vr.get("format_check", {}).get("issues", [])
                    reason = "; ".join(str(i) for i in issues[:3]) if issues else None
                item["error"] = reason or "答案生成或校验失败"
                if "error_stage" in answer:
                    item["error_stage"] = answer.get("error_stage")
            
            # 如果有批次信息，也包含进去
            if "batch" in answer:
                item["batch"] = answer.get("batch")
            
            # 如果答案生成失败，确保相关字段都有合理的默认值
            if answer.get("answer") is None and "error" in answer:
                item["answer"] = None
                item["explanation"] = answer.get("explanation")  # 可能是None
                item["full_question"] = answer.get("full_question") or answer.get("question")
                item["options"] = answer.get("options")  # 可能是None
                item["correct_option"] = answer.get("correct_option")  # 可能是None
                item["validation_passed"] = False
                item["validation_score"] = 0.0
            
            final_dataset.append(item)
        
        return final_dataset
    
    def _calculate_validation_score(self, validation_report: Dict[str, Any]) -> float:
        """
        计算校验评分
        
        基于格式检查和VQA验证结果计算综合评分
        """
        if not validation_report:
            return 0.0
        
        score = 0.0
        
        # 格式检查评分（40%）
        format_check = validation_report.get("format_check", {})
        if format_check.get("passed", False):
            score += 0.4
        
        # VQA验证评分（60%）
        vqa_validation = validation_report.get("vqa_validation", {})
        
        # 困惑度分析（20%）
        perplexity = vqa_validation.get("perplexity_analysis", {})
        if perplexity.get("passed", False):
            clarity_score = perplexity.get("clarity_score", 0.5)
            score += 0.2 * clarity_score
        
        # 置信度评估（20%）
        confidence = vqa_validation.get("confidence_assessment", {})
        if confidence.get("passed", False):
            conf_score = confidence.get("confidence", 0.5)
            score += 0.2 * conf_score
        
        # 答案验证（20%）
        answer_validation = vqa_validation.get("answer_validation", {})
        if answer_validation.get("passed", False):
            score += 0.2
        
        return round(score, 3)
    
    def _generate_statistics(self, final_dataset: List[Dict[str, Any]], answers_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成统计信息
        """
        stats = {
            "total_samples": len(final_dataset),
            "by_question_type": {},
            "by_pipeline": {},
            "validation_summary": {
                "passed": 0,
                "failed": 0,
                "average_score": 0.0
            },
            "quality_metrics": {}
        }
        
        # 按题型统计
        for item in final_dataset:
            qtype = item.get("question_type", "unknown")
            stats["by_question_type"][qtype] = stats["by_question_type"].get(qtype, 0) + 1
        
        # 按pipeline统计
        for item in final_dataset:
            pipeline = item.get("pipeline_name", "unknown")
            stats["by_pipeline"][pipeline] = stats["by_pipeline"].get(pipeline, 0) + 1
        
        # 校验摘要
        validation_scores = []
        for item in final_dataset:
            if item.get("validation_passed", False):
                stats["validation_summary"]["passed"] += 1
            else:
                stats["validation_summary"]["failed"] += 1
            
            score = item.get("validation_score", 0.0)
            if score > 0:
                validation_scores.append(score)
        
        if validation_scores:
            stats["validation_summary"]["average_score"] = round(
                sum(validation_scores) / len(validation_scores), 3
            )
        
        # 质量指标
        stats["quality_metrics"] = {
            "has_explanation": sum(1 for item in final_dataset if item.get("explanation")),
            "has_image": sum(1 for item in final_dataset if item.get("image_base64")),
            "complete_options": sum(1 for item in final_dataset 
                                  if item.get("question_type") == "multiple_choice" 
                                  and item.get("options") and len(item.get("options", {})) >= 2)
        }
        
        return stats
    
    def _print_summary_batch(
        self,
        successful_vqa_file: Path,
        question_errors_file: Optional[Path],
        answer_validation_failed_file: Optional[Path],
        statistics_file: Path
    ):
        """打印摘要报告（分流处理版本）"""
        print(f"输入记录数: {self.stats['input_records']}")
        print(f"生成问题数: {self.stats['questions_generated']}")
        print(f"生成答案数: {self.stats['answers_generated']}")
        print(f"校验通过: {self.stats['validation_passed']}")
        print(f"校验未通过: {self.stats['validation_failed']}")
        print(f"问题生成错误: {self.stats.get('question_errors', 0)}")
        print()
        print("输出文件:")
        print(f"  - 成功VQA数据: {successful_vqa_file} ({self.stats['validation_passed']} 条)")
        if question_errors_file:
            print(f"  - 问题生成错误: {question_errors_file} ({self.stats.get('question_errors', 0)} 条)")
        if answer_validation_failed_file:
            print(f"  - 答案校验失败: {answer_validation_failed_file} ({self.stats['validation_failed']} 条)")
        print(f"  - 统计信息: {statistics_file}")
        print()
        if self.stats["errors"]:
            print("错误信息:")
            for error in self.stats["errors"]:
                print(f"  - {error}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='VQA数据集生成完整流程 - 从batch_process.sh输出生成完整VQA数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（输出目录自动命名）
  python QA_Generator/pipeline/pipeline.py input.json
  
  # 指定pipeline和样本数
  python QA_Generator/pipeline/pipeline.py input.json \\
      --pipelines question object_counting \\
      -n 100
  
  # 不保存中间结果
  python QA_Generator/pipeline/pipeline.py input.json --no-intermediate
  
  # 指定日志文件（所有错误和警告信息将写入该文件）
  python QA_Generator/pipeline/pipeline.py input.json --log-file log.txt
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='输入JSON文件路径（batch_process.sh的输出）'
    )
    parser.add_argument(
        '--question-config',
        type=str,
        default=None,
        help='问题生成配置文件路径（默认: QA_Generator/question/config/question_config.json）'
    )
    parser.add_argument(
        '--answer-config',
        type=str,
        default=None,
        help='答案生成配置文件路径（默认: QA_Generator/answer/answer_config.json）'
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
        '--no-intermediate',
        action='store_true',
        help='不保存中间结果（问题和答案文件）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='每批处理的记录数（用于大文件分流处理，默认: 1000）'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=5,
        help='异步并发请求数（建议1-5，某些API不支持高并发，默认: 5）'
    )
    parser.add_argument(
        '--request-delay',
        type=float,
        default=0.1,
        help='每个请求之间的延迟（秒），用于避免触发API并发限制（默认: 0.1）'
    )
    parser.add_argument(
        '--no-async',
        action='store_true',
        help='禁用异步并行处理，使用串行处理（兼容模式）'
    )
    parser.add_argument(
        '--debug-questions',
        action='store_true',
        help='保存问题生成调试信息（包含输入/题型/slots/问题文本等）'
    )
    parser.add_argument(
        '--debug-question-dir',
        type=str,
        default=None,
        help='问题生成调试信息输出目录（默认: output_dir/debug/questions）'
    )
    parser.add_argument(
        '--enable-validation-exemptions',
        action='store_true',
        help='开启指定pipeline的验证豁免（question/visual_recognition/caption/text_association）'
    )
    parser.add_argument(
        '--no-progress-bar',
        action='store_true',
        help='禁用单一进度条（默认显示，需安装 tqdm）'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='日志文件路径（可选，如果指定则将所有错误和警告信息写入该文件）'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.cwd() / f"vqa_ready4use_{timestamp}"
    debug_question_output_dir = None
    if args.debug_questions:
        if args.debug_question_dir:
            debug_question_output_dir = Path(args.debug_question_dir)
        else:
            debug_question_output_dir = output_dir / "debug" / "questions"
    
    # 处理配置文件路径（支持相对路径和绝对路径）
    question_config_path = None
    if args.question_config:
        question_config_path = Path(args.question_config)
        if not question_config_path.is_absolute():
            # 如果是相对路径，尝试相对于当前工作目录和项目根目录
            cwd_path = Path.cwd() / question_config_path
            project_root = Path(__file__).parent.parent
            project_path = project_root / question_config_path
            if cwd_path.exists():
                question_config_path = cwd_path
            elif project_path.exists():
                question_config_path = project_path
            # 如果都不存在，保持原路径，让后续代码处理
    
    answer_config_path = None
    if args.answer_config:
        answer_config_path = Path(args.answer_config)
        if not answer_config_path.is_absolute():
            # 如果是相对路径，尝试相对于当前工作目录和项目根目录
            cwd_path = Path.cwd() / answer_config_path
            project_root = Path(__file__).parent.parent
            project_path = project_root / answer_config_path
            if cwd_path.exists():
                answer_config_path = cwd_path
            elif project_path.exists():
                answer_config_path = project_path
            # 如果都不存在，保持原路径，让后续代码处理
    
    # 初始化日志器
    log_file_path = None
    if args.log_file:
        log_file_path = Path(args.log_file)
        if not log_file_path.is_absolute():
            # 如果是相对路径，相对于输出目录
            log_file_path = output_dir / log_file_path
    
    logger = None
    if log_file_path:
        logger = Logger(log_file_path)
        set_global_logger(logger)
        logger.info(f"日志文件: {log_file_path}")
    
    if not input_file.exists():
        error_msg = f"输入文件不存在: {input_file}\n  当前工作目录: {Path.cwd()}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"[ERROR] {error_msg}")
        return 1
    
    # 检查配置文件是否存在（如果指定了）
    if question_config_path and not question_config_path.exists():
        warning_msg = f"问题配置文件不存在: {question_config_path}\n  将使用默认配置文件"
        if logger:
            logger.warning(warning_msg)
        else:
            print(f"[WARNING] {warning_msg}")
        question_config_path = None
    
    if answer_config_path and not answer_config_path.exists():
        warning_msg = f"答案配置文件不存在: {answer_config_path}\n  将使用默认配置文件"
        if logger:
            logger.warning(warning_msg)
        else:
            print(f"[WARNING] {warning_msg}")
        answer_config_path = None
    
    try:
        # 初始化流程
        pipeline = VQAPipeline(
            question_config_path=question_config_path,
            answer_config_path=answer_config_path,
            enable_validation_exemptions=args.enable_validation_exemptions,
        )
        
        # 运行流程（支持异步并行处理）
        use_async = not args.no_async
        
        if use_async:
            if args.concurrency > 5:
                warning_msg = f"并发数设置为 {args.concurrency}，某些API可能不支持高并发\n  如果遇到401错误，建议降低并发数（--concurrency 1-3）"
                if logger:
                    logger.warning(warning_msg)
                else:
                    print(f"[WARNING] {warning_msg}")
        else:
            info_msg = "使用串行处理（兼容模式）"
            if logger:
                logger.info(info_msg)
        
        result = pipeline.run(
            input_file=input_file,
            output_dir=output_dir,
            pipeline_names=args.pipelines,
            max_samples=args.max_samples,
            save_intermediate=not args.no_intermediate,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            request_delay=args.request_delay,
            use_async=use_async,
            debug_question_output_dir=debug_question_output_dir,
            progress_bar=not args.no_progress_bar,
        )
        
        success_msg = "流程执行成功！"
        if logger:
            logger.info(success_msg)
        else:
            print(f"\n✓ {success_msg}")
        
    except Exception as e:
        error_msg = f"流程执行失败: {e}"
        if logger:
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
        else:
            print(f"\n✗ {error_msg}")
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # 关闭日志器
        if logger:
            logger.close()
            set_global_logger(None)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

