"""
答案生成模块
根据问题和图片生成答案
"""
import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from QA_Generator.clients.gemini_client import GeminiClient
from QA_Generator.clients.async_client import AsyncGeminiClient
from QA_Generator.logging.logger import get_logger, log_warning, log_error, log_debug
from QA_Generator.utils.model_response_logger import log_model_response
from QA_Generator.utils.json_parse_utils import extract_json_from_response

# 错误记录中模型原始响应的最大保留长度
_RAW_RESPONSE_MAX = 500


class AnswerGenerator:
    """答案生成器"""
    
    def __init__(self, config_path: Optional[Path] = None, gemini_client: Optional[GeminiClient] = None):
        """
        初始化答案生成器
        
        Args:
            config_path: 配置文件路径（可选）
            gemini_client: Gemini客户端实例（可选）
        """
        self.gemini_client = gemini_client or GeminiClient()
        
        # 加载配置
        if config_path and config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # 默认配置
            self.config = {
                "multiple_choice": {
                    "wrong_options": {
                        "min_count": 2,
                        "max_count": 4
                    }
                },
                "generation_settings": {
                    "temperature": 0.7,
                    "max_tokens": 512
                }
            }
        
        self.mc_config = self.config.get("multiple_choice", {})
        self.gen_settings = self.config.get("generation_settings", {})
    
    def generate_answer(
        self,
        question: str,
        image_base64: str,
        question_type: str,
        pipeline_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成答案
        
        Args:
            question: 问题文本
            image_base64: 图片的base64编码
            question_type: 题型，"multiple_choice" 或 "fill_in_blank"
            pipeline_info: 可选的pipeline信息（用于上下文）
            
        Returns:
            包含answer、explanation等字段的字典
        """
        if question_type == "multiple_choice":
            return self._generate_multiple_choice_answer(
                question=question,
                image_base64=image_base64,
                pipeline_info=pipeline_info
            )
        elif question_type == "fill_in_blank":
            return self._generate_fill_in_blank_answer(
                question=question,
                image_base64=image_base64,
                pipeline_info=pipeline_info
            )
        else:
            raise ValueError(f"不支持的题型: {question_type}")
    
    async def generate_answer_async(
        self,
        question: str,
        image_base64: str,
        question_type: str,
        pipeline_info: Optional[Dict[str, Any]] = None,
        async_client: Optional[AsyncGeminiClient] = None,
        model: Optional[str] = None,
        retry_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        异步生成答案（使用 AsyncGeminiClient，与vlmtool/generate_vqa对齐）
        
        Args:
            question: 问题文本
            image_base64: 图片的base64编码
            question_type: 题型，"multiple_choice" 或 "fill_in_blank"
            pipeline_info: 可选的pipeline信息（用于上下文）
            async_client: 异步客户端实例（可选）
            model: 模型名称（可选，如果为None则从async_client或config读取）
            retry_context: 重试时的上下文，包含 previous_answer, regeneration_reason, retry_count，
                          用于在 prompt 中注入失败反馈，避免无效重复
        """
        if question_type == "multiple_choice":
            return await self._generate_multiple_choice_answer_async(
                question=question,
                image_base64=image_base64,
                pipeline_info=pipeline_info,
                async_client=async_client,
                model=model,
                retry_context=retry_context,
            )
        elif question_type == "fill_in_blank":
            return await self._generate_fill_in_blank_answer_async(
                question=question,
                image_base64=image_base64,
                pipeline_info=pipeline_info,
                async_client=async_client,
                model=model,
                retry_context=retry_context,
            )
        else:
            raise ValueError(f"不支持的题型: {question_type}")
    
    def _generate_multiple_choice_answer(
        self,
        question: str,
        image_base64: str,
        pipeline_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成选择题答案
        
        步骤：
        1. 生成正确答案
        2. 生成错误选项
        3. 打乱顺序，组合成完整选择题
        4. 确定正确答案的选项字母
        """
        # Step 1: 生成正确答案
        correct_answer = self._generate_correct_answer(
            question=question,
            image_base64=image_base64,
            pipeline_info=pipeline_info
        )
        
        if not correct_answer or not correct_answer.get("answer"):
            return {
                "answer": None,
                "explanation": "无法生成正确答案",
                "full_question": question,
                "options": None,
                "error": "无法生成正确答案",
                "error_type": "generation",
                "error_step": "correct_answer",
            }
        
        correct_answer_text = correct_answer["answer"]
        explanation = correct_answer.get("explanation", "")
        
        # Step 2: 生成错误选项
        wrong_options = self._generate_wrong_options(
            question=question,
            image_base64=image_base64,
            correct_answer=correct_answer_text,
            pipeline_info=pipeline_info
        )
        
        if not wrong_options:
            return {
                "answer": None,
                "explanation": explanation,
                "full_question": question,
                "options": None,
                "error": "错误选项生成失败",
                "error_type": "generation",
                "error_step": "wrong_options",
            }
        
        # Step 3: 组合所有选项并打乱顺序
        all_options = [correct_answer_text] + wrong_options
        random.shuffle(all_options)
        
        # Step 4: 确定正确答案的选项字母
        correct_index = all_options.index(correct_answer_text)
        correct_letter = chr(65 + correct_index)  # A, B, C, D...
        
        # Step 5: 构建完整选择题
        options_text = "\n".join([f"{chr(65 + i)}: {option}" for i, option in enumerate(all_options)])
        full_question = f"{question}\n{options_text}"
        
        return {
            "answer": correct_letter,  # 如 "A", "B", "C"
            "explanation": explanation,
            "full_question": full_question,
            "options": {chr(65 + i): option for i, option in enumerate(all_options)},
            "correct_option": correct_letter
        }
    
    async def _generate_multiple_choice_answer_async(
        self,
        question: str,
        image_base64: str,
        pipeline_info: Optional[Dict[str, Any]] = None,
        async_client: Optional[AsyncGeminiClient] = None,
        model: Optional[str] = None,
        retry_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        异步生成选择题答案
        """
        correct_answer = await self._generate_correct_answer_async(
            question=question,
            image_base64=image_base64,
            pipeline_info=pipeline_info,
            async_client=async_client,
            model=model,
            retry_context=retry_context,
        )
        
        if not correct_answer or not correct_answer.get("answer"):
            err = {
                "answer": None,
                "explanation": (correct_answer.get("explanation") if isinstance(correct_answer, dict) else None) or "无法生成正确答案",
                "full_question": question,
                "options": None,
                "error": (correct_answer.get("error") if isinstance(correct_answer, dict) else None) or "无法生成正确答案",
                "error_type": "generation",
                "error_step": "correct_answer",
            }
            if isinstance(correct_answer, dict) and correct_answer.get("raw_response_truncated"):
                err["raw_response_truncated"] = correct_answer["raw_response_truncated"]
            return err
        
        correct_answer_raw = correct_answer.get("answer", "")
        explanation = correct_answer.get("explanation", "")
        
        # 确保 correct_answer_text 是字符串（处理可能是字典的情况）
        if isinstance(correct_answer_raw, dict):
            # 尝试从字典中提取答案文本
            correct_answer_text = correct_answer_raw.get("Answer", correct_answer_raw.get("answer", ""))
            if not correct_answer_text or not isinstance(correct_answer_text, str):
                # 如果字典中没有找到，尝试获取第一个字符串值
                for key, value in correct_answer_raw.items():
                    if isinstance(value, str) and value.strip():
                        correct_answer_text = value.strip()
                        break
                else:
                    correct_answer_text = str(correct_answer_raw)
        elif isinstance(correct_answer_raw, str):
            correct_answer_text = correct_answer_raw.strip().strip('"\'')
            
            # 单数字（如计数题 "3"）是合法的；仅拒绝空或纯标点
            invalid_tokens = ["{", "}", "[", "]", "(", ")", ",", ".", ":", ";", "!", "?", "null", "None", "undefined"]
            if not correct_answer_text or correct_answer_text in invalid_tokens:
                log_error(f"正确答案无效: '{correct_answer_text}'，原始值: {correct_answer_raw}")
                return {
                    "answer": None,
                    "explanation": explanation,
                    "full_question": question,
                    "options": None,
                    "error": f"正确答案格式不正确: '{correct_answer_text}'",
                    "error_type": "format",
                    "error_step": "correct_answer",
                }
            
            # 如果看起来像不完整的JSON，尝试提取
            if correct_answer_text.startswith("{") and correct_answer_text.endswith("}") and len(correct_answer_text) > 2:
                try:
                    import json
                    answer_dict = json.loads(correct_answer_text)
                    # 尝试从字典中提取
                    extracted = answer_dict.get("Answer") or answer_dict.get("answer") or answer_dict.get("option") or answer_dict.get("text")
                    if extracted and isinstance(extracted, str) and len(extracted.strip()) >= 2:
                        correct_answer_text = extracted.strip()
                        log_warning(f"从JSON字符串中提取正确答案: {correct_answer_text[:50]}...")
                    else:
                        # 尝试使用正则表达式提取
                        import re
                        text_match = re.search(r'["\']?(?:Answer|answer|option|text)["\']?\s*[:=]\s*["\']?([^"\'}]+)["\']?', correct_answer_text, re.IGNORECASE)
                        if text_match:
                            extracted_text = text_match.group(1).strip().strip('"\'')
                            if extracted_text and len(extracted_text) >= 2:
                                correct_answer_text = extracted_text
                                log_warning(f"从格式不完整的JSON中提取正确答案: {extracted_text[:50]}...")
                            else:
                                log_error(f"无法从JSON字符串中提取有效答案: '{correct_answer_text}'")
                                return {
                                    "answer": None,
                                    "explanation": explanation,
                                    "full_question": question,
                                    "options": None,
                                    "error": f"无法从JSON字符串中提取有效答案: '{correct_answer_text}'",
                                    "error_type": "format",
                                    "error_step": "correct_answer",
                                }
                        else:
                            log_error(f"无法从JSON字符串中提取有效答案: '{correct_answer_text}'")
                            return {
                                "answer": None,
                                "explanation": explanation,
                                "full_question": question,
                                "options": None,
                                "error": f"无法从JSON字符串中提取有效答案: '{correct_answer_text}'",
                                "error_type": "format",
                                "error_step": "correct_answer",
                            }
                except (json.JSONDecodeError, ValueError) as e:
                    # JSON解析失败，尝试使用正则表达式提取
                    import re
                    text_match = re.search(r'["\']?(?:Answer|answer|option|text)["\']?\s*[:=]\s*["\']?([^"\'}]+)["\']?', correct_answer_text, re.IGNORECASE)
                    if text_match:
                        extracted_text = text_match.group(1).strip().strip('"\'')
                        if extracted_text and len(extracted_text) >= 2:
                            correct_answer_text = extracted_text
                            log_warning(f"从格式不完整的JSON中提取正确答案: {extracted_text[:50]}...")
                        else:
                            log_error(f"无法从JSON字符串中提取有效答案: '{correct_answer_text}' (错误: {str(e)})")
                            return {
                                "answer": None,
                                "explanation": explanation,
                                "full_question": question,
                                "options": None,
                                "error": f"无法从JSON字符串中提取有效答案: '{correct_answer_text}'",
                                "error_type": "format",
                                "error_step": "correct_answer",
                            }
                    else:
                        log_error(f"JSON解析失败且无法提取文本: '{correct_answer_text}' (错误: {str(e)})")
                        return {
                            "answer": None,
                            "explanation": explanation,
                            "full_question": question,
                            "options": None,
                            "error": f"JSON解析失败: '{correct_answer_text}'",
                            "error_type": "format",
                            "error_step": "correct_answer",
                        }
        else:
            correct_answer_text = str(correct_answer_raw).strip() if correct_answer_raw else ""
        
        # 最终验证（单数字如计数题 "2"/"21" 合法，仅拒绝空或无效）
        if not correct_answer_text or not correct_answer_text.strip():
            log_error(f"正确答案为空: '{correct_answer_text}'，原始值: {correct_answer}")
            return {
                "answer": None,
                "explanation": explanation,
                "full_question": question,
                "options": None,
                "error": f"正确答案格式不正确: '{correct_answer_text}'",
                "error_type": "format",
                "error_step": "correct_answer",
            }
        
        # 再次检查是否是无效的标点符号
        if correct_answer_text.strip() in ["{", "}", "[", "]", "(", ")", ",", ".", ":", ";", "!", "?", "null", "None", "undefined"]:
            log_error(f"正确答案是无效的标点符号: '{correct_answer_text}'，原始值: {correct_answer}")
            return {
                "answer": None,
                "explanation": explanation,
                "full_question": question,
                "options": None,
                "error": f"正确答案格式不正确: '{correct_answer_text}'",
                "error_type": "format",
                "error_step": "correct_answer",
            }
        
        wrong_options_result = await self._generate_wrong_options_async(
            question=question,
            image_base64=image_base64,
            correct_answer=correct_answer_text,
            pipeline_info=pipeline_info,
            async_client=async_client,
            model=model
        )
        wrong_options = (
            wrong_options_result.get("options", [])
            if isinstance(wrong_options_result, dict)
            else wrong_options_result
        )
        if not wrong_options:
            log_warning(f"错误选项生成失败，返回空列表。正确答案: {correct_answer_text}")
            err = {
                "answer": None,
                "explanation": explanation,
                "full_question": question,
                "options": None,
                "error": (wrong_options_result.get("error") if isinstance(wrong_options_result, dict) else None) or "错误选项生成失败",
                "error_type": "generation",
                "error_step": "wrong_options",
            }
            if isinstance(wrong_options_result, dict) and wrong_options_result.get("raw_response_truncated"):
                err["raw_response_truncated"] = wrong_options_result["raw_response_truncated"]
            return err
        
        import random as _random
        
        # 清理所有选项，确保都是纯文本字符串（不是 JSON 字符串）
        def clean_option_text(opt):
            """清理选项文本，确保是纯文本而不是 JSON 字符串"""
            if isinstance(opt, str):
                opt_str = opt.strip().strip('"\'')
                # 如果字符串太短（少于3个字符），不太可能是有效的JSON，直接返回
                if len(opt_str) < 3:
                    return opt_str
                # 检查是否是 JSON 字符串格式（必须是完整的JSON对象）
                if opt_str.startswith("{") and opt_str.endswith("}") and len(opt_str) > 2:
                    try:
                        import json
                        opt_dict = json.loads(opt_str)
                        # 从解析后的字典中提取 Answer 字段
                        cleaned = opt_dict.get("Answer") or opt_dict.get("answer") or opt_dict.get("option") or opt_dict.get("text")
                        if cleaned and isinstance(cleaned, str) and cleaned.strip():
                            return cleaned.strip()
                        # 如果找不到标准字段，尝试获取第一个字符串值
                        for key, value in opt_dict.items():
                            if key.lower() not in ["explanation", "reason"] and isinstance(value, str) and value.strip():
                                return value.strip()
                        # 如果都找不到，返回原始字符串（不应该发生）
                        log_warning(f"无法从 JSON 字符串中提取选项文本: {opt_str[:100]}")
                        return opt_str
                    except (json.JSONDecodeError, ValueError) as e:
                        # 如果解析失败，可能是格式不完整的JSON，尝试提取可能的文本内容
                        # 例如："{answer": "text"}" 或 "{Answer: text}"
                        # 尝试使用正则表达式提取可能的文本值
                        import re
                        # 尝试匹配 "key": "value" 或 "key": value 格式
                        text_match = re.search(r'["\']?(?:Answer|answer|option|text)["\']?\s*[:=]\s*["\']?([^"\'}]+)["\']?', opt_str, re.IGNORECASE)
                        if text_match:
                            extracted_text = text_match.group(1).strip().strip('"\'')
                            if extracted_text and len(extracted_text) > 0:
                                log_warning(f"从格式不完整的JSON中提取选项文本: {extracted_text[:50]}...")
                                return extracted_text
                        # 如果都提取不到，返回原始字符串（但记录警告）
                        log_warning(f"JSON解析失败且无法提取文本，使用原始字符串: {opt_str[:50]}... (错误: {str(e)})")
                        return opt_str
                else:
                    return opt_str
            elif isinstance(opt, dict):
                # 如果是字典，提取 Answer 字段
                cleaned = opt.get("Answer") or opt.get("answer") or opt.get("option") or opt.get("text")
                if cleaned and isinstance(cleaned, str) and cleaned.strip():
                    return cleaned.strip()
                # 如果找不到标准字段，尝试获取第一个非空字符串值
                for key, value in opt.items():
                    if key.lower() not in ["explanation", "reason"] and isinstance(value, str) and value.strip():
                        return value.strip()
                return str(opt)
            else:
                return str(opt).strip() if opt else ""
        
        # 清理所有选项
        cleaned_wrong_options = [clean_option_text(opt) for opt in wrong_options]
        cleaned_correct_answer = clean_option_text(correct_answer_text)
        
        # 验证清理后的选项是否有效（单数字/字母如 "2"/"A" 合法，仅拒绝空或无效标点）
        def validate_option(opt_text, opt_name):
            """验证选项文本是否有效"""
            if not opt_text or len(opt_text.strip()) == 0:
                log_warning(f"{opt_name} 选项为空")
                return False
            if opt_text.strip() in ["{", "}", "[", "]", "(", ")", ",", ".", ":", ";", "!", "?", "null", "None", "undefined"]:
                log_warning(f"{opt_name} 选项是无效标点/占位符: '{opt_text}'，这可能是解析错误")
                return False
            return True
        
        # 检查并记录无效选项
        if not validate_option(cleaned_correct_answer, "正确答案"):
            log_error(f"正确答案无效: '{cleaned_correct_answer}'，原始值: {correct_answer_text}")
            return {
                "answer": None,
                "explanation": explanation,
                "full_question": question,
                "options": None,
                "error": f"正确答案格式无效: '{cleaned_correct_answer}'",
                "error_type": "format",
                "error_step": "correct_answer",
            }
        
        invalid_wrong_options = []
        for i, opt in enumerate(cleaned_wrong_options):
            if not validate_option(opt, f"错误选项{i+1}"):
                invalid_wrong_options.append((i, opt, wrong_options[i] if i < len(wrong_options) else None))
        
        if invalid_wrong_options:
            log_error(f"发现 {len(invalid_wrong_options)} 个无效的错误选项:")
            for idx, invalid_opt, original_opt in invalid_wrong_options:
                log_error(f"  错误选项{idx+1}: '{invalid_opt}' (原始值: {original_opt})")
            # 如果所有错误选项都无效，返回错误
            if len(invalid_wrong_options) == len(cleaned_wrong_options):
                return {
                    "answer": None,
                    "explanation": explanation,
                    "full_question": question,
                    "options": None,
                    "error": "所有错误选项格式无效",
                    "error_type": "format",
                    "error_step": "wrong_options",
                }
        
        all_options = [cleaned_correct_answer] + cleaned_wrong_options
        _random.shuffle(all_options)
        
        correct_index = all_options.index(cleaned_correct_answer)
        correct_letter = chr(65 + correct_index)
        
        options_text = "\n".join([f"{chr(65 + i)}: {option}" for i, option in enumerate(all_options)])
        full_question = f"{question}\n{options_text}"
        
        # 构建最终的 options 字典，确保所有值都是纯文本字符串
        final_options = {chr(65 + i): option for i, option in enumerate(all_options)}
        
        return {
            "answer": correct_letter,
            "explanation": explanation,
            "full_question": full_question,
            "options": final_options,
            "correct_option": correct_letter
        }
    
    def _generate_fill_in_blank_answer(
        self,
        question: str,
        image_base64: str,
        pipeline_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成填空题答案
        """
        result = self._generate_correct_answer(
            question=question,
            image_base64=image_base64,
            pipeline_info=pipeline_info
        )
        
        if not result or not result.get("answer"):
            return {
                "answer": None,
                "explanation": "无法生成答案",
                "full_question": question,
                "error": "无法生成答案",
                "error_type": "generation",
                "error_step": "correct_answer",
            }
        
        return {
            "answer": result["answer"],
            "explanation": result.get("explanation", ""),
            "full_question": question
        }
    
    async def _generate_fill_in_blank_answer_async(
        self,
        question: str,
        image_base64: str,
        pipeline_info: Optional[Dict[str, Any]] = None,
        async_client: Optional[AsyncGeminiClient] = None,
        model: Optional[str] = None,
        retry_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        异步生成填空题答案
        """
        result = await self._generate_correct_answer_async(
            question=question,
            image_base64=image_base64,
            pipeline_info=pipeline_info,
            async_client=async_client,
            model=model,
            retry_context=retry_context,
        )
        
        if not result or not result.get("answer"):
            return {
                "answer": None,
                "explanation": "无法生成答案",
                "full_question": question,
                "error": "无法生成答案",
                "error_type": "generation",
                "error_step": "correct_answer",
            }
        
        return {
            "answer": result["answer"],
            "explanation": result.get("explanation", ""),
            "full_question": question
        }

    def _generate_correct_answer(
        self,
        question: str,
        image_base64: str,
        pipeline_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, str]]:
        """
        生成正确答案
        
        Returns:
            {"answer": "答案文本", "explanation": "解释"} 或 None
        """
        # 提取 claim 和 prefilled_values（与问题生成阶段一致）
        prefill_claim = None
        prefilled_values = {}
        if isinstance(pipeline_info, dict):
            prefill_claim = pipeline_info.get("prefill_claim")
            prefilled_values = pipeline_info.get("prefilled_values", {})
            # 如果没有直接提供，尝试从 prefill_object 提取
            if not prefill_claim or not prefilled_values:
                prefill_object = pipeline_info.get("prefill_object")
                if isinstance(prefill_object, dict):
                    if not prefill_claim:
                        prefill_claim = prefill_object.get("claim")
                    if not prefilled_values:
                        prefilled_values = prefill_object.get("prefilled_values", {})
        
        # 构建 hint（使用 claim + prefilled_values，不再强制使用 target_object）
        target_hint = ""
        if prefill_claim or prefilled_values:
            hint_lines = ["Additional context (do not change the question):"]
            if prefill_claim:
                hint_lines.append(f"- Claim template: {prefill_claim}")
            if prefilled_values:
                prefilled_str = ", ".join([f"{k}={v}" for k, v in prefilled_values.items()])
                hint_lines.append(f"- Prefilled values: {prefilled_str}")
            target_hint = "\n" + "\n".join(hint_lines) + "\n"

        prompt = f"""Based on the image and the question, provide a concise and accurate answer.

Question: {question}
{target_hint}

Requirements:
1. Provide ONLY the answer text, keep it concise and direct (typically 1-5 words)
2. Do NOT include any analysis, reasoning, or explanation in the answer field
3. The answer should be factual and based solely on the image content
4. If the question asks for a specific format (e.g., number, location, object name), provide the answer in that format
5. Keep the answer brief - avoid unnecessary words

Provide your response in the following format:
Answer: [your concise answer here, 1-5 words typically]
Explanation: [optional brief explanation, only if needed for clarity]

Important: The Answer field should contain ONLY the answer itself, nothing else. Keep it very brief."""

        try:
            response = self.gemini_client.analyze_image(
                image_input=image_base64,
                prompt=prompt,
                temperature=self.gen_settings.get("temperature", 0.7),
                context="generate_correct_answer",
                max_tokens=self.gen_settings.get("max_tokens", 512)
            )
            log_model_response(
                stage="answer_generation",
                prompt=prompt,
                response=response,
                context={"question": question, "sub_stage": "correct_answer"},
            )
            # 解析响应
            answer, explanation = self._parse_answer_response(response)
            
            return {
                "answer": answer,
                "explanation": explanation
            }
            
        except Exception as e:
            log_error(f"生成正确答案失败: {e}")
            return None
    
    async def _generate_correct_answer_async(
        self,
        question: str,
        image_base64: str,
        pipeline_info: Optional[Dict[str, Any]] = None,
        async_client: Optional[AsyncGeminiClient] = None,
        model: Optional[str] = None,
        retry_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        """
        异步生成正确答案（使用OpenAI兼容接口，与vlmtool/generate_vqa完全对齐）
        retry_context: 重试时传入，包含 previous_answer, regeneration_reason，用于注入失败反馈
        """
        # 提取 claim 和 prefilled_values（与问题生成阶段一致）
        prefill_claim = None
        prefilled_values = {}
        if isinstance(pipeline_info, dict):
            prefill_claim = pipeline_info.get("prefill_claim")
            prefilled_values = pipeline_info.get("prefilled_values", {})
            # 如果没有直接提供，尝试从 prefill_object 提取
            if not prefill_claim or not prefilled_values:
                prefill_object = pipeline_info.get("prefill_object")
                if isinstance(prefill_object, dict):
                    if not prefill_claim:
                        prefill_claim = prefill_object.get("claim")
                    if not prefilled_values:
                        prefilled_values = prefill_object.get("prefilled_values", {})
        
        # 构建 hint（使用 claim + prefilled_values，不再强制使用 target_object）
        target_hint = ""
        if prefill_claim or prefilled_values:
            hint_lines = ["Additional context (do not change the question):"]
            if prefill_claim:
                hint_lines.append(f"- Claim template: {prefill_claim}")
            if prefilled_values:
                prefilled_str = ", ".join([f"{k}={v}" for k, v in prefilled_values.items()])
                hint_lines.append(f"- Prefilled values: {prefilled_str}")
            target_hint = "\n" + "\n".join(hint_lines) + "\n"

        retry_hint = ""
        temperature = 0.3
        if retry_context:
            prev_ans = retry_context.get("previous_answer")
            reason = retry_context.get("regeneration_reason", "验证失败")
            retry_cnt = retry_context.get("retry_count", 0)
            retry_hint = "\n[RETRY] A previous answer failed validation. "
            if prev_ans is not None and str(prev_ans).strip():
                retry_hint += f"Previous answer was: \"{str(prev_ans)[:100]}\". "
            retry_hint += f"Failure reason: {reason[:200]}. "
            retry_hint += "Please provide a different or corrected answer based on the image. Re-examine the image carefully.\n"
            temperature = 0.5 + min(0.2 * retry_cnt, 0.2)  # 重试时提高 temperature (0.5~0.7) 增加多样性

        prompt = f"""Based on the image and the question, provide a concise and accurate answer.
{retry_hint}
Question: {question}
{target_hint}

Requirements:
1. Provide ONLY the answer text, keep it concise and direct (typically 1-5 words)
2. Do NOT include any analysis, reasoning, or explanation in the answer field
3. The answer should be factual and based solely on the image content
4. If the question asks for a specific format (e.g., number, location, object name), provide the answer in that format
5. Keep the answer brief - avoid unnecessary words
6. The answer must be a plain text string, NOT a JSON object or JSON string

Provide your response in the following format:
Answer: [your concise answer here, 1-5 words typically]
Explanation: [optional brief explanation, only if needed for clarity]

Important: The Answer field should contain ONLY the answer itself, nothing else. Keep it very brief."""
        try:
            # 构建图像内容（OpenAI兼容格式）
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
            
            text_content = {
                "type": "text",
                "text": prompt
            }
            
            # 使用OpenAI兼容接口（与vlmtool/generate_vqa完全对齐）
            # 确定使用的模型名称
            if model is None:
                if async_client is not None:
                    model = async_client.model_name
                else:
                    from QA_Generator.config import config
                    model = config.MODEL_NAME
            
            # 使用传入的模型名称（与vlmtool一致：model=self.model）；重试时 temperature 更高以增加多样性
            if async_client is None:
                async with AsyncGeminiClient() as client:
                    response = await client.chat.completions.create(
                        model=model,  # 使用传入的model参数，与vlmtool一致
                        messages=[
                            {
                                "role": "user",
                                "content": [text_content, image_content]
                            }
                        ],
                        max_completion_tokens=1000,
                        temperature=temperature
                    )
            else:
                response = await async_client.chat.completions.create(
                    model=model,  # 使用传入的model参数，与vlmtool一致
                    messages=[
                        {
                            "role": "user",
                            "content": [text_content, image_content]
                        }
                    ],
                    max_completion_tokens=1000,
                    temperature=temperature
                )
            
            # 提取响应内容
            response_text = response.choices[0].message.content
            log_model_response(
                stage="answer_generation",
                prompt=prompt,
                response=response_text,
                context={"question": question, "sub_stage": "correct_answer"},
            )
            # 解析响应（先文本格式，必要时兼容JSON）
            answer, explanation = self._parse_answer_response(response_text)

            response_str = response_text.strip()
            if response_str.startswith("{") and response_str.endswith("}"):
                try:
                    import json
                    result = json.loads(response_str)
                    answer_raw = result.get("answer") or result.get("Answer")
                    explanation_raw = result.get("explanation") or result.get("Explanation")
                    if answer_raw:
                        answer = str(answer_raw).strip().strip('"\'')
                    if explanation_raw:
                        explanation = str(explanation_raw).strip()
                except Exception:
                    pass

            # 最终校验：答案必须有效（单数字如 "3" 对计数题是合法的，不应因长度被拒绝）
            ans_stripped = answer.strip()
            invalid_tokens = ["{", "}", "[", "]", "(", ")", ",", ".", ":", ";", "!", "?", "null", "None", "undefined"]
            if not ans_stripped or ans_stripped in invalid_tokens:
                return {
                    "answer": None,
                    "explanation": explanation or "",
                    "error": "答案无效或格式不符合要求",
                    "error_type": "generation",
                    "error_step": "correct_answer",
                    "raw_response_truncated": (response_text or "")[:_RAW_RESPONSE_MAX] if response_text else None,
                }
                
            return {"answer": answer, "explanation": explanation}
        except Exception as e:
            error_msg = str(e)
            if "400" in error_msg or "Bad Request" in error_msg:
                error_detail = f"异步生成正确答案失败 (400 Bad Request): {error_msg}\n"
                error_detail += f"  问题: {question[:100]}...\n"
                error_detail += f"  图片base64长度: {len(image_base64) if image_base64 else 0}\n"
                error_detail += "  可能的原因:\n"
                error_detail += "    1. 图片 base64 编码格式不正确\n"
                error_detail += "    2. 请求体大小超过 API 限制\n"
                error_detail += "    3. API 参数格式不匹配"
                log_error(error_detail)
            else:
                log_error(f"异步生成正确答案失败: {error_msg}")
            return {
                "answer": None,
                "explanation": "",
                "error": f"API异常: {error_msg[:200]}",
                "error_type": "generation",
                "error_step": "correct_answer",
            }
    
    def _generate_wrong_options(
        self,
        question: str,
        image_base64: str,
        correct_answer: str,
        pipeline_info: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        生成错误选项（迷惑选项）
        
        Args:
            question: 问题文本
            image_base64: 图片base64
            correct_answer: 正确答案
            pipeline_info: pipeline信息
            
        Returns:
            错误选项列表
        """
        # 随机选择错误选项数量
        min_count = self.mc_config.get("wrong_options", {}).get("min_count", 2)
        max_count = self.mc_config.get("wrong_options", {}).get("max_count", 4)
        wrong_count = random.randint(min_count, max_count)
        
        prompt = f"""Based on the image and question, generate {wrong_count} plausible but incorrect answer options.

Question: {question}
Correct Answer: {correct_answer}

Requirements:
1. Generate exactly {wrong_count} wrong options.
2. Each option MUST have the same answer TYPE and style as the correct answer ("{correct_answer}") — e.g. if the correct answer is a short direction word like "left", all options must also be short direction words, NOT location phrases like "on the road".
3. Options must be plausible but clearly incorrect when compared to the image.
4. Options must be diverse and not repetitive (no duplicate or near-duplicate texts).
5. Each option should be concise (similar length to the correct answer, typically 1-5 words).
6. DO NOT include any analysis or explanation, only the options.
7. Make sure each option is a single, concise phrase similar to the correct answer format.
8. DO NOT output options that are synonyms, paraphrases, or slight variants of the correct answer (e.g. if the correct answer is "left", do NOT output "on the left", "left side", "to the left", etc.).
9. For numeric or percentage answers (e.g., "5%", "10", "3"): wrong options must be sufficiently different from the CORRECT answer — avoid options that are too close to the correct value (e.g., if correct is "5%", do NOT use "4%", "6%", "7%"; use "25%", "50%", "80%" instead). Aim for at least 10–15 percentage points or a meaningful numeric gap between each wrong option and the correct answer.

Provide your response in the following format (one option per line):
Option 1: [option text]
Option 2: [option text]
...
Option {wrong_count}: [option text]

Important: Each option should be brief and similar in style to "{correct_answer}", but clearly WRONG and not ambiguous with it."""

        try:
            response = self.gemini_client.analyze_image(
                image_input=image_base64,
                prompt=prompt,
                temperature=self.gen_settings.get("temperature", 0.9),  # 更高温度以增加多样性
                context="generate_wrong_options",
                max_tokens=self.gen_settings.get("max_tokens", 512)
            )
            log_model_response(
                stage="answer_generation",
                prompt=prompt,
                response=response,
                context={"question": question, "sub_stage": "wrong_options", "correct_answer": correct_answer},
            )
            # 解析错误选项
            wrong_options = self._parse_wrong_options_response(response, wrong_count)
            # 进行去重和与正确答案的模糊冲突过滤
            wrong_options = self._postprocess_wrong_options(wrong_options, correct_answer)
            return wrong_options
            
        except Exception as e:
            log_error(f"生成错误选项失败: {e}")
            return []
    
    def _parse_answer_response(self, response: str) -> Tuple[str, str]:
        """
        解析答案响应（增强版：支持多种格式）
        
        Returns:
            (answer, explanation)
        """
        answer = ""
        explanation = ""
        
        if not response or not isinstance(response, str):
            return "", ""
        
        response_clean = response.strip()
        if not response_clean:
            return "", ""
        
        # 策略1: 尝试解析JSON格式（优先）
        result = extract_json_from_response(response_clean)
        if result is not None:
            answer = (
                result.get("answer") or result.get("Answer") or
                result.get("text") or result.get("Text") or
                result.get("option") or result.get("Option") or
                ""
            )
            explanation = (
                result.get("explanation") or result.get("Explanation") or
                result.get("reason") or result.get("Reason") or
                ""
            )
            if answer:
                answer = str(answer).strip().strip('"\'')
            if explanation:
                explanation = str(explanation).strip()
            if answer:
                return answer, explanation
        
        # 策略2: 提取 "Answer: ..." 格式（标准文本格式）
        answer_match = re.search(r'Answer\s*[:：]\s*(.+?)(?:\n|Explanation|$)', response_clean, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            # 提取Explanation（如果存在）
            explanation_match = re.search(r'Explanation\s*[:：]\s*(.+?)(?:\n|$)', response_clean, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
        
        # 策略3: 如果没有找到Answer格式，尝试按行提取
        if not answer:
            lines = [line.strip() for line in response_clean.split('\n') if line.strip()]
            for i, line in enumerate(lines):
                # 跳过明显的标题行
                if re.match(r'^(Answer|Explanation|Question|问题|答案|解释)\s*[:：]?\s*$', line, re.IGNORECASE):
                    continue
                # 尝试提取答案（第一行非空内容，或包含关键词的行）
                if not answer:
                    # 移除可能的"Answer:"前缀
                    cleaned = re.sub(r'^(Answer|答案)\s*[:：]\s*', '', line, flags=re.IGNORECASE).strip()
                    if cleaned and len(cleaned) > 0:
                        answer = cleaned
                # 提取解释（在答案之后的行）
                elif not explanation and i > 0:
                    cleaned = re.sub(r'^(Explanation|解释)\s*[:：]\s*', '', line, flags=re.IGNORECASE).strip()
                    if cleaned:
                        explanation = cleaned
                        break
        
        # 策略4: 如果仍然没有答案，使用整个响应的第一行（去除引号）
        if not answer:
            first_line = response_clean.split('\n')[0].strip()
            # 移除引号和常见前缀
            answer = re.sub(r'^["\']|["\']$', '', first_line)
            answer = re.sub(r'^(Answer|答案|A)\s*[:：.]\s*', '', answer, flags=re.IGNORECASE).strip()
        
        # 最终清理
        answer = answer.strip('"\'`').strip()
        explanation = explanation.strip('"\'`').strip()
        
        # 如果答案看起来像JSON片段，尝试提取
        if answer.startswith('{') and '}' in answer:
            answer_dict = extract_json_from_response(answer)
            if answer_dict:
                answer = answer_dict.get("answer") or answer_dict.get("Answer") or str(answer_dict)
        
        return answer, explanation
    
    def _parse_wrong_options_response(self, response: str, expected_count: int) -> List[str]:
        """
        解析错误选项响应（增强版：支持多种格式）
        
        Returns:
            错误选项列表
        """
        options = []
        
        if not response or not isinstance(response, str):
            return []
        
        response_clean = response.strip()
        if not response_clean:
            return []
        
        # 策略1: 尝试解析JSON格式（优先）
        result = extract_json_from_response(response_clean)
        if result is not None:
            if isinstance(result, dict):
                options_list = (
                    result.get("options") or result.get("wrong_options") or
                    result.get("alternatives") or result.get("wrong_answers") or
                    []
                )
                if isinstance(options_list, list) and options_list:
                    options = [str(opt).strip().strip('"\'') for opt in options_list if opt]
            elif isinstance(result, list):
                options = [str(opt).strip().strip('"\'') for opt in result if opt]
            if options:
                return options[:expected_count]
        
        # 策略2: 标准格式 "Option 1: ... Option 2: ..."
        pattern = r'Option\s+\d+\s*[:：.]\s*(.+?)(?=\n\s*Option\s+\d+\s*[:：.]|$)'
        matches = re.findall(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
        if matches:
            options = [match.strip().strip('"\'') for match in matches]
        
        # 策略3: 字母格式 "A: ... B: ..." 或 "A. ... B. ..."
        if not options:
            pattern = r'[A-Z]\s*[:：.]\s*(.+?)(?=\n\s*[A-Z]\s*[:：.]|$)'
            matches = re.findall(pattern, response_clean, re.MULTILINE)
            if matches:
                options = [match.strip().strip('"\'') for match in matches]
        
        # 策略4: 数字编号格式 "1. ... 2. ..."
        if not options:
            pattern = r'\d+\.\s*(.+?)(?=\n\s*\d+\.|$)'
            matches = re.findall(pattern, response_clean, re.MULTILINE)
            if matches:
                options = [match.strip().strip('"\'') for match in matches]
        
        # 策略5: 按行分割（过滤标题行）
        if not options:
            lines = [line.strip() for line in response_clean.split('\n') if line.strip()]
            # 过滤掉明显的标题行和空行
            filtered_lines = []
            for line in lines:
                # 跳过标题行
                if re.match(r'^(Option|选项|错误选项|Wrong|Requirements|要求)\s*[:：]?\s*$', line, re.IGNORECASE):
                    continue
                # 跳过纯数字或字母行（可能是编号）
                if re.match(r'^[A-Z\d]+\.?\s*$', line):
                    continue
                # 提取内容（移除可能的编号前缀）
                cleaned = re.sub(r'^(Option\s+\d+|选项\s*\d+|[A-Z]\s*|[\d]+\.)\s*[:：.]\s*', '', line, flags=re.IGNORECASE).strip()
                if cleaned and len(cleaned) > 1:  # 至少2个字符
                    filtered_lines.append(cleaned)
            options = filtered_lines[:expected_count]
        
        # 清理选项（移除引号、去除空白）
        options = [opt.strip('"\'`').strip() for opt in options if opt and opt.strip()]
        
        # 如果选项数量不足，返回已有的（后续再做去重/过滤）
        return options[:expected_count] if options else []

    def _postprocess_wrong_options(self, wrong_options: List[str], correct_answer: str) -> List[str]:
        """
        对错误选项做后处理：
        - 去重（大小写不敏感）
        - 过滤与正确答案过于相似/模糊的选项（完全相同、包含关系）
        
        注意：这是启发式过滤，宁可丢掉一些选项，也要避免“几乎等于正确答案”的干扰项。
        """
        if not wrong_options:
            return wrong_options
        
        ca = (correct_answer or "").strip().lower()
        result: List[str] = []
        seen: set[str] = set()
        
        for opt in wrong_options:
            if not isinstance(opt, str):
                opt_text = str(opt)
            else:
                opt_text = opt
            text = opt_text.strip()
            if not text:
                continue
            key = text.lower()
            # 去重
            if key in seen:
                continue
            # 过滤与正确答案过于接近的情况：
            # - 完全相同
            # - 其中一个是另一个的子串（如 "left" vs "on the left", "left side")
            if ca:
                if key == ca or ca in key or key in ca:
                    continue
            seen.add(key)
            result.append(text)
        
        return result
    
    async def _generate_wrong_options_async(
        self,
        question: str,
        image_base64: str,
        correct_answer: str,
        pipeline_info: Optional[Dict[str, Any]] = None,
        async_client: Optional[AsyncGeminiClient] = None,
        model: Optional[str] = None
    ) -> List[str]:
        """
        异步生成错误选项（迷惑选项，使用OpenAI兼容接口，与vlmtool/generate_vqa对齐）
        """
        min_count = self.mc_config.get("wrong_options", {}).get("min_count", 2)
        max_count = self.mc_config.get("wrong_options", {}).get("max_count", 4)
        wrong_count = random.randint(min_count, max_count)
        
        prompt = f"""Based on the image and question, generate {wrong_count} plausible but incorrect answer options.

Question: {question}
Correct Answer: {correct_answer}

Requirements:
1. Generate exactly {wrong_count} wrong options.
2. Each option MUST have the same answer TYPE and style as the correct answer ("{correct_answer}") — e.g. if the correct answer is a short direction word like "left", all options must also be short direction words, NOT location phrases like "on the road".
3. Options must be plausible but clearly incorrect when compared to the image.
4. Options must be diverse and not repetitive (no duplicate or near-duplicate texts).
5. Each option should be concise (similar length to the correct answer, typically 1-5 words).
6. DO NOT include any analysis or explanation, only the options.
7. Make sure each option is a single, concise phrase similar to the correct answer format.
8. Each option must be a plain text string, NOT a JSON object or JSON string.
9. DO NOT output options that are synonyms, paraphrases, or slight variants of the correct answer (e.g. if the correct answer is "left", do NOT output "on the left", "left side", "to the left", etc.).
10. For numeric or percentage answers (e.g., "5%", "10", "3"): wrong options must be sufficiently different from the CORRECT answer — avoid options that are too close to the correct value (e.g., if correct is "5%", do NOT use "4%", "6%", "7%"; use "25%", "50%", "80%" instead). Aim for at least 10–15 percentage points or a meaningful numeric gap between each wrong option and the correct answer.

Provide your response in the following format (one option per line):
Option 1: [option text]
Option 2: [option text]
...
Option {wrong_count}: [option text]"""
        try:
            # 构建图像内容（OpenAI兼容格式）
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
            
            text_content = {
                "type": "text",
                "text": prompt
            }
            
            # 使用OpenAI兼容接口（与vlmtool/generate_vqa对齐）
            # 确定使用的模型名称
            if model is None:
                if async_client is not None:
                    model = async_client.model_name
                else:
                    from QA_Generator.config import config
                    model = config.MODEL_NAME
            
            # 使用传入的模型名称（与vlmtool一致：model=self.model）
            if async_client is None:
                async with AsyncGeminiClient() as client:
                    response = await client.chat.completions.create(
                        model=model,  # 使用传入的model参数，与vlmtool一致
                        messages=[
                            {
                                "role": "user",
                                "content": [text_content, image_content]
                            }
                        ],
                        max_completion_tokens=1000,
                        temperature=0.3
                    )
            else:
                response = await async_client.chat.completions.create(
                    model=model,  # 使用传入的model参数，与vlmtool一致
                    messages=[
                        {
                            "role": "user",
                            "content": [text_content, image_content]
                        }
                    ],
                    max_completion_tokens=1000,
                    temperature=0.3
                )
            
            # 提取响应内容
            response_text = response.choices[0].message.content
            log_model_response(
                stage="answer_generation",
                prompt=prompt,
                response=response_text,
                context={"question": question, "sub_stage": "wrong_options", "correct_answer": correct_answer},
            )
            # 记录原始响应（用于调试）
            log_debug(f"错误选项生成的原始响应: {response_text[:500]}...")
            
            wrong_options = self._parse_wrong_options_response(response_text, wrong_count)
            if not wrong_options:
                log_error(f"错误选项解析失败，返回空列表。原始响应: {response_text[:300]}...")
                return {
                    "options": [],
                    "error": "错误选项解析失败",
                    "raw_response_truncated": (response_text or "")[:_RAW_RESPONSE_MAX] if response_text else None,
                }
            # 进行去重和与正确答案的模糊冲突过滤
            wrong_options = self._postprocess_wrong_options(wrong_options, correct_answer)
            return wrong_options
        except Exception as e:
            log_error(f"异步生成错误选项失败: {e}")
            return {"options": [], "error": f"API异常: {str(e)[:200]}"}

