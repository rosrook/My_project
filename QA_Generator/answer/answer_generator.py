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
        model: Optional[str] = None
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
        """
        if question_type == "multiple_choice":
            return await self._generate_multiple_choice_answer_async(
                question=question,
                image_base64=image_base64,
                pipeline_info=pipeline_info,
                async_client=async_client,
                model=model
            )
        elif question_type == "fill_in_blank":
            return await self._generate_fill_in_blank_answer_async(
                question=question,
                image_base64=image_base64,
                pipeline_info=pipeline_info,
                async_client=async_client,
                model=model
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
                "options": None
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
                "options": None
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
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        异步生成选择题答案
        """
        correct_answer = await self._generate_correct_answer_async(
            question=question,
            image_base64=image_base64,
            pipeline_info=pipeline_info,
            async_client=async_client,
            model=model
        )
        
        if not correct_answer or not correct_answer.get("answer"):
            return {
                "answer": None,
                "explanation": "无法生成正确答案",
                "full_question": question,
                "options": None
            }
        
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
            
            # 如果字符串太短或只是单个标点符号，可能是解析错误
            if len(correct_answer_text) < 2:
                log_error(f"正确答案太短或无效: '{correct_answer_text}'，原始值: {correct_answer_raw}")
                return {
                    "answer": None,
                    "explanation": explanation,
                    "full_question": question,
                    "options": None,
                    "error": f"正确答案格式不正确: '{correct_answer_text}'"
                }
            
            # 检查是否是单个标点符号
            if correct_answer_text in ["{", "}", "[", "]", "(", ")", ",", ".", ":", ";", "!", "?", "null", "None", "undefined"]:
                log_error(f"正确答案是无效的标点符号: '{correct_answer_text}'，原始值: {correct_answer_raw}")
                return {
                    "answer": None,
                    "explanation": explanation,
                    "full_question": question,
                    "options": None,
                    "error": f"正确答案格式不正确: '{correct_answer_text}'"
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
                                    "error": f"无法从JSON字符串中提取有效答案: '{correct_answer_text}'"
                                }
                        else:
                            log_error(f"无法从JSON字符串中提取有效答案: '{correct_answer_text}'")
                            return {
                                "answer": None,
                                "explanation": explanation,
                                "full_question": question,
                                "options": None,
                                "error": f"无法从JSON字符串中提取有效答案: '{correct_answer_text}'"
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
                                "error": f"无法从JSON字符串中提取有效答案: '{correct_answer_text}'"
                            }
                    else:
                        log_error(f"JSON解析失败且无法提取文本: '{correct_answer_text}' (错误: {str(e)})")
                        return {
                            "answer": None,
                            "explanation": explanation,
                            "full_question": question,
                            "options": None,
                            "error": f"JSON解析失败: '{correct_answer_text}'"
                        }
        else:
            correct_answer_text = str(correct_answer_raw).strip() if correct_answer_raw else ""
        
        # 最终验证
        if not correct_answer_text or len(correct_answer_text.strip()) < 2:
            log_error(f"正确答案为空或格式不正确: '{correct_answer_text}'，原始值: {correct_answer}")
            return {
                "answer": None,
                "explanation": explanation,
                "full_question": question,
                "options": None,
                "error": f"正确答案格式不正确: '{correct_answer_text}'"
            }
        
        # 再次检查是否是无效的标点符号
        if correct_answer_text.strip() in ["{", "}", "[", "]", "(", ")", ",", ".", ":", ";", "!", "?", "null", "None", "undefined"]:
            log_error(f"正确答案是无效的标点符号: '{correct_answer_text}'，原始值: {correct_answer}")
            return {
                "answer": None,
                "explanation": explanation,
                "full_question": question,
                "options": None,
                "error": f"正确答案格式不正确: '{correct_answer_text}'"
            }
        
        wrong_options = await self._generate_wrong_options_async(
            question=question,
            image_base64=image_base64,
            correct_answer=correct_answer_text,
            pipeline_info=pipeline_info,
            async_client=async_client,
            model=model
        )
        
        if not wrong_options:
            log_warning(f"错误选项生成失败，返回空列表。正确答案: {correct_answer_text}")
            return {
                "answer": None,
                "explanation": explanation,
                "full_question": question,
                "options": None,
                "error": "错误选项生成失败"
            }
        
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
        
        # 验证清理后的选项是否有效（不能是单个字符如 "{" 或空字符串）
        def validate_option(opt_text, opt_name):
            """验证选项文本是否有效"""
            if not opt_text or len(opt_text.strip()) == 0:
                log_warning(f"{opt_name} 选项为空")
                return False
            if opt_text.strip() in ["{", "}", "[", "]", "(", ")", ",", ".", ":", ";", "!", "?"]:
                log_warning(f"{opt_name} 选项是单个标点符号: '{opt_text}'，这可能是解析错误")
                return False
            if len(opt_text.strip()) < 2:
                log_warning(f"{opt_name} 选项太短: '{opt_text}'，可能不是有效的选项文本")
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
                "error": f"正确答案格式无效: '{cleaned_correct_answer}'"
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
                    "error": "所有错误选项格式无效"
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
                "full_question": question
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
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        异步生成填空题答案
        """
        result = await self._generate_correct_answer_async(
            question=question,
            image_base64=image_base64,
            pipeline_info=pipeline_info,
            async_client=async_client,
            model=model
        )
        
        if not result or not result.get("answer"):
            return {
                "answer": None,
                "explanation": "无法生成答案",
                "full_question": question
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
        target_object = None
        prefill_claim = None
        if isinstance(pipeline_info, dict):
            target_object = pipeline_info.get("target_object")
            prefill_claim = pipeline_info.get("prefill_claim")
            if not target_object:
                prefill_object = pipeline_info.get("prefill_object")
                if isinstance(prefill_object, dict):
                    target_object = prefill_object.get("name")
        target_hint = ""
        if target_object or prefill_claim:
            hint_lines = ["Additional context (do not change the question):"]
            if prefill_claim:
                hint_lines.append(f"- Claim: {prefill_claim}")
            if target_object:
                hint_lines.append(f"- Target object (use as the answer): {target_object}")
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
        model: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """
        异步生成正确答案（使用OpenAI兼容接口，与vlmtool/generate_vqa完全对齐）
        """
        target_object = None
        prefill_claim = None
        if isinstance(pipeline_info, dict):
            target_object = pipeline_info.get("target_object")
            prefill_claim = pipeline_info.get("prefill_claim")
            if not target_object:
                prefill_object = pipeline_info.get("prefill_object")
                if isinstance(prefill_object, dict):
                    target_object = prefill_object.get("name")
        target_hint = ""
        if target_object or prefill_claim:
            hint_lines = ["Additional context (do not change the question):"]
            if prefill_claim:
                hint_lines.append(f"- Claim: {prefill_claim}")
            if target_object:
                hint_lines.append(f"- Target object (use as the answer): {target_object}")
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

            # 最终校验：答案必须有效
            if not answer or len(answer.strip()) < 2 or answer.strip() in ["{", "}", "[", "]", "(", ")", ",", ".", ":", ";", "!", "?", "null", "None", "undefined"]:
                            return None
                
            return {"answer": answer, "explanation": explanation}
        except Exception as e:
            error_msg = str(e)
            # 如果是 400 错误，添加更详细的诊断信息
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
            return None
    
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
1. Generate exactly {wrong_count} wrong options
2. Each option should be similar in format and length to the correct answer ("{correct_answer}")
3. Options should be plausible but clearly incorrect when compared to the image
4. Options should be diverse and not repetitive
5. Each option should be concise (similar length to the correct answer, typically 1-5 words)
6. Do NOT include any analysis or explanation, only the options
7. Make sure each option is a single, concise phrase similar to the correct answer format

Provide your response in the following format (one option per line):
Option 1: [option text]
Option 2: [option text]
...
Option {wrong_count}: [option text]

Important: Each option should be brief and similar in style to "{correct_answer}"."""

        try:
            response = self.gemini_client.analyze_image(
                image_input=image_base64,
                prompt=prompt,
                temperature=self.gen_settings.get("temperature", 0.9),  # 更高温度以增加多样性
                context="generate_wrong_options",
                max_tokens=self.gen_settings.get("max_tokens", 512)
            )
            
            # 解析错误选项
            wrong_options = self._parse_wrong_options_response(response, wrong_count)
            
            return wrong_options
            
        except Exception as e:
            log_error(f"生成错误选项失败: {e}")
            return []
    
    def _parse_answer_response(self, response: str) -> Tuple[str, str]:
        """
        解析答案响应
        
        Returns:
            (answer, explanation)
        """
        answer = ""
        explanation = ""
        
        # 提取Answer行
        answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.MULTILINE)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        # 提取Explanation行
        explanation_match = re.search(r'Explanation:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        
        # 如果没有找到Answer格式，尝试提取第一行或整个响应
        if not answer:
            lines = response.strip().split('\n')
            if lines:
                # 移除可能的"Answer:"前缀
                answer = re.sub(r'^Answer:\s*', '', lines[0], flags=re.IGNORECASE).strip()
                if not answer:
                    answer = lines[0].strip()
        
        # 清理答案（移除引号等）
        answer = answer.strip('"\'')
        
        return answer, explanation
    
    def _parse_wrong_options_response(self, response: str, expected_count: int) -> List[str]:
        """
        解析错误选项响应
        
        Returns:
            错误选项列表
        """
        options = []
        
        # 尝试按格式解析：Option 1: ... Option 2: ...
        pattern = r'Option\s+\d+:\s*(.+?)(?:\n|Option|$)'
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        
        if matches:
            options = [match.strip().strip('"\'') for match in matches]
        else:
            # 如果没有找到格式化的选项，尝试按行分割
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            # 过滤掉明显的标题行
            lines = [line for line in lines if not re.match(r'^(Option|选项|错误选项)', line, re.IGNORECASE)]
            options = lines[:expected_count]
        
        # 清理选项
        options = [opt.strip('"\'') for opt in options if opt.strip()]
        
        # 如果选项数量不足，返回已有的
        return options[:expected_count] if options else []
    
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
1. Generate exactly {wrong_count} wrong options
2. Each option should be similar in format and length to the correct answer ("{correct_answer}")
3. Options should be plausible but clearly incorrect when compared to the image
4. Options should be diverse and not repetitive
5. Each option should be concise (similar length to the correct answer, typically 1-5 words)
6. Do NOT include any analysis or explanation, only the options
7. Make sure each option is a single, concise phrase similar to the correct answer format
8. Each option must be a plain text string, NOT a JSON object or JSON string

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
            
            # 记录原始响应（用于调试）
            log_debug(f"错误选项生成的原始响应: {response_text[:500]}...")
            
            wrong_options = self._parse_wrong_options_response(response_text, wrong_count)
            if not wrong_options:
                log_error(f"错误选项解析失败，返回空列表。原始响应: {response_text[:300]}...")
            return wrong_options
        except Exception as e:
            log_error(f"异步生成错误选项失败: {e}")
            return []

