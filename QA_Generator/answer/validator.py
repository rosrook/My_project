"""
答案校验模块
包含格式检查与修复、VQA验证等功能
"""
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from QA_Generator.clients.gemini_client import GeminiClient
from QA_Generator.logging.logger import log_warning, log_debug, log_debug_dict


class AnswerValidator:
    """答案校验器"""
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        初始化校验器
        
        Args:
            gemini_client: Gemini客户端实例
        """
        self.gemini_client = gemini_client or GeminiClient()
    
    def validate_and_fix(
        self,
        result: Dict[str, Any],
        image_base64: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        完整的校验和修复流程
        
        Args:
            result: 答案生成结果
            image_base64: 图片base64编码
            
        Returns:
            (修复后的结果, 校验报告)
        """
        log_debug("========== Validator.validate_and_fix 开始 ==========")
        log_debug_dict("Validator 接收到的 result", {
            "question": result.get("question"),
            "question_type": result.get("question_type"),
            "answer": result.get("answer"),
            "full_question": result.get("full_question"),
            "options": result.get("options"),
            "correct_option": result.get("correct_option"),
            "explanation": result.get("explanation", "")[:100] if result.get("explanation") else ""
        })
        
        # 特别详细记录 options
        if result.get("options"):
            log_debug("Validator 接收到的 options 字典:")
            for key, value in result.get("options", {}).items():
                log_debug(f"  {key}: {value} (type: {type(value).__name__})")
        else:
            log_warning("Validator 接收到的 options 为空或 None!")
        
        validation_report = {
            "format_check": {},
            "vqa_validation": {},
            "fixed_issues": [],
            "validation_passed": True,
            "should_regenerate": False,  # 是否应该重新生成
            "regeneration_reason": ""  # 重新生成的原因
        }
        
        # Step 1: 格式检查与修复
        log_debug("========== Step 1: 格式检查与修复 ==========")
        fixed_result, format_report = self._format_check_and_fix(result)
        validation_report["format_check"] = format_report
        
        log_debug_dict("格式检查后的 fixed_result", {
            "question": fixed_result.get("question"),
            "question_type": fixed_result.get("question_type"),
            "answer": fixed_result.get("answer"),
            "full_question": fixed_result.get("full_question"),
            "options": fixed_result.get("options"),
            "correct_option": fixed_result.get("correct_option")
        })
        
        # 确保所有原始字段都被保留（特别是options字段）
        # 如果fixed_result中缺少某些字段，从原始result中补充
        for key in result.keys():
            if key not in fixed_result:
                fixed_result[key] = result[key]
                log_debug(f"从原始 result 补充字段: {key} = {result[key]}")
        
        if not format_report.get("passed", False):
            validation_report["validation_passed"] = False
            # 如果格式问题无法自动修复，建议重新生成
            if format_report.get("issues") and len(format_report.get("fixes_applied", [])) == 0:
                validation_report["should_regenerate"] = True
                validation_report["regeneration_reason"] = "格式检查失败且无法自动修复: " + "; ".join(format_report.get("issues", [])[:2])
            return fixed_result, validation_report
        
        # Step 2: VQA验证
        log_debug("========== Step 2: VQA验证 ==========")
        log_debug_dict("传递给 _vqa_validation 的 fixed_result", {
            "question": fixed_result.get("question"),
            "question_type": fixed_result.get("question_type"),
            "answer": fixed_result.get("answer"),
            "full_question": fixed_result.get("full_question"),
            "options": fixed_result.get("options"),
            "correct_option": fixed_result.get("correct_option")
        })
        
        vqa_report = self._vqa_validation(
            result=fixed_result,
            image_base64=image_base64
        )
        validation_report["vqa_validation"] = vqa_report
        
        log_debug("========== Validator.validate_and_fix 结束 ==========\n")
        
        if not vqa_report.get("passed", False):
            validation_report["validation_passed"] = False
            # VQA验证失败时，建议重新生成
            validation_report["should_regenerate"] = True
            
            # 收集失败原因（不再包含困惑度分析）
            reasons = []
            # 困惑度分析已禁用，不再作为验证要求
            # if not vqa_report.get("perplexity_analysis", {}).get("passed", True):
            #     reasons.append("困惑度分析未通过")
            if not vqa_report.get("confidence_assessment", {}).get("passed", True):
                reasons.append("置信度评估未通过")
            if not vqa_report.get("answer_validation", {}).get("passed", True):
                reasons.append("答案验证未通过")
            
            validation_report["regeneration_reason"] = "; ".join(reasons) if reasons else "VQA验证失败"
        
        return fixed_result, validation_report
    
    def _format_check_and_fix(
        self,
        result: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Step 1: 格式检查与修复
        
        检查项：
        - 占位符检查
        - 选项重复检查
        - 答案完整性检查
        - 自动修复问题选项
        - 验证修复结果
        """
        report = {
            "passed": True,
            "issues": [],
            "fixes_applied": []
        }
        
        fixed_result = result.copy()
        question_type = result.get("question_type", "")
        
        # 首先修复答案格式：如果 answer 是 JSON 字符串格式，解析并提取纯文本
        answer_raw = fixed_result.get("answer", "")
        if isinstance(answer_raw, str) and answer_raw.strip().startswith("{") and answer_raw.strip().endswith("}"):
            try:
                import json
                answer_dict = json.loads(answer_raw.strip().strip('"\''))
                # 提取 Answer 字段
                cleaned_answer = answer_dict.get("Answer") or answer_dict.get("answer") or answer_dict.get("option") or answer_dict.get("text")
                if cleaned_answer and isinstance(cleaned_answer, str):
                    fixed_result["answer"] = cleaned_answer.strip()
                    # 如果 explanation 为空，尝试从 JSON 中提取
                    if not fixed_result.get("explanation"):
                        explanation = answer_dict.get("Explanation") or answer_dict.get("explanation")
                        if explanation and isinstance(explanation, str):
                            fixed_result["explanation"] = explanation.strip()
                    log_debug(f"修复了 JSON 字符串格式的答案: {answer_raw[:50]}... -> {fixed_result['answer']}")
            except (json.JSONDecodeError, ValueError):
                # 如果解析失败，保持原样
                pass
        
        # 检查占位符（使用修复后的结果）
        placeholder_issues = self._check_placeholders(fixed_result)
        if placeholder_issues:
            report["issues"].extend(placeholder_issues)
            report["passed"] = False
        
        # 对于选择题，进行选项相关检查
        if question_type == "multiple_choice":
            # 检查选项重复
            duplicate_issues = self._check_option_duplicates(result)
            if duplicate_issues:
                report["issues"].extend(duplicate_issues)
                report["passed"] = False
                # 尝试修复
                fixed_result = self._fix_option_duplicates(fixed_result)
                if fixed_result != result:
                    report["fixes_applied"].append("修复了选项重复问题")
            
            # 检查答案完整性
            answer_completeness = self._check_answer_completeness(result)
            if not answer_completeness["passed"]:
                report["issues"].extend(answer_completeness["issues"])
                report["passed"] = False
                # 尝试修复
                fixed_result = self._fix_answer_completeness(fixed_result)
                if fixed_result != result:
                    report["fixes_applied"].append("修复了答案完整性问题")
            
            # 验证修复结果
            if report["fixes_applied"]:
                verification = self._verify_fixes(fixed_result)
                if not verification["passed"]:
                    report["issues"].extend(verification["issues"])
                    report["passed"] = False
                else:
                    report["passed"] = True  # 修复成功，重新标记为通过
        
        # 对于填空题，检查答案是否存在
        elif question_type == "fill_in_blank":
            if not result.get("answer") or result.get("answer") == "":
                report["issues"].append("填空题答案为空")
                report["passed"] = False
        
        return fixed_result, report
    
    def _check_placeholders(self, result: Dict[str, Any]) -> List[str]:
        """
        检查占位符
        
        检查问题、答案、选项中是否包含未填充的占位符
        """
        issues = []
        placeholders = [
            r'\[.*?\]',  # [object], [number] 等
            r'\{.*?\}',  # {placeholder} 等
            r'<.*?>',    # <placeholder> 等
            r'___+',     # 下划线占位符
            r'\.\.\.+',  # 省略号占位符
        ]
        
        # 检查问题
        question = result.get("question", "")
        full_question = result.get("full_question", "")
        
        for placeholder_pattern in placeholders:
            if re.search(placeholder_pattern, question):
                issues.append(f"问题中包含未填充的占位符: {re.findall(placeholder_pattern, question)[0]}")
            if re.search(placeholder_pattern, full_question):
                issues.append(f"完整问题中包含未填充的占位符: {re.findall(placeholder_pattern, full_question)[0]}")
        
        # 检查答案
        answer = result.get("answer", "")
        if isinstance(answer, str):
            for placeholder_pattern in placeholders:
                if re.search(placeholder_pattern, answer):
                    issues.append(f"答案中包含未填充的占位符: {re.findall(placeholder_pattern, answer)[0]}")
        
        # 检查选项（选择题）
        if result.get("question_type") == "multiple_choice":
            options = result.get("options", {})
            for option_key, option_value in options.items():
                if isinstance(option_value, str):
                    for placeholder_pattern in placeholders:
                        if re.search(placeholder_pattern, option_value):
                            issues.append(f"选项{option_key}中包含未填充的占位符: {re.findall(placeholder_pattern, option_value)[0]}")
        
        return issues
    
    def _check_option_duplicates(self, result: Dict[str, Any]) -> List[str]:
        """
        检查选项重复
        
        检查选择题的选项是否有重复
        """
        issues = []
        options = result.get("options", {})
        
        if not options:
            issues.append("选项字典为空")
            return issues
        
        # 检查选项值是否重复
        option_values = list(options.values())
        seen = set()
        duplicates = []
        
        for i, value in enumerate(option_values):
            # 标准化比较（去除空格、转小写）
            normalized = value.strip().lower()
            if normalized in seen:
                duplicates.append(value)
            seen.add(normalized)
        
        if duplicates:
            issues.append(f"发现重复选项: {', '.join(set(duplicates))}")
        
        # 检查选项数量是否合理（至少2个选项）
        if len(options) < 2:
            issues.append(f"选项数量不足: 只有{len(options)}个选项")
        
        return issues
    
    def _check_answer_completeness(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查答案完整性
        
        检查：
        - answer字段是否存在且有效
        - correct_option是否与answer一致
        - 答案对应的选项是否存在
        """
        issues = []
        passed = True
        
        answer = result.get("answer")
        correct_option = result.get("correct_option")
        options = result.get("options", {})
        
        # 检查answer字段
        if not answer or answer == "":
            issues.append("answer字段为空")
            passed = False
        
        # 检查answer是否为有效的选项字母
        if answer and isinstance(answer, str):
            if answer.upper() not in options:
                issues.append(f"answer字段'{answer}'对应的选项不存在")
                passed = False
        
        # 检查correct_option与answer是否一致
        if answer and correct_option:
            if answer.upper() != correct_option.upper():
                issues.append(f"answer字段'{answer}'与correct_option字段'{correct_option}'不一致")
                passed = False
        
        # 检查correct_option对应的选项是否存在
        if correct_option and correct_option.upper() not in options:
            issues.append(f"correct_option字段'{correct_option}'对应的选项不存在")
            passed = False
        
        return {
            "passed": passed,
            "issues": issues
        }
    
    def _fix_option_duplicates(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        修复选项重复问题
        
        如果发现重复选项，尝试生成新的选项替换
        """
        fixed_result = result.copy()
        options = result.get("options", {}).copy()
        
        # 找出重复的选项
        option_values = list(options.values())
        seen = {}
        duplicates = []
        
        for key, value in options.items():
            normalized = value.strip().lower()
            if normalized in seen:
                duplicates.append(key)
            else:
                seen[normalized] = key
        
        # 如果有重复，标记需要修复
        # 注意：这里只是标记，实际修复可能需要重新生成选项
        if duplicates:
            # 可以在这里添加逻辑，尝试从其他来源生成新选项
            # 目前先返回原结果，标记需要人工处理
            pass
        
        return fixed_result
    
    def _fix_answer_completeness(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        修复答案完整性问题
        
        尝试修复answer和correct_option的不一致
        """
        fixed_result = result.copy()
        answer = result.get("answer")
        correct_option = result.get("correct_option")
        options = result.get("options", {})
        
        # 如果answer无效但correct_option有效，使用correct_option
        if (not answer or answer.upper() not in options) and correct_option:
            if correct_option.upper() in options:
                fixed_result["answer"] = correct_option.upper()
        
        # 如果correct_option无效但answer有效，使用answer
        elif answer and answer.upper() in options:
            if not correct_option or correct_option.upper() != answer.upper():
                fixed_result["correct_option"] = answer.upper()
        
        return fixed_result
    
    def _verify_fixes(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证修复结果
        
        重新检查修复后的结果是否通过所有格式检查
        """
        issues = []
        
        # 重新检查选项重复
        duplicate_issues = self._check_option_duplicates(result)
        if duplicate_issues:
            issues.extend(duplicate_issues)
        
        # 重新检查答案完整性
        completeness = self._check_answer_completeness(result)
        if not completeness["passed"]:
            issues.extend(completeness["issues"])
        
        return {
            "passed": len(issues) == 0,
            "issues": issues
        }
    
    def _vqa_validation(
        self,
        result: Dict[str, Any],
        image_base64: str
    ) -> Dict[str, Any]:
        """
        Step 2: VQA验证
        
        包含：
        - 困惑度分析
        - 置信度评估
        - 答案验证
        """
        report = {
            "passed": True,
            "perplexity_analysis": {},
            "confidence_assessment": {},
            "answer_validation": {}
        }
        
        question_type = result.get("question_type", "")
        question = result.get("question", "")
        answer = result.get("answer", "")
        full_question = result.get("full_question", question)
        
        # 困惑度分析（已禁用，不再作为验证要求）
        # perplexity = self._analyze_perplexity(
        #     question=full_question,
        #     answer=answer,
        #     image_base64=image_base64,
        #     question_type=question_type
        # )
        # 仍然记录，但不作为验证要求
        report["perplexity_analysis"] = {
            "passed": True,  # 始终通过，不作为验证要求
            "disabled": True,
            "note": "困惑度分析已禁用，不再作为验证要求"
        }
        
        # 置信度评估
        confidence = self._assess_confidence(
            question=full_question,
            answer=answer,
            image_base64=image_base64,
            question_type=question_type,
            options=result.get("options")
        )
        report["confidence_assessment"] = confidence
        
        # 答案验证
        answer_validation = self._validate_answer(
            question=full_question,
            answer=answer,
            image_base64=image_base64,
            question_type=question_type,
            options=result.get("options")
        )
        report["answer_validation"] = answer_validation
        
        # 综合判断（不再检查困惑度）
        if not confidence.get("passed", True) or \
           not answer_validation.get("passed", True):
            report["passed"] = False
        
        return report
    
    def _analyze_perplexity(
        self,
        question: str,
        answer: str,
        image_base64: str,
        question_type: str
    ) -> Dict[str, Any]:
        """
        困惑度分析
        
        分析问题是否清晰、是否容易产生歧义
        """
        prompt = f"""Analyze the clarity and potential ambiguity of this VQA question-answer pair.

Question: {question}
Answer: {answer}
Question Type: {question_type}

Evaluate:
1. Is the question clear and unambiguous?
2. Could the question be interpreted in multiple ways?
3. Is the answer clearly correct based on the question?

Return a JSON object:
{{
    "clarity_score": 0.0-1.0,
    "ambiguity_level": "low/medium/high",
    "is_clear": true/false,
    "issues": ["list of any clarity issues"]
}}"""

        try:
            response = self.gemini_client.analyze_image(
                image_input=image_base64,
                prompt=prompt,
                temperature=0.3,
                context="perplexity_analysis"
            )
            
            # 解析JSON响应
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                clarity_score = result.get("clarity_score", 0.5)
                is_clear = result.get("is_clear", False)
                issues = result.get("issues", [])
                
                return {
                    "passed": is_clear and clarity_score >= 0.7,
                    "clarity_score": clarity_score,
                    "ambiguity_level": result.get("ambiguity_level", "medium"),
                    "issues": issues
                }
            
            return {
                "passed": False,
                "clarity_score": 0.5,
                "ambiguity_level": "unknown",
                "issues": ["无法解析困惑度分析结果"]
            }
            
        except Exception as e:
            log_warning(f"困惑度分析失败: {e}")
            return {
                "passed": False,
                "clarity_score": 0.5,
                "ambiguity_level": "unknown",
                "issues": [f"分析过程出错: {str(e)}"]
            }
    
    def _assess_confidence(
        self,
        question: str,
        answer: str,
        image_base64: str,
        question_type: str,
        options: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        置信度评估
        
        评估答案的置信度，判断答案是否正确
        
        注意：对于选择题，question 参数应该是 full_question（包含选项的完整问题）
        """
        if question_type == "multiple_choice" and options:
            # 对于选择题，优先使用 options 字典构建完整问题，确保信息完整
            if options and len(options) > 0:
                # 从 question 中提取原始问题（去掉可能存在的选项部分）
                base_question = question
                if "\nA:" in question:
                    base_question = question.split("\nA:")[0].strip()
                elif "\nA. " in question:
                    base_question = question.split("\nA. ")[0].strip()
                
                # 使用 options 字典构建完整的选项文本
                options_list = []
                for key in sorted(options.keys()):  # 按字母顺序排序（A, B, C, D...）
                    option_value = options[key]
                    # 确保选项值是字符串
                    if isinstance(option_value, dict):
                        option_value = option_value.get("option", option_value.get("text", str(option_value)))
                    elif not isinstance(option_value, str):
                        option_value = str(option_value)
                    options_list.append(f"{key}: {option_value}")
                
                options_text = "\n".join(options_list)
                full_question_text = f"{base_question}\n{options_text}"
            else:
                # 如果没有 options 字典，使用 question（可能是 full_question）
                full_question_text = question
            
            prompt = f"""Assess the confidence and correctness of this multiple-choice VQA answer.

Full Question (including options):
{full_question_text}
Given Answer: {answer}

Evaluate:
1. Is the given answer correct based on the image?
2. What is the confidence level (0.0-1.0)?
3. Are there other plausible options that could be correct?

Return a JSON object:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "correctness_reason": "explanation",
    "alternative_options": ["list of other plausible options if any"]
}}"""
        else:
            prompt = f"""Assess the confidence and correctness of this VQA answer.

Question: {question}
Answer: {answer}

Evaluate:
1. Is the answer correct based on the image?
2. What is the confidence level (0.0-1.0)?
3. Is the answer complete and appropriate?

Return a JSON object:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "correctness_reason": "explanation"
}}"""

        try:
            response = self.gemini_client.analyze_image(
                image_input=image_base64,
                prompt=prompt,
                temperature=0.3,
                context="confidence_assessment"
            )
            
            # 解析JSON响应
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                is_correct = result.get("is_correct", False)
                confidence = result.get("confidence", 0.5)
                
                return {
                    "passed": is_correct and confidence >= 0.7,
                    "is_correct": is_correct,
                    "confidence": confidence,
                    "correctness_reason": result.get("correctness_reason", ""),
                    "alternative_options": result.get("alternative_options", [])
                }
            
            return {
                "passed": False,
                "is_correct": False,
                "confidence": 0.5,
                "correctness_reason": "无法解析置信度评估结果"
            }
            
        except Exception as e:
            log_warning(f"置信度评估失败: {e}")
            return {
                "passed": False,
                "is_correct": False,
                "confidence": 0.5,
                "correctness_reason": f"评估过程出错: {str(e)}"
            }
    
    def _validate_answer(
        self,
        question: str,
        answer: str,
        image_base64: str,
        question_type: str,
        options: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        答案验证
        
        验证答案是否与图片内容一致
        
        注意：对于选择题，question 参数应该是 full_question（包含选项的完整问题）
        """
        log_debug("========== _validate_answer 开始 ==========")
        log_debug_dict("_validate_answer 接收到的参数", {
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "options": options,
            "image_base64_length": len(image_base64) if image_base64 else 0
        })
        
        if question_type == "multiple_choice" and options:
            log_debug("处理选择题验证")
            log_debug(f"原始 question: {question}")
            log_debug(f"options 字典: {options}")
            # 对于选择题，优先使用 options 字典构建完整问题，确保信息完整
            # 即使 question 看起来已经包含选项，也使用 options 字典重新构建，避免格式问题
            if options and len(options) > 0:
                # 从 question 中提取原始问题（去掉可能存在的选项部分）
                base_question = question
                # 如果 question 包含选项标记，尝试提取原始问题
                if "\nA:" in question:
                    base_question = question.split("\nA:")[0].strip()
                    log_debug(f"从 question 中提取 base_question (包含 \\nA:): {base_question}")
                elif "\nA. " in question:
                    base_question = question.split("\nA. ")[0].strip()
                    log_debug(f"从 question 中提取 base_question (包含 \\nA. ): {base_question}")
                else:
                    log_debug(f"question 不包含选项标记，直接使用: {base_question}")
                
                # 使用 options 字典构建完整的选项文本
                options_list = []
                log_debug("构建选项列表:")
                for key in sorted(options.keys()):  # 按字母顺序排序（A, B, C, D...）
                    option_value = options[key]
                    log_debug(f"  处理选项 {key}: {option_value} (type: {type(option_value).__name__})")
                    
                    # 确保选项值是字符串
                    if isinstance(option_value, dict):
                        # 如果是字典，尝试提取文本
                        option_value = option_value.get("option", option_value.get("text", str(option_value)))
                        log_debug(f"    从字典中提取: {option_value}")
                    elif not isinstance(option_value, str):
                        option_value = str(option_value)
                        log_debug(f"    转换为字符串: {option_value}")
                    
                    options_list.append(f"{key}: {option_value}")
                    log_debug(f"    最终选项文本: {key}: {option_value}")
                
                options_text = "\n".join(options_list)
                full_question_text = f"{base_question}\n{options_text}"
                log_debug(f"构建的完整问题 (full_question_text):")
                log_debug(full_question_text)
            else:
                # 如果没有 options 字典，使用 question（可能是 full_question）
                full_question_text = question
                log_warning("options 字典为空，直接使用 question 作为 full_question_text")
                log_debug(f"使用的 full_question_text: {full_question_text}")
            
            # 获取答案对应的文本
            if options and answer.upper() in options:
                answer_text = options[answer.upper()]
                log_debug(f"从 options 中获取答案文本: {answer} -> {answer_text} (type: {type(answer_text).__name__})")
                
                # 确保 answer_text 是字符串
                if isinstance(answer_text, dict):
                    answer_text = answer_text.get("option", answer_text.get("text", str(answer_text)))
                    log_debug(f"从字典中提取答案文本: {answer_text}")
                elif not isinstance(answer_text, str):
                    answer_text = str(answer_text)
                    log_debug(f"转换答案文本为字符串: {answer_text}")
            else:
                answer_text = answer
                log_warning(f"答案 {answer} 不在 options 中，直接使用答案作为 answer_text")
            
            prompt = f"""Validate whether the given answer is correct for this multiple-choice VQA question.

Full Question (including options):
{full_question_text}
Given Answer: {answer} (which corresponds to: {answer_text})

Verify:
1. Does the answer match what is shown in the image?
2. Is the answer logically consistent with the question?
3. Are there any contradictions?

Return a JSON object:
{{
    "is_valid": true/false,
    "validation_reason": "detailed explanation",
    "issues": ["list of any validation issues"]
}}"""
            
            log_debug("构建的验证 prompt (前1000字符):")
            log_debug(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        else:
            prompt = f"""Validate whether the given answer is correct for this VQA question.

Question: {question}
Answer: {answer}

Verify:
1. Does the answer match what is shown in the image?
2. Is the answer logically consistent with the question?
3. Is the answer complete and appropriate?

Return a JSON object:
{{
    "is_valid": true/false,
    "validation_reason": "detailed explanation",
    "issues": ["list of any validation issues"]
}}"""

        try:
            response = self.gemini_client.analyze_image(
                image_input=image_base64,
                prompt=prompt,
                temperature=0.3,
                context="answer_validation"
            )
            
            # 解析JSON响应
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                is_valid = result.get("is_valid", False)
                issues = result.get("issues", [])
                
                return {
                    "passed": is_valid and len(issues) == 0,
                    "is_valid": is_valid,
                    "validation_reason": result.get("validation_reason", ""),
                    "issues": issues
                }
            
            return {
                "passed": False,
                "is_valid": False,
                "validation_reason": "无法解析答案验证结果",
                "issues": ["解析失败"]
            }
            
        except Exception as e:
            log_warning(f"答案验证失败: {e}")
            return {
                "passed": False,
                "is_valid": False,
                "validation_reason": f"验证过程出错: {str(e)}",
                "issues": [f"验证错误: {str(e)}"]
            }

