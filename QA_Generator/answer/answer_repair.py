"""
答案自动修复模块

在答案生成 + 初次校验失败时，尝试用一次额外的 LLM 调用对答案进行“抢救式修复”，
而不是立刻丢弃整条样本。
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

from QA_Generator.clients.gemini_client import GeminiClient
from QA_Generator.logging.logger import log_warning, log_debug, log_debug_dict


class AnswerRepairer:
    """
    答案修复器

    典型使用场景：
    - 多选题：当前选项字母可能错误，让模型在已有选项中重新挑一个最合理的选项字母
    - 填空题：当前答案与图片不符或置信度较低，让模型在问题和图片的基础上给出一个更可靠的短答案
    """

    def __init__(self, gemini_client: Optional[GeminiClient] = None) -> None:
        self.gemini_client = gemini_client or GeminiClient()

    def repair_once(
        self,
        result: Dict[str, Any],
        image_base64: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        尝试调用模型修复一次答案。

        Args:
            result: 当前这条记录的答案结果（通常是 answer_generator 或 validator 返回的 dict）
            image_base64: 对应图片的 base64 编码

        Returns:
            (repaired_result, repair_report)
            - repaired_result: 修复后的结果（如果修复失败或不需要修复，则等于输入 result）
            - repair_report: {
                  "repair_attempted": bool,
                  "repair_successful": bool,
                  "reason": str,
              }
        """
        repair_report: Dict[str, Any] = {
            "repair_attempted": True,
            "repair_successful": False,
            "reason": "",
        }

        question_type = result.get("question_type", "")
        question = result.get("question") or result.get("full_question") or ""
        answer = result.get("answer")
        options = result.get("options") or {}

        if not question or answer is None:
            repair_report["reason"] = "缺少question或answer，无法修复"
            return result, repair_report

        try:
            if question_type == "multiple_choice" and isinstance(options, dict) and options:
                repaired, reason = self._repair_multiple_choice(result, question, image_base64)
            else:
                repaired, reason = self._repair_fill_in_blank(result, question, image_base64)
        except Exception as e:
            log_warning(f"答案修复过程中发生异常: {e}")
            repair_report["reason"] = f"修复过程异常: {e}"
            return result, repair_report

        if repaired is None:
            repair_report["reason"] = reason or "模型未给出有效修复方案"
            return result, repair_report

        repair_report["repair_successful"] = True
        repair_report["reason"] = reason or "修复成功"
        return repaired, repair_report

    # ------------------------------------------------------------------
    # 多选题修复
    # ------------------------------------------------------------------
    def _repair_multiple_choice(
        self,
        result: Dict[str, Any],
        question: str,
        image_base64: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        对多选题进行修复：在现有选项中重新挑选一个最合理的选项字母。
        """
        options = result.get("options") or {}
        current_answer = str(result.get("answer") or "").strip().upper()

        if not options or not current_answer:
            return None, "多选题缺少options或当前answer，无法修复"

        # 构建完整题面（问题 + 选项）
        base_question = question
        if "\nA:" in question:
            base_question = question.split("\nA:")[0].strip()
        elif "\nA. " in question:
            base_question = question.split("\nA. ")[0].strip()

        option_lines = []
        for key in sorted(options.keys()):
            value = options[key]
            if isinstance(value, dict):
                value = value.get("option", value.get("text", str(value)))
            elif not isinstance(value, str):
                value = str(value)
            option_lines.append(f"{key}: {value}")
        options_text = "\n".join(option_lines)
        full_question = f"{base_question}\n{options_text}"

        prompt = f"""You are a strict VQA multiple-choice grader.

Image-based question with options:
{full_question}

The currently selected answer is: {current_answer}

Your tasks:
1. Carefully check the image and the question.
2. Decide whether the CURRENT answer is clearly correct.
3. If it is NOT clearly correct, choose the SINGLE BEST correct option letter from the existing options.
4. Never propose options outside the given list.

Return ONLY a JSON object in this format:
{{
  "action": "KEEP" | "CHANGE",
  "new_answer": "A" | "B" | "C" | "D" | null,
  "reason": "brief explanation"
}}"""

        response = self.gemini_client.analyze_image(
            image_input=image_base64,
            prompt=prompt,
            temperature=0.2,
            context="answer_repair_mc",
        )

        log_debug("AnswerRepairer 多选题修复原始响应:")
        log_debug(response[:800] + "..." if len(response) > 800 else response)

        # 解析 JSON
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                return None, "未找到JSON格式修复结果"
            import json

            data = json.loads(json_match.group())
        except Exception as e:
            return None, f"解析修复JSON失败: {e}"

        action = str(data.get("action") or "").upper()
        new_answer = str(data.get("new_answer") or "").strip().upper()
        reason = str(data.get("reason") or "").strip()

        if action == "KEEP":
            return None, reason or "模型认为当前答案已足够合理"

        if action == "CHANGE":
            if new_answer and new_answer in options and new_answer != current_answer:
                repaired = result.copy()
                repaired["answer"] = new_answer
                # 同步更新 correct_option（如果存在）
                if repaired.get("correct_option"):
                    repaired["correct_option"] = new_answer
                return repaired, reason or f"更换答案为 {new_answer}"
            else:
                return None, "修复结果 new_answer 无效或与当前答案相同"

        return None, "未知的修复action"

    # ------------------------------------------------------------------
    # 填空题修复
    # ------------------------------------------------------------------
    def _repair_fill_in_blank(
        self,
        result: Dict[str, Any],
        question: str,
        image_base64: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        对填空题进行修复：在同一问题下，尝试给出一个更可靠的短答案。
        """
        current_answer_raw = result.get("answer")
        current_answer = str(current_answer_raw or "").strip()

        if not current_answer:
            return None, "当前填空题答案为空，无法修复"

        prompt = f"""You are a strict VQA short-answer grader and fixer.

Question:
{question}

Current Answer:
{current_answer}

Tasks:
1. Check whether the CURRENT answer is clearly correct and well-formed based on the image.
2. If it is clearly correct, keep it.
3. If it is incorrect, incomplete, or poorly formatted, propose a BETTER concise answer (1-5 words).

Return ONLY a JSON object in this format:
{{
  "action": "KEEP" | "CHANGE",
  "new_answer": "short corrected answer or null",
  "reason": "brief explanation"
}}"""

        response = self.gemini_client.analyze_image(
            image_input=image_base64,
            prompt=prompt,
            temperature=0.2,
            context="answer_repair_fib",
        )

        log_debug("AnswerRepairer 填空题修复原始响应:")
        log_debug(response[:800] + "..." if len(response) > 800 else response)

        # 解析 JSON
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                return None, "未找到JSON格式修复结果"
            import json

            data = json.loads(json_match.group())
        except Exception as e:
            return None, f"解析修复JSON失败: {e}"

        action = str(data.get("action") or "").upper()
        new_answer_raw = data.get("new_answer")
        new_answer = str(new_answer_raw or "").strip() if new_answer_raw is not None else ""
        reason = str(data.get("reason") or "").strip()

        if action == "KEEP":
            return None, reason or "模型认为当前答案已足够合理"

        if action == "CHANGE":
            if new_answer and new_answer != current_answer:
                repaired = result.copy()
                repaired["answer"] = new_answer
                return repaired, reason or f"修复填空题答案为: {new_answer}"
            else:
                return None, "修复结果 new_answer 无效或未变化"

        return None, "未知的修复action"

