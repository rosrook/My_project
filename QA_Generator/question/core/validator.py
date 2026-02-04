"""
问题验证模块
验证生成的问题是否符合约束条件
"""
import json
import re
from typing import Dict, Any, Optional, Tuple
from QA_Generator.clients.gemini_client import GeminiClient
from QA_Generator.clients.async_client import AsyncGeminiClient
from QA_Generator.utils.model_response_logger import log_model_response


class QuestionValidator:
    """问题验证器"""
    
    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        enable_validation_exemptions: bool = False,
    ):
        """
        初始化验证器
        
        Args:
            gemini_client: Gemini客户端实例
        """
        self.gemini_client = gemini_client or GeminiClient()
        self.enable_validation_exemptions = enable_validation_exemptions
        self.exempt_pipelines = {"question", "visual_recognition", "caption", "text_association"}
    
    def validate(
        self,
        question: str,
        image_input: Any,
        pipeline_config: Dict[str, Any],
        global_constraints: Dict[str, Any]
    ) -> tuple[bool, str]:
        """
        验证问题是否符合约束
        
        Args:
            question: 生成的问题文本
            image_input: 图片输入
            pipeline_config: Pipeline配置
            global_constraints: 全局约束
            
        Returns:
            (是否通过验证, 验证原因)
        """
        # 基本检查
        if not question or len(question.strip()) == 0:
            return False, "问题为空"
        
        # 检查全局约束
        if not self._check_global_constraints(question, global_constraints):
            return False, "违反全局约束"
        
        # 检查Pipeline特定约束
        if not self._check_pipeline_constraints(question, pipeline_config):
            return False, "违反Pipeline约束"

        # 对于“概念/对象识别类”的通用问题，跳过LLM深度验证
        if self.enable_validation_exemptions:
            pipeline_name = pipeline_config.get("intent") or pipeline_config.get("name")
            if pipeline_name in self.exempt_pipelines:
                return True, "skip_validation_for_exempt_pipeline"
        
        # 使用LLM进行深度验证
        is_valid, reason = self._validate_with_llm(
            question=question,
            image_input=image_input,
            pipeline_config=pipeline_config,
            global_constraints=global_constraints
        )
        
        return is_valid, reason
    
    def _check_global_constraints(
        self,
        question: str,
        global_constraints: Dict[str, Any]
    ) -> bool:
        """检查全局约束"""
        # 检查禁止的问题类型
        forbidden_types = global_constraints.get("forbidden_question_types", [])
        question_lower = question.lower()
        
        forbidden_keywords = {
            "generic_scene_description": ["describe", "what do you see", "what's in"],
            "opinion_based": ["do you like", "do you think", "prefer", "favorite"],
            "hypothetical": ["what if", "suppose", "imagine", "if"],
            "commonsense_only": ["why", "how come", "what causes"],
            "unanswerable_from_image": ["what happened before", "what will happen"]
        }
        
        for forbidden_type in forbidden_types:
            keywords = forbidden_keywords.get(forbidden_type, [])
            if any(keyword in question_lower for keyword in keywords):
                return False
        
        return True
    
    def _check_pipeline_constraints(
        self,
        question: str,
        pipeline_config: Dict[str, Any]
    ) -> bool:
        """检查Pipeline特定约束"""
        constraints = pipeline_config.get("question_constraints", [])
        
        # 这里可以添加更详细的约束检查
        # 目前简化处理，主要依赖LLM验证
        
        return True
    
    def _validate_with_llm(
        self,
        question: str,
        image_input: Any,
        pipeline_config: Dict[str, Any],
        global_constraints: Dict[str, Any]
    ) -> tuple[bool, str]:
        """使用LLM进行深度验证"""
        validation_rules = global_constraints.get("validation_rules", [])
        question_constraints = pipeline_config.get("question_constraints", [])
        
        prompt = f"""You are a VQA question validation expert. Validate whether the given question meets all requirements.

Question: "{question}"

Pipeline Intent: {pipeline_config.get("intent", "")}

Global Validation Rules:
{chr(10).join(f"- {rule}" for rule in validation_rules)}

Pipeline-Specific Constraints:
{chr(10).join(f"- {constraint}" for constraint in question_constraints)}

Check if the question:
1. Explicitly references at least one visual entity or region in the image
2. Is answerable solely based on the image
3. Would have a different answer if the image content changes
4. Does NOT rely on external knowledge or commonsense only
5. Follows the pipeline intent and constraints
6. Does NOT directly reveal the answer, and the answer is NOT directly inferable from the question text alone (without the image)

Return your response in plain text using this format:
valid: true/false
reason: <short explanation>"""

        try:
            response = self.gemini_client.analyze_image(
                image_input=image_input,
                prompt=prompt,
                temperature=0.3,
                context="question_validation"
            )
            
            # 解析响应（优先JSON，其次文本格式）
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                is_valid = result.get("valid", False)
                reason = result.get("reason", "验证失败")
                return is_valid, reason

            valid_match = re.search(r'valid\s*[:：]\s*(true|false|yes|no)', response, re.IGNORECASE)
            reason_match = re.search(r'reason\s*[:：]\s*(.+)', response, re.IGNORECASE)
            if valid_match:
                is_valid = valid_match.group(1).lower() in {"true", "yes"}
                reason = reason_match.group(1).strip() if reason_match else "验证结果未提供原因"
                return is_valid, reason

            return False, "无法解析验证结果"
            
        except Exception as e:
            print(f"[WARNING] 问题验证失败: {e}")
            return False, f"验证过程出错: {str(e)}"
    
    async def validate_async(
        self,
        question: str,
        image_base64: str,
        pipeline_config: Dict[str, Any],
        global_constraints: Dict[str, Any],
        async_client: Optional[AsyncGeminiClient] = None,
        model: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        异步验证问题是否符合约束
        
        Args:
            question: 生成的问题文本
            image_base64: 图片的base64编码
            pipeline_config: Pipeline配置
            global_constraints: 全局约束
            async_client: 异步客户端实例（可选）
            model: 模型名称（可选）
            
        Returns:
            (是否通过验证, 验证原因)
        """
        # 基本检查
        if not question or len(question.strip()) == 0:
            return False, "问题为空"
        
        # 检查全局约束
        if not self._check_global_constraints(question, global_constraints):
            return False, "违反全局约束"
        
        # 检查Pipeline特定约束
        if not self._check_pipeline_constraints(question, pipeline_config):
            return False, "违反Pipeline约束"

        # 对于“概念/对象识别类”的通用问题，跳过LLM深度验证
        if self.enable_validation_exemptions:
            pipeline_name = pipeline_config.get("intent") or pipeline_config.get("name")
            if pipeline_name in self.exempt_pipelines:
                return True, "skip_validation_for_exempt_pipeline"
        
        # 使用LLM进行深度验证
        is_valid, reason = await self._validate_with_llm_async(
            question=question,
            image_base64=image_base64,
            pipeline_config=pipeline_config,
            global_constraints=global_constraints,
            async_client=async_client,
            model=model
        )
        
        return is_valid, reason
    
    async def _validate_with_llm_async(
        self,
        question: str,
        image_base64: str,
        pipeline_config: Dict[str, Any],
        global_constraints: Dict[str, Any],
        async_client: Optional[AsyncGeminiClient] = None,
        model: Optional[str] = None
    ) -> tuple[bool, str]:
        """异步使用LLM进行深度验证"""
        validation_rules = global_constraints.get("validation_rules", [])
        question_constraints = pipeline_config.get("question_constraints", [])
        
        prompt = f"""You are a VQA question validation expert. Validate whether the given question meets all requirements.

Question: "{question}"

Pipeline Intent: {pipeline_config.get("intent", "")}

Global Validation Rules:
{chr(10).join(f"- {rule}" for rule in validation_rules)}

Pipeline-Specific Constraints:
{chr(10).join(f"- {constraint}" for constraint in question_constraints)}

Check if the question:
1. Explicitly references at least one visual entity or region in the image
2. Is answerable solely based on the image
3. Would have a different answer if the image content changes
4. Does NOT rely on external knowledge or commonsense only
5. Follows the pipeline intent and constraints
6. Does NOT directly reveal the answer, and the answer is NOT directly inferable from the question text alone (without the image)

Return your response in plain text using this format:
valid: true/false
reason: <short explanation>"""

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
            
            # 确定使用的模型名称
            if model is None:
                if async_client is not None:
                    model = async_client.model_name
                else:
                    from QA_Generator.config import config
                    model = config.MODEL_NAME
            
            # 使用异步客户端
            if async_client is None:
                async with AsyncGeminiClient() as client:
                    response = await client.chat.completions.create(
                        model=model,
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
                    model=model,
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
                stage="question_validation",
                prompt=prompt,
                response=response_text,
                context={"question": question, "pipeline_intent": pipeline_config.get("intent", "")},
            )
            
            # 解析响应（优先JSON，其次文本格式）
            try:
                result = json.loads(response_text)
                is_valid = result.get("valid", False)
                reason = result.get("reason", "验证失败")
                return is_valid, reason
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    is_valid = result.get("valid", False)
                    reason = result.get("reason", "验证失败")
                    return is_valid, reason

            valid_match = re.search(r'valid\s*[:：]\s*(true|false|yes|no)', response_text, re.IGNORECASE)
            reason_match = re.search(r'reason\s*[:：]\s*(.+)', response_text, re.IGNORECASE)
            if valid_match:
                is_valid = valid_match.group(1).lower() in {"true", "yes"}
                reason = reason_match.group(1).strip() if reason_match else "验证结果未提供原因"
                return is_valid, reason

            return False, "无法解析验证结果"
            
        except Exception as e:
            print(f"[WARNING] 异步问题验证失败: {e}")
            log_model_response(
                stage="question_validation",
                prompt=prompt,
                response="",
                context={"question": question, "pipeline_intent": pipeline_config.get("intent", "")},
                error=str(e),
            )
            return False, f"验证过程出错: {str(e)}"

