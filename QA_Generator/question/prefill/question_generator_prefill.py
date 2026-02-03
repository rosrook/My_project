"""
问题生成模块（预填充对象版本）
基于指定的目标对象生成问题
"""
from typing import Dict, Any, Optional
from QA_Generator.clients.gemini_client import GeminiClient
from QA_Generator.clients.async_client import AsyncGeminiClient
import re


class QuestionGeneratorPrefill:
    """
    问题生成器（预填充对象版本）
    
    与原始版本的主要区别：
    - 必须使用预填充的目标对象（不能为空）
    - prompt中强调使用指定的目标对象
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        初始化问题生成器
        
        Args:
            gemini_client: Gemini客户端实例
        """
        self.gemini_client = gemini_client or GeminiClient()
    
    def generate_question(
        self,
        image_input: Any,
        pipeline_config: Dict[str, Any],
        slots: Dict[str, str],
        prefill_object: Dict[str, Any],  # 预填充的对象信息（必需）
        question_type: Optional[str] = None
    ) -> Optional[str]:
        """
        生成VQA问题（使用预填充对象）
        
        Args:
            image_input: 图片输入
            pipeline_config: Pipeline配置
            slots: 填充的槽位字典
            prefill_object: 预填充的目标对象信息（必需，不能为None）
            question_type: 题型，可选值："multiple_choice"（选择题）或"fill_in_blank"（填空题）
            
        Returns:
            生成的问题文本，如果生成失败返回None
        """
        # 验证预填充对象（新格式：claim + prefilled_values）
        if not prefill_object:
            print(f"[WARNING] 预填充对象为空，无法生成问题")
            return None
        
        # 新流程统一使用 claim + prefilled_values 格式
        if not prefill_object.get("claim"):
            print(f"[WARNING] 预填充对象缺少 claim 字段，无法生成问题")
            return None
        
        # 验证 slots 是否已填充（必需槽位应该已经填充完成）
        if not slots:
            print(f"[WARNING] 槽位未填充，无法生成问题")
            return None
        
        # 获取配置
        intent = pipeline_config.get("intent", "")
        example_template = pipeline_config.get("example_template", "")
        question_constraints = pipeline_config.get("question_constraints", [])
        description = pipeline_config.get("description", "")
        
        # 构建prompt（强调使用预填充对象）
        prompt = self._build_generation_prompt(
            intent=intent,
            description=description,
            example_template=example_template,
            question_constraints=question_constraints,
            slots=slots,
            prefill_object=prefill_object,
            question_type=question_type
        )
        
        try:
            # 使用LLM生成问题
            response = self.gemini_client.analyze_image(
                image_input=image_input,
                prompt=prompt,
                temperature=0.7,
                context="question_generation_prefill"
            )
            
            # 提取问题（可能包含引号或其他格式）
            question = self._extract_question(response)
            
            return question
            
        except Exception as e:
            print(f"[WARNING] 问题生成失败: {e}")
            return None
    
    def _build_generation_prompt(
        self,
        intent: str,
        description: str,
        example_template: str,
        question_constraints: list,
        slots: Dict[str, str],
        prefill_object: Dict[str, Any],  # 预填充对象（必需）
        question_type: Optional[str] = None
    ) -> str:
        """
        构建问题生成prompt（强调使用预填充对象）
        """
        # 槽位信息
        slot_info = ""
        if slots:
            slot_info = "\nFilled Slots:\n"
            for key, value in slots.items():
                slot_info += f"- {key}: {value}\n"
        
        # 预填充对象信息（新格式：claim + prefilled_values）
        # 直接使用 claim 和已填充的 slots 生成问题
        claim = prefill_object.get('claim', '')
        prefilled_values = prefill_object.get('prefilled_values', {})
        
        object_info = f"\n**REQUIRED: Generate question based on the following claim template:**\n"
        object_info += f"Claim Template: {claim}\n"
        if prefilled_values:
            prefilled_str = ", ".join([f"{k}={v}" for k, v in prefilled_values.items()])
            object_info += f"Prefilled Values: {prefilled_str}\n"
        object_info += f"\nYou MUST generate a question based on this claim template and the filled slots below. "
        object_info += f"The question should reference the objects and relationships described in the claim and be answerable from the image.\n"
        
        # 题型要求
        question_type_instruction = ""
        if question_type == "multiple_choice":
            question_type_instruction = "\nQuestion Type: MULTIPLE CHOICE\n- Generate ONLY the question stem, NOT the answer options\n- The question should be phrased to expect a selection from multiple options\n- Do NOT include any answer choices or options in your response\n"
        elif question_type == "fill_in_blank":
            question_type_instruction = "\nQuestion Type: FILL IN THE BLANK\n- Generate a question that requires a direct answer\n- The question should be phrased to expect a direct textual or numerical answer\n"
        
        prompt = f"""You are a VQA question generation expert. Generate a natural language question based on the given specifications.

Pipeline Intent: {intent}
Description: {description}
{object_info}
{slot_info}
Example Template: "{example_template}"
{question_type_instruction}
Question Constraints:
{chr(10).join(f"- {constraint}" for constraint in question_constraints)}

**CRITICAL REQUIREMENT:**
{self._build_critical_requirement(prefill_object, slots)}

Requirements:
1. {self._build_requirement_1(prefill_object, slots)}
2. The question must be grounded in the image (explicitly reference visual entities)
3. The question must be answerable using the image alone
4. You should vary the question from the example template in multiple ways while preserving the intent:
   - Use different wording, phrasing, and sentence structures
   - Vary the questioning style or approach when appropriate
   - Avoid copying the example template verbatim, but maintain the same semantic intent
5. You must NOT introduce new objects not in the image
6. You must NOT use external or commonsense-only knowledge

Generate a natural, fluent question that follows the template and constraints. Return ONLY the question text (question stem for multiple choice, NO options), no explanation or additional text."""
        
        return prompt
    
    def _build_critical_requirement(self, prefill_object: Dict[str, Any], slots: Dict[str, str]) -> str:
        """构建关键要求文本（新格式：基于 claim 和已填充的 slots）"""
        claim = prefill_object.get('claim', '')
        slot_names = list(slots.keys())
        slot_values = list(slots.values())
        
        requirement = f"You MUST generate a question based on the claim template: \"{claim}\". "
        requirement += f"The question MUST use the filled slots ({', '.join(slot_names)}) with their values ({', '.join(slot_values)}). "
        requirement += f"The question should be answerable from the image and reference the objects/relationships described in the claim."
        
        return requirement
    
    def _build_requirement_1(self, prefill_object: Dict[str, Any], slots: Dict[str, str]) -> str:
        """构建第一个要求文本（新格式：基于 claim 和已填充的 slots）"""
        claim = prefill_object.get('claim', '')
        slot_info = ", ".join([f"{k}={v}" for k, v in slots.items()])
        
        requirement = f"The question MUST be based on the claim template \"{claim}\" and use the filled slots: {slot_info}"
        
        return requirement
    
    def _extract_question(self, response: str) -> str:
        """从LLM响应中提取问题文本"""
        # 移除可能的引号
        question = response.strip()
        
        # 移除首尾引号
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]
        elif question.startswith("'") and question.endswith("'"):
            question = question[1:-1]
        
        # 移除可能的"Question:"等前缀
        question = re.sub(r'^(Question|Q|问题)[:：]\s*', '', question, flags=re.IGNORECASE)
        
        # 只取第一行（问题通常是一行）
        question = question.split('\n')[0].strip()
        
        return question
    
    async def generate_question_async(
        self,
        image_base64: str,
        pipeline_config: Dict[str, Any],
        slots: Dict[str, str],
        prefill_object: Dict[str, Any],
        question_type: Optional[str] = None,
        async_client: Optional[AsyncGeminiClient] = None,
        model: Optional[str] = None
    ) -> Optional[str]:
        """
        异步生成问题（使用预填充对象）
        
        Args:
            image_base64: 图片base64编码
            pipeline_config: Pipeline配置
            slots: 填充的槽位字典
            prefill_object: 预填充的目标对象信息（新格式：claim + prefilled_values）
            question_type: 题型
            async_client: 异步客户端实例（可选）
            model: 模型名称（可选）

        Returns:
            生成的问题文本，如果生成失败返回None
        """
        # 验证预填充对象（新格式：claim + prefilled_values）
        if not prefill_object:
            print(f"[WARNING] 预填充对象为空，无法生成问题")
            return None
        
        # 新流程统一使用 claim + prefilled_values 格式
        if not prefill_object.get("claim"):
            print(f"[WARNING] 预填充对象缺少 claim 字段，无法生成问题")
            return None
        
        # 验证 slots 是否已填充（必需槽位应该已经填充完成）
        if not slots:
            print(f"[WARNING] 槽位未填充，无法生成问题")
            return None
        
        # 获取配置
        intent = pipeline_config.get("intent", "")
        example_template = pipeline_config.get("example_template", "")
        question_constraints = pipeline_config.get("question_constraints", [])
        description = pipeline_config.get("description", "")
        
        # 构建prompt（强调使用预填充对象）
        prompt = self._build_generation_prompt(
            intent=intent,
            description=description,
            example_template=example_template,
            question_constraints=question_constraints,
            slots=slots,
            prefill_object=prefill_object,
            question_type=question_type
        )
        
        try:
            if async_client is None:
                async with AsyncGeminiClient(model_name=model, use_lb_client=False) as client:
                    response = await client.analyze_image_async(
                        image_input=image_base64,
                        prompt=prompt,
                        temperature=0.7
                    )
            else:
                response = await async_client.analyze_image_async(
                    image_input=image_base64,
                    prompt=prompt,
                    temperature=0.7
                )
            
            question = self._extract_question(response)
            return question
        
        except Exception as e:
            print(f"[WARNING] 问题生成失败(异步): {e}")
            return None
