"""
对象预填充处理模块
处理两种输入方式：claim和target object

注意：claim方式不再提取对象，直接将claim作为问题生成prompt的一部分
"""
from typing import Dict, Any, Optional
import json
import re
from QA_Generator.clients.gemini_client import GeminiClient
from QA_Generator.clients.async_client import AsyncGeminiClient


class PrefillProcessor:
    """
    预填充对象处理器
    
    支持两种输入方式：
    1. Claim方式：提供一句包含基于对象的该图片的claim，从中提取目标对象
    2. Target Object方式：直接提供target object名字
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        初始化预填充处理器
        
        Args:
            gemini_client: Gemini客户端实例（可选，用于claim解析）
        """
        self.gemini_client = gemini_client or GeminiClient()
    
    def process_prefill(
        self,
        prefill_input: Dict[str, Any],
        image_input: Any,
        pipeline_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        处理预填充输入，提取目标对象信息
        
        Args:
            prefill_input: 预填充输入，包含以下字段之一：
                - "claim": 一句包含基于对象的该图片的claim（字符串）
                - "target_object": 目标对象名字（字符串）
                - "target_object_info": 目标对象详细信息（字典，包含name, category等）
            image_input: 图片输入（用于claim解析时的上下文）
            pipeline_config: Pipeline配置（用于上下文）
            
        Returns:
            目标对象信息字典，格式：
            {
                "name": "对象名称",
                "category": "对象类别",
                "source": "claim" 或 "target_object",  # 来源
                "claim": "原始claim（如果来源是claim）",
                "confidence": 1.0  # 预填充对象置信度为1.0
            }
            如果处理失败返回None
        """
        # 检查输入类型
        if "claim" in prefill_input and prefill_input["claim"]:
            # 方式1: 从claim中提取目标对象
            return self._extract_object_from_claim(
                claim=prefill_input["claim"],
                image_input=image_input,
                pipeline_config=pipeline_config
            )
        elif "target_object" in prefill_input and prefill_input["target_object"]:
            # 方式2: 直接使用target object
            return self._process_target_object(
                target_object=prefill_input["target_object"],
                target_object_info=prefill_input.get("target_object_info")
            )
        else:
            print(f"[WARNING] 预填充输入格式不正确，需要包含'claim'或'target_object'字段")
            return None
    
    def _extract_object_from_claim(
        self,
        claim: str,
        image_input: Any,
        pipeline_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理claim输入：使用模型从claim+图片中提取目标对象
        
        Args:
            claim: 一句包含基于对象的该图片的claim
            image_input: 图片输入（保留参数以保持接口一致性，但实际不使用）
            pipeline_config: Pipeline配置（保留参数以保持接口一致性，但实际不使用）
            
        Returns:
            目标对象信息字典（包含name与claim）
        """
        prompt = self._build_claim_selection_prompt(
            claim=claim,
            pipeline_config=pipeline_config
        )

        try:
            response = self.gemini_client.analyze_image(
                image_input=image_input,
                prompt=prompt,
                temperature=0.3,
                context="claim_object_select"
            )
            parsed = self._parse_json_from_response(response)
        except Exception as e:
            print(f"[WARNING] claim对象提取失败: {e}")
            return None

        name = (parsed.get("name") or "").strip()
        if not name:
            return None

        return {
            "name": name,
            "category": (parsed.get("category") or "").strip(),
            "source": "claim",
            "claim": claim,
            "confidence": float(parsed.get("confidence", 1.0)) if parsed.get("confidence") is not None else 1.0
        }
    
    def _process_target_object(
        self,
        target_object: str,
        target_object_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理直接提供的target object
        
        Args:
            target_object: 目标对象名字（字符串）
            target_object_info: 可选的详细信息（字典）
            
        Returns:
            目标对象信息字典
        """
        # 如果提供了详细信息，使用它；否则从target_object构建
        if target_object_info:
            return {
                "name": target_object_info.get("name", target_object),
                "category": target_object_info.get("category", ""),
                "source": "target_object",
                "confidence": 1.0
            }
        else:
            return {
                "name": target_object,
                "category": "",  # 可以后续通过LLM推断
                "source": "target_object",
                "confidence": 1.0
            }
    
    async def process_prefill_async(
        self,
        prefill_input: Dict[str, Any],
        image_base64: str,
        pipeline_config: Dict[str, Any],
        async_client: Optional[AsyncGeminiClient] = None,
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        异步处理预填充输入
        
        Args:
            prefill_input: 预填充输入
            image_base64: 图片base64编码
            pipeline_config: Pipeline配置
            async_client: 异步客户端实例（可选）
            model: 模型名称（可选）
            
        Returns:
            目标对象信息字典
        """
        # 检查输入类型
        if "claim" in prefill_input and prefill_input["claim"]:
            return await self._extract_object_from_claim_async(
                claim=prefill_input["claim"],
                image_base64=image_base64,
                pipeline_config=pipeline_config,
                async_client=async_client,
                model=model
            )
        elif "target_object" in prefill_input and prefill_input["target_object"]:
            # 同步处理即可
            return self._process_target_object(
                target_object=prefill_input["target_object"],
                target_object_info=prefill_input.get("target_object_info")
            )
        else:
            print(f"[WARNING] 预填充输入格式不正确")
            return None
    
    async def _extract_object_from_claim_async(
        self,
        claim: str,
        image_base64: str,
        pipeline_config: Dict[str, Any],
        async_client: Optional[AsyncGeminiClient] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        异步处理claim输入：使用模型从claim+图片中提取目标对象
        
        Args:
            claim: 一句包含基于对象的该图片的claim
            image_base64: 图片base64编码（保留参数以保持接口一致性，但实际不使用）
            pipeline_config: Pipeline配置（保留参数以保持接口一致性，但实际不使用）
            async_client: 异步客户端实例（保留参数以保持接口一致性，但实际不使用）
            model: 模型名称（保留参数以保持接口一致性，但实际不使用）
            
        Returns:
            目标对象信息字典（包含name与claim）
        """
        prompt = self._build_claim_selection_prompt(
            claim=claim,
            pipeline_config=pipeline_config
        )

        try:
            if async_client is None:
                async with AsyncGeminiClient(use_lb_client=False) as client:
                    response = await client.analyze_image_async(
                        image_input=image_base64,
                        prompt=prompt,
                        temperature=0.3
                    )
            else:
                response = await async_client.analyze_image_async(
                    image_input=image_base64,
                    prompt=prompt,
                    temperature=0.3
                )

            parsed = self._parse_json_from_response(response)
        except Exception as e:
            print(f"[WARNING] claim对象提取失败(异步): {e}")
            return None

        name = (parsed.get("name") or "").strip()
        if not name:
            return None

        return {
            "name": name,
            "category": (parsed.get("category") or "").strip(),
            "source": "claim",
            "claim": claim,
            "confidence": float(parsed.get("confidence", 1.0)) if parsed.get("confidence") is not None else 1.0
        }

    def _build_claim_selection_prompt(
        self,
        claim: str,
        pipeline_config: Dict[str, Any]
    ) -> str:
        intent = pipeline_config.get("intent", "")
        description = pipeline_config.get("description", "")
        return f"""You are selecting a target object for VQA question generation.

Given the claim and the image, identify the most relevant single target object that should be used for downstream question generation.

Pipeline Intent: {intent}
Description: {description}

Claim:
{claim}

Return your response in plain text using the following format:
name: <object name>
category: <object category or empty>
confidence: <0.0-1.0>

If no clear object is present, return:
name:
category:
confidence: 0.0"""

    def _parse_json_from_response(self, response_text: str) -> Dict[str, Any]:
        if not response_text:
            return {}

        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except Exception:
                pass

        name_match = re.search(r'name\s*[:：]\s*(.+)', response_text, re.IGNORECASE)
        category_match = re.search(r'category\s*[:：]\s*(.+)', response_text, re.IGNORECASE)
        confidence_match = re.search(r'confidence\s*[:：]\s*([0-9]*\.?[0-9]+)', response_text, re.IGNORECASE)

        name = name_match.group(1).strip() if name_match else ""
        category = category_match.group(1).strip() if category_match else ""
        confidence = confidence_match.group(1).strip() if confidence_match else ""

        if not name:
            lines = [line.strip() for line in response_text.splitlines() if line.strip()]
            if lines:
                name = re.sub(r'^(name\s*[:：]\s*)', '', lines[0], flags=re.IGNORECASE).strip()

        data: Dict[str, Any] = {}
        if name:
            data["name"] = name
        if category:
            data["category"] = category
        if confidence:
            try:
                data["confidence"] = float(confidence)
            except ValueError:
                pass
        return data
