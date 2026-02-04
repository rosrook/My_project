"""
槽位填充模块
根据配置填充required_slots和optional_slots
"""
from typing import Dict, Any, Optional, List, Tuple
import json
import re
import random
from QA_Generator.clients.gemini_client import GeminiClient
from QA_Generator.clients.async_client import AsyncGeminiClient
from QA_Generator.logging.logger import log_debug
from QA_Generator.utils.model_response_logger import log_model_response


class SlotFiller:
    """槽位填充器"""
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        初始化槽位填充器
        
        Args:
            gemini_client: Gemini客户端实例
        """
        self.gemini_client = gemini_client or GeminiClient()
    
    def fill_slots(
        self,
        image_input: Any,
        pipeline_config: Dict[str, Any],
        selected_object: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, str]]:
        """
        填充槽位
        
        Args:
            image_input: 图片输入
            pipeline_config: Pipeline配置
            selected_object: 选中的对象信息（如果有）
            
        Returns:
            填充后的槽位字典，如果必需槽位无法填充则返回None
        """
        slots = {}
        
        # 填充必需槽位
        log_debug("SlotFiller.fill_slots: start")
        log_debug(f"required_slots: {pipeline_config.get('required_slots', [])}")
        log_debug(f"optional_slots: {pipeline_config.get('optional_slots', [])}")
        if selected_object:
            log_debug(f"selected_object: {selected_object}")
        else:
            log_debug("selected_object: None")
        required_slots = pipeline_config.get("required_slots", [])
        for slot in required_slots:
            value = self._resolve_slot(
                slot=slot,
                image_input=image_input,
                pipeline_config=pipeline_config,
                selected_object=selected_object
            )
            
            if value is None:
                # 必需槽位无法解析，丢弃
                log_debug(f"required slot '{slot}' unresolved -> discard")
                print(f"[WARNING] 必需槽位 '{slot}' 无法解析，丢弃样本")
                return None
            
            slots[slot] = value
        
        # 填充可选槽位（随机采样以增加多样性）
        optional_slots = pipeline_config.get("optional_slots", [])
        for slot in optional_slots:
            # 随机决定是否填充（增加多样性）
            rand_val = random.random()
            log_debug(f"optional slot '{slot}' random={rand_val:.3f}")
            if rand_val < 0.5:  # 50%概率填充可选槽位
                value = self._resolve_slot(
                    slot=slot,
                    image_input=image_input,
                    pipeline_config=pipeline_config,
                    selected_object=selected_object,
                    is_optional=True
                )
                if value is not None:
                    slots[slot] = value
                    log_debug(f"optional slot '{slot}' resolved='{value}'")
                else:
                    log_debug(f"optional slot '{slot}' unresolved")
            else:
                log_debug(f"optional slot '{slot}' skipped")
        
        log_debug(f"SlotFiller.fill_slots: result={slots}")
        return slots
    
    def _resolve_slot(
        self,
        slot: str,
        image_input: Any,
        pipeline_config: Dict[str, Any],
        selected_object: Optional[Dict[str, Any]] = None,
        is_optional: bool = False
    ) -> Optional[str]:
        """
        解析单个槽位值
        
        Args:
            slot: 槽位名称
            image_input: 图片输入
            pipeline_config: Pipeline配置
            selected_object: 选中的对象信息
            is_optional: 是否为可选槽位
            
        Returns:
            槽位值，如果无法解析返回None
        """
        # 从选中对象中解析
        log_debug(
            f"_resolve_slot: slot='{slot}', "
            f"is_optional={is_optional}, "
            f"has_selected_object={bool(selected_object)}"
        )
        if slot in ["object", "objects"] and selected_object:
            if slot == "object":
                value = selected_object.get("name", "")
                log_debug(f"_resolve_slot: from selected_object.name -> '{value}'")
                return value
            elif slot == "objects":
                # 对于复数形式，可能需要返回对象类别
                value = selected_object.get("name", "")
                log_debug(f"_resolve_slot: from selected_object.name -> '{value}'")
                return value
        
        # 从图像信息中解析
        if slot in ["region", "spatial_granularity", "direction_granularity"]:
            value = self._resolve_from_image(
                slot=slot,
                image_input=image_input,
                pipeline_config=pipeline_config
            )
            log_debug(f"_resolve_slot: from image -> '{value}'")
            return value
        
        # 其他槽位的默认值
        slot_defaults = {
            "object_category_granularity": random.choice(["basic", "detailed"]),
            "caption_style": random.choice(["descriptive", "concise"]),
            # "location_granularity": random.choice(["city", "landmark", "region"]),
            "platform_context": random.choice(["twitter", "instagram", "facebook"]),
            "expression_format": random.choice(["percentage", "fraction", "ratio"]),
            "spatial_granularity": random.choice(["coarse", "fine"]),
            "reference_frame": random.choice(["absolute", "relative"]),
            "region_partition": random.choice(["corners", "quadrants"]),
            "direction_granularity": random.choice(["cardinal", "intercardinal", "fine"]),
            "count_scope": random.choice(["all"])
        }
        
        if slot in slot_defaults:
            value = slot_defaults[slot]
            log_debug(f"_resolve_slot: from defaults -> '{value}'")
            return value
        
        # 如果无法解析且是可选槽位，返回None
        if is_optional:
            log_debug("_resolve_slot: optional unresolved -> None")
            return None
        
        # 必需槽位无法解析，尝试使用LLM
        log_debug("_resolve_slot: calling LLM fallback")
        value = self._resolve_with_llm(
            slot=slot,
            image_input=image_input,
            pipeline_config=pipeline_config,
            selected_object=selected_object
        )
        log_debug(f"_resolve_slot: from llm -> '{value}'")
        return value
    
    async def fill_slots_async(
        self,
        image_base64: str,
        pipeline_config: Dict[str, Any],
        selected_object: Optional[Dict[str, Any]] = None,
        async_client: Optional[AsyncGeminiClient] = None,
        fallback_events: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, str]]:
        """
        异步填充槽位
        
        Args:
            image_base64: 图片的base64编码
            pipeline_config: Pipeline配置
            selected_object: 选中的对象信息（如果有）
            
        Returns:
            填充后的槽位字典，如果必需槽位无法填充则返回None
        """
        slots = {}
        
        # 填充必需槽位
        log_debug("SlotFiller.fill_slots_async: start")
        log_debug(f"required_slots: {pipeline_config.get('required_slots', [])}")
        log_debug(f"optional_slots: {pipeline_config.get('optional_slots', [])}")
        if selected_object:
            log_debug(f"selected_object: {selected_object}")
        else:
            log_debug("selected_object: None")
        required_slots = pipeline_config.get("required_slots", [])
        for slot in required_slots:
            value = await self._resolve_slot_async(
                slot=slot,
                image_base64=image_base64,
                pipeline_config=pipeline_config,
                selected_object=selected_object,
                async_client=async_client,
                fallback_events=fallback_events
            )
            
            if value is None:
                # 必需槽位无法解析，丢弃
                log_debug(f"required slot '{slot}' unresolved -> discard")
                print(f"[WARNING] 必需槽位 '{slot}' 无法解析，丢弃样本")
                return None
            
            slots[slot] = value
        
        # 填充可选槽位（随机采样以增加多样性）
        optional_slots = pipeline_config.get("optional_slots", [])
        for slot in optional_slots:
            # 随机决定是否填充（增加多样性）
            rand_val = random.random()
            log_debug(f"optional slot '{slot}' random={rand_val:.3f}")
            if rand_val < 0.5:  # 50%概率填充可选槽位
                value = await self._resolve_slot_async(
                    slot=slot,
                    image_base64=image_base64,
                    pipeline_config=pipeline_config,
                    selected_object=selected_object,
                    is_optional=True,
                    async_client=async_client,
                    fallback_events=fallback_events
                )
                if value is not None:
                    slots[slot] = value
                    log_debug(f"optional slot '{slot}' resolved='{value}'")
                else:
                    log_debug(f"optional slot '{slot}' unresolved")
            else:
                log_debug(f"optional slot '{slot}' skipped")
        
        log_debug(f"SlotFiller.fill_slots_async: result={slots}")
        return slots
    
    async def _resolve_slot_async(
        self,
        slot: str,
        image_base64: str,
        pipeline_config: Dict[str, Any],
        selected_object: Optional[Dict[str, Any]] = None,
        is_optional: bool = False,
        async_client: Optional[AsyncGeminiClient] = None,
        fallback_events: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """
        异步解析单个槽位值
        
        Args:
            slot: 槽位名称
            image_base64: 图片的base64编码
            pipeline_config: Pipeline配置
            selected_object: 选中的对象信息
            is_optional: 是否为可选槽位
            
        Returns:
            槽位值，如果无法解析返回None
        """
        # 从选中对象中解析
        log_debug(
            f"_resolve_slot_async: slot='{slot}', "
            f"is_optional={is_optional}, "
            f"has_selected_object={bool(selected_object)}"
        )
        if slot in ["object", "objects"] and selected_object:
            if slot == "object":
                value = selected_object.get("name", "")
                log_debug(f"_resolve_slot_async: from selected_object.name -> '{value}'")
                return value
            elif slot == "objects":
                # 对于复数形式，可能需要返回对象类别
                value = selected_object.get("name", "")
                log_debug(f"_resolve_slot_async: from selected_object.name -> '{value}'")
                return value
        
        # 从图像信息中解析
        if slot in ["region", "spatial_granularity", "direction_granularity"]:
            value = self._resolve_from_image_async(
                slot=slot,
                pipeline_config=pipeline_config
            )
            log_debug(f"_resolve_slot_async: from image -> '{value}'")
            return value
        
        # 其他槽位的默认值
        slot_defaults = {
            "object_category_granularity": random.choice(["basic", "detailed"]),
            "caption_style": random.choice(["descriptive", "concise"]),
            # "location_granularity": random.choice(["city", "landmark", "region"]),
            "platform_context": random.choice(["twitter", "instagram", "facebook"]),
            "expression_format": random.choice(["percentage", "fraction", "ratio"]),
            "spatial_granularity": random.choice(["coarse", "fine"]),
            "reference_frame": random.choice(["absolute", "relative"]),
            "region_partition": random.choice(["corners", "quadrants"]),
            "direction_granularity": random.choice(["cardinal", "intercardinal", "fine"]),
            "count_scope": random.choice(["all"])
        }
        
        if slot in slot_defaults:
            value = slot_defaults[slot]
            log_debug(f"_resolve_slot_async: from defaults -> '{value}'")
            return value
        
        # 如果无法解析且是可选槽位，返回None
        if is_optional:
            log_debug("_resolve_slot_async: optional unresolved -> None")
            return None
        
        # 必需槽位无法解析，启用LLM兜底
        fallback_value, fallback_detail = await self._resolve_with_llm_async(
            slot=slot,
            image_base64=image_base64,
            pipeline_config=pipeline_config,
            selected_object=selected_object,
            async_client=async_client
        )
        if fallback_value:
            print(f"[WARNING] 触发LLM兜底: slot='{slot}', value='{fallback_value}'")
            if fallback_events is not None:
                fallback_events.append({
                    "slot": slot,
                    "value": fallback_value,
                    "detail": fallback_detail
                })
            return fallback_value

        log_debug("_resolve_slot_async: LLM fallback failed -> None")
        return None

    async def _resolve_with_llm_async(
        self,
        slot: str,
        image_base64: str,
        pipeline_config: Dict[str, Any],
        selected_object: Optional[Dict[str, Any]] = None,
        async_client: Optional[AsyncGeminiClient] = None
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        使用LLM兜底解析槽位值（异步）
        """
        intent = pipeline_config.get("intent", "")
        description = pipeline_config.get("description", "")
        required_slots = pipeline_config.get("required_slots", [])
        optional_slots = pipeline_config.get("optional_slots", [])
        selected_object_name = selected_object.get("name") if isinstance(selected_object, dict) else ""
        selected_object_category = selected_object.get("category") if isinstance(selected_object, dict) else ""

        prompt = f"""You are a VQA slot filler. Provide a short value for the requested slot.

Pipeline Intent: {intent}
Pipeline Description: {description}
Required Slots: {required_slots}
Optional Slots: {optional_slots}
Selected Object: name={selected_object_name}, category={selected_object_category}
Target Slot: {slot}

Return your response in plain text using this format:
value: <string>
"""
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

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                value = parsed.get("value")
                if isinstance(value, str):
                    value = value.strip()
                if value:
                    return value, {"raw_response": response, "parsed": parsed}

            value_match = re.search(r'value\s*[:：]\s*(.+)', response, re.IGNORECASE)
            if value_match:
                value = value_match.group(1).strip().strip('"\'')
                if value:
                    return value, {"raw_response": response}

            lines = [line.strip() for line in response.splitlines() if line.strip()]
            if lines:
                value = re.sub(r'^(value\s*[:：]\s*)', '', lines[0], flags=re.IGNORECASE).strip()
                if value:
                    return value, {"raw_response": response}

            return None, {"raw_response": response}
        except Exception as e:
            return None, {"error": str(e)}
    
    def _resolve_from_image_async(
        self,
        slot: str,
        pipeline_config: Dict[str, Any]
    ) -> Optional[str]:
        """从图像中解析槽位值（异步版本使用相同逻辑）"""
        # 这些槽位通常需要从图像分析中获取
        # 这里简化处理，返回默认值
        defaults = {
            "region": "center",
            "spatial_granularity": "coarse",
            "direction_granularity": "cardinal"
        }
        return defaults.get(slot)
    
    def _resolve_from_image(
        self,
        slot: str,
        image_input: Any,
        pipeline_config: Dict[str, Any]
    ) -> Optional[str]:
        """从图像中解析槽位值"""
        # 这些槽位通常需要从图像分析中获取
        # 这里简化处理，返回默认值
        defaults = {
            "region": "center",
            "spatial_granularity": "coarse",
            "direction_granularity": "cardinal"
        }
        return defaults.get(slot)
    
    def _resolve_with_llm(
        self,
        slot: str,
        image_input: Any,
        pipeline_config: Dict[str, Any],
        selected_object: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """使用LLM解析槽位值"""
        # 对于复杂槽位，可以使用LLM分析
        # 这里简化处理，返回None表示无法解析
        return None
    
    async def fill_required_slots_with_llm_async(
        self,
        image_base64: str,
        pipeline_config: Dict[str, Any],
        claim: str,
        prefilled_values: Dict[str, str],
        async_client: Optional[AsyncGeminiClient] = None
    ) -> Optional[Dict[str, str]]:
        """
        使用 LLM 一次性填充所有必需槽位（简化版本）
        
        Args:
            image_base64: 图片的base64编码
            pipeline_config: Pipeline配置
            claim: Claim template（包含占位符，如 "The [OBJECT_A] is [RELATIVE_DIRECTION] the [OBJECT_B]."）
            prefilled_values: 预填充的值（如 {"OBJECT_A": "dog", "OBJECT_B": "hurdle"}）
            async_client: 异步客户端实例（可选）
        
        Returns:
            填充后的槽位字典（如 {"object": "dog", "other_object": "hurdle"}），如果失败返回None
        """
        intent = pipeline_config.get("intent", "")
        description = pipeline_config.get("description", "")
        required_slots = pipeline_config.get("required_slots", [])
        
        if not required_slots:
            return {}
        
        # 构建 prompt，让 LLM 一次性填充所有 required_slots
        prefilled_str = ", ".join([f"{k}={v}" for k, v in prefilled_values.items()])
        
        prompt = f"""You are a VQA slot filler. Fill all required slots based on the claim template and prefilled values.

Pipeline Intent: {intent}
Pipeline Description: {description}
Claim Template: {claim}
Prefilled Values: {prefilled_str}
Required Slots: {required_slots}

Your task:
1. Map the prefilled placeholder values (like OBJECT_A, OBJECT_B) to the required slot names (like object, other_object)
2. Fill all required slots with appropriate values from the prefilled_values or infer from the claim template
3. Return a JSON object with slot names as keys and values as strings

Example:
- Claim: "The [OBJECT_A] is [RELATIVE_DIRECTION] the [OBJECT_B]."
- Prefilled Values: OBJECT_A=dog, OBJECT_B=hurdle
- Required Slots: ["object", "other_object"]
- Output: {{"object": "dog", "other_object": "hurdle"}}

Return your response as a JSON object:
{{
  "object": "...",
  "other_object": "...",
  ...
}}
"""
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
            
            log_model_response(
                stage="slot_filling",
                prompt=prompt,
                response=response,
                context={"claim": claim, "prefilled_values": prefilled_values, "required_slots": required_slots},
            )
            
            # 尝试解析 JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if isinstance(parsed, dict):
                        # 验证所有 required_slots 都有值
                        result = {}
                        for slot in required_slots:
                            value = parsed.get(slot)
                            if isinstance(value, str) and value.strip():
                                result[slot] = value.strip()
                            else:
                                # 缺少必需槽位，返回 None
                                log_debug(f"LLM response missing required slot '{slot}'")
                                return None
                        return result
                except json.JSONDecodeError:
                    pass
            
            # 如果 JSON 解析失败，尝试其他格式
            log_debug(f"Failed to parse LLM response as JSON: {response[:200]}")
            return None
            
        except Exception as e:
            log_debug(f"Error in fill_required_slots_with_llm_async: {str(e)}")
            log_model_response(
                stage="slot_filling",
                prompt=prompt,
                response="",
                context={"claim": claim, "prefilled_values": prefilled_values, "required_slots": required_slots},
                error=str(e),
            )
            return None

