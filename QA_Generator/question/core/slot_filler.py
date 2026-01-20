"""
槽位填充模块
根据配置填充required_slots和optional_slots
"""
from typing import Dict, Any, Optional, List
from QA_Generator.clients.gemini_client import GeminiClient
from QA_Generator.clients.async_client import AsyncGeminiClient
import random


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
        print("[DEBUG] SlotFiller.fill_slots: start")
        print(f"[DEBUG] required_slots: {pipeline_config.get('required_slots', [])}")
        print(f"[DEBUG] optional_slots: {pipeline_config.get('optional_slots', [])}")
        if selected_object:
            print(f"[DEBUG] selected_object: {selected_object}")
        else:
            print("[DEBUG] selected_object: None")
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
                print(f"[DEBUG] required slot '{slot}' unresolved -> discard")
                print(f"[WARNING] 必需槽位 '{slot}' 无法解析，丢弃样本")
                return None
            
            slots[slot] = value
        
        # 填充可选槽位（随机采样以增加多样性）
        optional_slots = pipeline_config.get("optional_slots", [])
        for slot in optional_slots:
            # 随机决定是否填充（增加多样性）
            rand_val = random.random()
            print(f"[DEBUG] optional slot '{slot}' random={rand_val:.3f}")
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
                    print(f"[DEBUG] optional slot '{slot}' resolved='{value}'")
                else:
                    print(f"[DEBUG] optional slot '{slot}' unresolved")
            else:
                print(f"[DEBUG] optional slot '{slot}' skipped")
        
        print(f"[DEBUG] SlotFiller.fill_slots: result={slots}")
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
        print(
            f"[DEBUG] _resolve_slot: slot='{slot}', "
            f"is_optional={is_optional}, "
            f"has_selected_object={bool(selected_object)}"
        )
        if slot in ["object", "objects"] and selected_object:
            if slot == "object":
                value = selected_object.get("name", "")
                print(f"[DEBUG] _resolve_slot: from selected_object.name -> '{value}'")
                return value
            elif slot == "objects":
                # 对于复数形式，可能需要返回对象类别
                value = selected_object.get("name", "")
                print(f"[DEBUG] _resolve_slot: from selected_object.name -> '{value}'")
                return value
        
        # 从图像信息中解析
        if slot in ["region", "spatial_granularity", "direction_granularity"]:
            value = self._resolve_from_image(
                slot=slot,
                image_input=image_input,
                pipeline_config=pipeline_config
            )
            print(f"[DEBUG] _resolve_slot: from image -> '{value}'")
            return value
        
        # 其他槽位的默认值
        slot_defaults = {
            "object_category_granularity": random.choice(["basic", "detailed"]),
            "caption_style": random.choice(["descriptive", "concise"]),
            "location_granularity": random.choice(["city", "landmark", "region"]),
            "platform_context": random.choice(["twitter", "instagram", "facebook"]),
            "expression_format": random.choice(["percentage", "fraction", "ratio"]),
            "spatial_granularity": random.choice(["coarse", "fine"]),
            "reference_frame": random.choice(["absolute", "relative"]),
            "region_partition": random.choice(["corners", "quadrants", "grid"]),
            "direction_granularity": random.choice(["cardinal", "intercardinal", "fine"]),
            "count_scope": random.choice(["all", "visible", "distinct"])
        }
        
        if slot in slot_defaults:
            value = slot_defaults[slot]
            print(f"[DEBUG] _resolve_slot: from defaults -> '{value}'")
            return value
        
        # 如果无法解析且是可选槽位，返回None
        if is_optional:
            print("[DEBUG] _resolve_slot: optional unresolved -> None")
            return None
        
        # 必需槽位无法解析，尝试使用LLM
        print("[DEBUG] _resolve_slot: calling LLM fallback")
        value = self._resolve_with_llm(
            slot=slot,
            image_input=image_input,
            pipeline_config=pipeline_config,
            selected_object=selected_object
        )
        print(f"[DEBUG] _resolve_slot: from llm -> '{value}'")
        return value
    
    async def fill_slots_async(
        self,
        image_base64: str,
        pipeline_config: Dict[str, Any],
        selected_object: Optional[Dict[str, Any]] = None
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
        print("[DEBUG] SlotFiller.fill_slots_async: start")
        print(f"[DEBUG] required_slots: {pipeline_config.get('required_slots', [])}")
        print(f"[DEBUG] optional_slots: {pipeline_config.get('optional_slots', [])}")
        if selected_object:
            print(f"[DEBUG] selected_object: {selected_object}")
        else:
            print("[DEBUG] selected_object: None")
        required_slots = pipeline_config.get("required_slots", [])
        for slot in required_slots:
            value = await self._resolve_slot_async(
                slot=slot,
                image_base64=image_base64,
                pipeline_config=pipeline_config,
                selected_object=selected_object
            )
            
            if value is None:
                # 必需槽位无法解析，丢弃
                print(f"[DEBUG] required slot '{slot}' unresolved -> discard")
                print(f"[WARNING] 必需槽位 '{slot}' 无法解析，丢弃样本")
                return None
            
            slots[slot] = value
        
        # 填充可选槽位（随机采样以增加多样性）
        optional_slots = pipeline_config.get("optional_slots", [])
        for slot in optional_slots:
            # 随机决定是否填充（增加多样性）
            rand_val = random.random()
            print(f"[DEBUG] optional slot '{slot}' random={rand_val:.3f}")
            if rand_val < 0.5:  # 50%概率填充可选槽位
                value = await self._resolve_slot_async(
                    slot=slot,
                    image_base64=image_base64,
                    pipeline_config=pipeline_config,
                    selected_object=selected_object,
                    is_optional=True
                )
                if value is not None:
                    slots[slot] = value
                    print(f"[DEBUG] optional slot '{slot}' resolved='{value}'")
                else:
                    print(f"[DEBUG] optional slot '{slot}' unresolved")
            else:
                print(f"[DEBUG] optional slot '{slot}' skipped")
        
        print(f"[DEBUG] SlotFiller.fill_slots_async: result={slots}")
        return slots
    
    async def _resolve_slot_async(
        self,
        slot: str,
        image_base64: str,
        pipeline_config: Dict[str, Any],
        selected_object: Optional[Dict[str, Any]] = None,
        is_optional: bool = False
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
        print(
            f"[DEBUG] _resolve_slot_async: slot='{slot}', "
            f"is_optional={is_optional}, "
            f"has_selected_object={bool(selected_object)}"
        )
        if slot in ["object", "objects"] and selected_object:
            if slot == "object":
                value = selected_object.get("name", "")
                print(f"[DEBUG] _resolve_slot_async: from selected_object.name -> '{value}'")
                return value
            elif slot == "objects":
                # 对于复数形式，可能需要返回对象类别
                value = selected_object.get("name", "")
                print(f"[DEBUG] _resolve_slot_async: from selected_object.name -> '{value}'")
                return value
        
        # 从图像信息中解析
        if slot in ["region", "spatial_granularity", "direction_granularity"]:
            value = self._resolve_from_image_async(
                slot=slot,
                pipeline_config=pipeline_config
            )
            print(f"[DEBUG] _resolve_slot_async: from image -> '{value}'")
            return value
        
        # 其他槽位的默认值
        slot_defaults = {
            "object_category_granularity": random.choice(["basic", "detailed"]),
            "caption_style": random.choice(["descriptive", "concise"]),
            "location_granularity": random.choice(["city", "landmark", "region"]),
            "platform_context": random.choice(["twitter", "instagram", "facebook"]),
            "expression_format": random.choice(["percentage", "fraction", "ratio"]),
            "spatial_granularity": random.choice(["coarse", "fine"]),
            "reference_frame": random.choice(["absolute", "relative"]),
            "region_partition": random.choice(["corners", "quadrants", "grid"]),
            "direction_granularity": random.choice(["cardinal", "intercardinal", "fine"]),
            "count_scope": random.choice(["all", "visible", "distinct"])
        }
        
        if slot in slot_defaults:
            value = slot_defaults[slot]
            print(f"[DEBUG] _resolve_slot_async: from defaults -> '{value}'")
            return value
        
        # 如果无法解析且是可选槽位，返回None
        if is_optional:
            print("[DEBUG] _resolve_slot_async: optional unresolved -> None")
            return None
        
        # 必需槽位无法解析，返回None（不在这里使用LLM，保持简单）
        print("[DEBUG] _resolve_slot_async: LLM fallback not used in async path")
        print("[DEBUG] _resolve_slot_async: unresolved -> None")
        return None
    
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

