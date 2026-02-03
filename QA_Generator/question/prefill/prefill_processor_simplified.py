"""
简化的预填充对象处理模块
统一处理方式：接收 claim 和 prefilled_values，不再区分两种模式
"""
from typing import Dict, Any, Optional


class PrefillProcessorSimplified:
    """
    简化的预填充对象处理器
    
    统一处理方式：
    - 接收 claim（claim template）和 prefilled_values（占位符和值的映射）
    - 不再区分 claim/target_object 两种模式
    - 直接返回 claim 和 prefilled_values，供后续槽位填充使用
    """
    
    def __init__(self):
        """初始化预填充处理器（不再需要 gemini_client）"""
        pass
    
    def process_prefill(
        self,
        prefill_input: Dict[str, Any],
        image_input: Any,
        pipeline_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        处理预填充输入（简化版本）
        
        Args:
            prefill_input: 预填充输入，包含：
                - "claim": claim template（字符串，包含占位符如 [OBJECT_A]）
                - "prefilled_values": 占位符和值的映射（字典，如 {"OBJECT_A": "dog", "OBJECT_B": "hurdle"}）
            image_input: 图片输入（保留参数以保持接口一致性，但实际不使用）
            pipeline_config: Pipeline配置（保留参数以保持接口一致性，但实际不使用）
            
        Returns:
            预填充信息字典，格式：
            {
                "claim": "claim template",
                "prefilled_values": {"OBJECT_A": "dog", "OBJECT_B": "hurdle"}
            }
            如果处理失败返回None
        """
        claim = prefill_input.get("claim")
        prefilled_values = prefill_input.get("prefilled_values", {})
        
        if not isinstance(claim, str) or not claim.strip():
            print(f"[WARNING] 预填充输入缺少有效的 claim 字段")
            return None
        
        if not isinstance(prefilled_values, dict):
            prefilled_values = {}
        
        return {
            "claim": claim.strip(),
            "prefilled_values": prefilled_values
        }
    
    async def process_prefill_async(
        self,
        prefill_input: Dict[str, Any],
        image_base64: str,
        pipeline_config: Dict[str, Any],
        async_client: Optional[Any] = None,
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        异步处理预填充输入（简化版本）
        
        Args:
            prefill_input: 预填充输入
            image_base64: 图片base64编码（保留参数以保持接口一致性，但实际不使用）
            pipeline_config: Pipeline配置（保留参数以保持接口一致性，但实际不使用）
            async_client: 异步客户端实例（保留参数以保持接口一致性，但实际不使用）
            model: 模型名称（保留参数以保持接口一致性，但实际不使用）
            
        Returns:
            预填充信息字典
        """
        # 同步处理即可
        return self.process_prefill(
            prefill_input=prefill_input,
            image_input=None,
            pipeline_config=pipeline_config
        )
