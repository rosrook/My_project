"""
配置文件加载模块
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, config_path: Path):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config
    
    def get_pipeline_config(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定pipeline的配置
        
        Args:
            pipeline_name: Pipeline名称
            
        Returns:
            Pipeline配置字典，如果不存在返回None
        """
        pipelines = self.config.get("pipelines", {})
        return pipelines.get(pipeline_name)
    
    def get_global_constraints(self) -> Dict[str, Any]:
        """获取全局约束"""
        return self.config.get("global_constraints", {})
    
    def get_object_selection_policy(self) -> Dict[str, Any]:
        """获取对象选择策略"""
        return self.config.get("object_selection_policy", {})
    
    def get_generation_policy(self) -> Dict[str, Any]:
        """获取生成策略"""
        return self.config.get("generation_policy", {})

    def get_question_type_ratio(self) -> Dict[str, float]:
        """
        获取题型比例配置
        
        Returns:
            包含multiple_choice和fill_in_blank比例的字典，默认7:3
        """
        generation_policy = self.get_generation_policy()
        question_type_ratio = generation_policy.get("question_type_ratio", {})
        
        # 默认比例：选择题70%，填空题30%
        default_ratio = {
            "multiple_choice": 0.7,
            "fill_in_blank": 0.3
        }
        
        # 合并用户配置和默认配置
        ratio = {
            "multiple_choice": question_type_ratio.get("multiple_choice", default_ratio["multiple_choice"]),
            "fill_in_blank": question_type_ratio.get("fill_in_blank", default_ratio["fill_in_blank"])
        }
        
        # 归一化比例（确保总和为1）
        total = ratio["multiple_choice"] + ratio["fill_in_blank"]
        if total > 0:
            ratio["multiple_choice"] /= total
            ratio["fill_in_blank"] /= total
        
        return ratio
    
    def list_pipelines(self) -> list:
        """列出所有可用的pipeline名称"""
        return list(self.config.get("pipelines", {}).keys())

