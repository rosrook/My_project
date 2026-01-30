"""
输入/输出数据结构：images + 自然语言筛选要素集合。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# 单条样本：一张图 + 其关联的筛选要素文本集合
@dataclass
class ImageFilterSample:
    image_path: str
    filter_factor_texts: List[str]
    image_id: Optional[str] = None

# 抽象后的原语：factor_id → 原始文本列表（可解释性）
@dataclass
class FactorPrimitive:
    factor_id: str
    original_texts: List[str]

# 全局抽象结果：所有原语 + 文本→factor_id 映射
@dataclass
class AbstractionResult:
    primitives: List[FactorPrimitive]
    text_to_factor_id: Dict[str, str]
    factor_id_to_texts: Dict[str, List[str]]

# 带 multi-hot 标签的样本（用于训练）
@dataclass
class LabeledSample:
    image_path: str
    factor_ids: List[str]
    multi_hot: List[int]
    image_id: Optional[str] = None

# 路由用：factor_id → 可调用的原语执行逻辑描述（demo 中可模拟）
@dataclass
class PrimitiveSpec:
    factor_id: str
    description: str
    original_texts: List[str]

