"""
监督构造：将每张图片关联的原始筛选要素文本集合映射为 factor_id 集合，并编码为 multi-hot 向量。
"""

from .multi_hot_labels import (
    texts_to_factor_ids,
    factor_ids_to_multi_hot,
    build_labeled_samples,
)

__all__ = [
    "texts_to_factor_ids",
    "factor_ids_to_multi_hot",
    "build_labeled_samples",
]
