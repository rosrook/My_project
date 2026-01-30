"""
筛选要素抽象：自然语言筛选要素 → 向量表示 → 相似度图 → 离散筛选原语（factor_id + 可解释映射）。
"""

from .text_encoder import TextEncoder
from .similarity_graph import build_similarity_graph, connected_components_from_graph
from .factor_primitives import build_factor_primitives, run_abstraction_pipeline

__all__ = [
    "TextEncoder",
    "build_similarity_graph",
    "connected_components_from_graph",
    "build_factor_primitives",
    "run_abstraction_pipeline",
]
