"""
将语义簇映射为 factor_id，并保留 factor_id → 原始文本列表 的可解释映射。
"""

from __future__ import annotations

from typing import Dict, List

from ..data.schemas import AbstractionResult, FactorPrimitive

from .text_encoder import TextEncoder
from .similarity_graph import clusters_from_embeddings


def build_factor_primitives(
    unique_texts: List[str],
    clusters: List[List[int]],
    factor_id_prefix: str = "factor_",
) -> AbstractionResult:
    """
    根据簇（每个簇为 unique_texts 的索引列表）构建 FactorPrimitive 列表，
    以及 text → factor_id、factor_id → [texts] 的映射。
    """
    idx_to_text = {i: unique_texts[i] for i in range(len(unique_texts))}
    primitives: List[FactorPrimitive] = []
    text_to_factor_id: Dict[str, str] = {}
    factor_id_to_texts: Dict[str, List[str]] = {}

    for k, indices in enumerate(clusters):
        factor_id = f"{factor_id_prefix}{k}"
        original_texts = [idx_to_text[i] for i in indices]
        primitives.append(FactorPrimitive(factor_id=factor_id, original_texts=original_texts))
        factor_id_to_texts[factor_id] = original_texts
        for t in original_texts:
            text_to_factor_id[t] = factor_id

    return AbstractionResult(
        primitives=primitives,
        text_to_factor_id=text_to_factor_id,
        factor_id_to_texts=factor_id_to_texts,
    )


def run_abstraction_pipeline(
    all_texts: List[str],
    text_encoder: TextEncoder,
    similarity_threshold: float = 0.75,
    factor_id_prefix: str = "factor_",
) -> AbstractionResult:
    """
    完整抽象流程：
    1. 去重得到 unique_texts
    2. 用 text_encoder 编码
    3. 用 threshold 做相似度图 + 连通分量得到簇
    4. 构建 FactorPrimitive 与映射
    """
    unique_texts = list(dict.fromkeys(all_texts))
    if not unique_texts:
        return AbstractionResult(primitives=[], text_to_factor_id={}, factor_id_to_texts={})

    embeddings = text_encoder.encode(unique_texts)
    clusters = clusters_from_embeddings(embeddings, unique_texts, similarity_threshold)
    return build_factor_primitives(unique_texts, clusters, factor_id_prefix)
