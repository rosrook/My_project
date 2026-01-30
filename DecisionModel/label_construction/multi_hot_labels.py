"""
将原始筛选要素文本集合映射为 factor_id 集合，并编码为 multi-hot 向量作为监督信号。
"""

from __future__ import annotations

from typing import Dict, List

from ..data.schemas import AbstractionResult, ImageFilterSample, LabeledSample


def texts_to_factor_ids(
    filter_factor_texts: List[str],
    text_to_factor_id: Dict[str, str],
) -> List[str]:
    """
    将原始筛选要素文本列表映射为 factor_id 列表（去重）。
    未在 text_to_factor_id 中出现的文本会被忽略。
    """
    factor_ids = []
    seen = set()
    for t in filter_factor_texts:
        fid = text_to_factor_id.get(t)
        if fid is not None and fid not in seen:
            factor_ids.append(fid)
            seen.add(fid)
    return factor_ids


def factor_ids_to_multi_hot(
    factor_ids: List[str],
    factor_id_to_idx: Dict[str, int],
    num_factors: int,
) -> List[int]:
    """
    将 factor_id 列表编码为 multi-hot 向量（长度为 num_factors）。
    """
    multi_hot = [0] * num_factors
    for fid in factor_ids:
        idx = factor_id_to_idx.get(fid)
        if idx is not None:
            multi_hot[idx] = 1
    return multi_hot


def build_labeled_samples(
    samples: List[ImageFilterSample],
    abstraction: AbstractionResult,
) -> List[LabeledSample]:
    """
    将 ImageFilterSample 列表（每张图 + 原始筛选要素文本）转为 LabeledSample 列表
    （每张图 + factor_ids + multi-hot 向量）。
    """
    factor_ids_ordered = [p.factor_id for p in abstraction.primitives]
    factor_id_to_idx = {fid: i for i, fid in enumerate(factor_ids_ordered)}
    num_factors = len(factor_ids_ordered)

    labeled: List[LabeledSample] = []
    for s in samples:
        factor_ids = texts_to_factor_ids(s.filter_factor_texts, abstraction.text_to_factor_id)
        multi_hot = factor_ids_to_multi_hot(factor_ids, factor_id_to_idx, num_factors)
        labeled.append(
            LabeledSample(
                image_path=s.image_path,
                factor_ids=factor_ids,
                multi_hot=multi_hot,
                image_id=s.image_id,
            )
        )
    return labeled
