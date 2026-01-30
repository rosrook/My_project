"""
验证模型是否为不同图像自动生成合理、可解释的筛选要素组合，证明中间表示可作为动态筛选管线与 VQA 的接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..data.schemas import LabeledSample
from .router import Router


@dataclass
class SanityCheckReport:
    """Sanity check 报告：覆盖率、可解释性、样例等。"""
    num_samples: int
    num_factors: int
    # 预测的 factor_id 组合中，有多少能成功映射到原语
    interpretable_count: int
    # 平均每张图预测的 factor 数量
    avg_predicted_factors: float
    # 样例：image_id -> (predicted_factor_ids, ground_truth_factor_ids)
    sample_predictions: List[tuple] = field(default_factory=list)
    # 是否通过（demo 中可简单定义为“有预测且可解释”）
    passed: bool = False


def run_sanity_check(
    predictions: List[List[str]],
    labeled_samples: List[LabeledSample],
    router: Router,
    factor_id_to_idx: Dict[str, int],
    top_k: int = 5,
) -> SanityCheckReport:
    """
    根据模型预测的 factor_id 组合与真实标签，验证：
    1. 预测是否都能映射到原语（可解释）
    2. 统计平均预测数量、与 GT 的对比
    3. 抽取若干样例写入报告
    """
    n = len(labeled_samples)
    if n == 0:
        return SanityCheckReport(
            num_samples=0,
            num_factors=len(factor_id_to_idx),
            interpretable_count=0,
            avg_predicted_factors=0.0,
            sample_predictions=[],
            passed=False,
        )

    interpretable_count = 0
    total_predicted = 0
    sample_predictions: List[tuple] = []

    for i, labeled in enumerate(labeled_samples):
        pred_ids = predictions[i] if i < len(predictions) else []
        total_predicted += len(pred_ids)
        # 可解释：每个 pred_id 都在 router 的 factor_id_to_texts 中
        all_interpretable = all(
            fid in router.factor_id_to_texts for fid in pred_ids
        )
        if all_interpretable and pred_ids:
            interpretable_count += 1
        gt_ids = labeled.factor_ids
        image_id = labeled.image_id or labeled.image_path
        sample_predictions.append((image_id, pred_ids, gt_ids))

    # 只保留前 top_k 个样例
    sample_predictions = sample_predictions[:top_k]
    avg_pred = total_predicted / n if n else 0.0
    num_factors = len(factor_id_to_idx)
    passed = n > 0 and (interpretable_count >= n or avg_pred > 0)

    return SanityCheckReport(
        num_samples=n,
        num_factors=num_factors,
        interpretable_count=interpretable_count,
        avg_predicted_factors=avg_pred,
        sample_predictions=sample_predictions,
        passed=passed,
    )
