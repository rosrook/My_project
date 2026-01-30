"""
多标签评估指标：用于训练/测试结果对比（ResNet+MLP vs Qwen2-VL）。
"""

from __future__ import annotations

from typing import Dict, List

try:
    import torch
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def predictions_to_multi_hot(
    predictions: List[List[str]],
    factor_ids_ordered: List[str],
) -> "torch.Tensor":
    """将预测的 factor_id 列表转为 (N, C) multi-hot 张量。"""
    if not HAS_DEPS:
        raise ImportError("torch required")
    fid2idx = {fid: i for i, fid in enumerate(factor_ids_ordered)}
    C = len(factor_ids_ordered)
    out = torch.zeros(len(predictions), C, dtype=torch.float32)
    for i, pred_ids in enumerate(predictions):
        for fid in pred_ids:
            if fid in fid2idx:
                out[i, fid2idx[fid]] = 1.0
    return out


def compute_multilabel_metrics(
    pred_multi_hot: "torch.Tensor",
    true_multi_hot: "torch.Tensor",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    计算多标签指标。
    pred_multi_hot / true_multi_hot: (N, C)，0/1 或概率。
    返回: bce_loss, subset_accuracy, f1_macro, f1_micro, precision_micro, recall_micro
    """
    if not HAS_DEPS:
        raise ImportError("torch required")
    pred = pred_multi_hot.float()
    true = true_multi_hot.float()
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        true = true.unsqueeze(0)
    N, C = pred.shape
    # BCE（对概率形式的 pred；若 pred 已是 0/1 则 BCE 等价于 01 误差的缩放）
    if pred.max() <= 1.0 and pred.min() >= 0.0:
        bce = torch.nn.functional.binary_cross_entropy(pred.clamp(1e-7, 1 - 1e-7), true, reduction="mean").item()
    else:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, true, reduction="mean").item()
    # 二值化
    pred_bin = (pred >= threshold).float()
    # Subset accuracy: 每条样本的 C 维完全一致才算对
    subset_correct = (pred_bin == true).all(dim=1).float().sum().item()
    subset_accuracy = subset_correct / N if N else 0.0
    # F1 macro: 对每个类别算 F1 再平均
    tp = (pred_bin * true).sum(dim=0)
    fp = (pred_bin * (1 - true)).sum(dim=0)
    fn = ((1 - pred_bin) * true).sum(dim=0)
    prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
    rec = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
    f1_per_class = torch.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-8), torch.zeros_like(prec))
    f1_macro = f1_per_class.mean().item()
    # Micro: 全局 TP/FP/FN
    tp_m = (pred_bin * true).sum().item()
    fp_m = (pred_bin * (1 - true)).sum().item()
    fn_m = ((1 - pred_bin) * true).sum().item()
    prec_micro = tp_m / (tp_m + fp_m + 1e-8)
    rec_micro = tp_m / (tp_m + fn_m + 1e-8)
    f1_micro = 2 * prec_micro * rec_micro / (prec_micro + rec_micro + 1e-8)
    return {
        "bce_loss": bce,
        "subset_accuracy": subset_accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_micro": prec_micro,
        "recall_micro": rec_micro,
    }
