"""
因子预测推理：给定图片（路径或张量）+ 抽象结果或 checkpoint，返回 factor_id 列表。
建议在项目根目录下运行或使用：from DecisionModel.inference import predict_factor_ids, load_model_from_checkpoint
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Union

# 作为脚本运行时确保可导入 DecisionModel
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import torch
    from PIL import Image
    from torchvision import transforms
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def load_image_tensor(image_path: str, image_size: tuple = (224, 224)) -> "torch.Tensor":
    """将单张图片路径转为 (1, 3, H, W) 的归一化张量。"""
    if not HAS_DEPS:
        raise ImportError("torch, PIL, torchvision required for load_image_tensor")
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size)
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = T(img).unsqueeze(0)
    return x


def predict_factor_ids(
    model: "torch.nn.Module",
    image_input: Union[str, "torch.Tensor"],
    factor_ids_ordered: List[str],
    threshold: float = 0.5,
    device: str = "cpu",
) -> List[str]:
    """
    对单张图或单 batch 张量预测 factor_id 列表（sigmoid > threshold 的 id）。
    - image_input: 图片路径或 (B, 3, H, W) 张量；若为路径则返回单张图的 id 列表，若为 batch 则返回 list of list。
    """
    if not HAS_DEPS:
        raise ImportError("torch required for predict_factor_ids")
    model.eval()
    if isinstance(image_input, str):
        x = load_image_tensor(image_input).to(device)
        single = True
    else:
        x = image_input.to(device)
        single = x.dim() == 4 and x.size(0) == 1
    with torch.no_grad():
        probs = model.predict_probs(x)
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    pred_ids_list: List[List[str]] = []
    for i in range(probs.size(0)):
        pred_idx = (probs[i] > threshold).nonzero(as_tuple=True)[0].cpu().tolist()
        pred_ids_list.append([factor_ids_ordered[j] for j in pred_idx])
    if single and len(pred_ids_list) == 1:
        return pred_ids_list[0]
    return pred_ids_list


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str = "cpu",
) -> tuple["torch.nn.Module", List[str]]:
    """
    从 run_demo 保存的 checkpoint 加载模型与 factor_ids_ordered。
    返回 (model, factor_ids_ordered)；model 需由调用方与 VisionEncoder + FactorPredictionModel 结构一致并 load_state_dict。
    """
    if not HAS_DEPS:
        raise ImportError("torch required for load_model_from_checkpoint")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    factor_ids_ordered = ckpt["factor_ids_ordered"]
    num_factors = ckpt["num_factors"]
    # 重建模型结构（与 run_demo 一致）
    from DecisionModel.factor_prediction import VisionEncoder, FactorPredictionModel
    from DecisionModel.config import NUM_FROZEN_LAYERS, FACTOR_HEAD_HIDDEN_DIM, FACTOR_HEAD_DROPOUT
    vision_encoder = VisionEncoder(
        backbone_name="resnet18",
        pretrained=False,
        num_frozen_layers=NUM_FROZEN_LAYERS,
    )
    model = FactorPredictionModel(
        vision_encoder=vision_encoder,
        num_factors=num_factors,
        hidden_dim=FACTOR_HEAD_HIDDEN_DIM,
        dropout=FACTOR_HEAD_DROPOUT,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    return model, factor_ids_ordered
