"""
基于 Qwen2-VL 的因子预测模型：用 Qwen2-VL 作为视觉骨干，接多标签头，训练数据仍为「图片 + 筛选要素 id 组合」。
与 ResNet+MLP 方案相同监督方式（BCE），仅骨干替换为 Qwen2-VL。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


def _load_qwen2vl_and_processor(
    model_name: str,
    device: str = "cpu",
    torch_dtype: Optional["torch.dtype"] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    attn_implementation: str = "eager",  # "flash_attention_2" 需额外安装
) -> tuple[Any, Any]:
    """加载 Qwen2-VL 模型与 processor，返回 (model, processor)。"""
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device == "cuda" else None,
        attn_implementation=attn_implementation,
    )
    if device == "cpu":
        model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    if min_pixels is not None and hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = min_pixels
    if max_pixels is not None and hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = max_pixels
    return model, processor


class Qwen2VLFactorModel(nn.Module if HAS_TORCH else object):  # type: ignore[misc]
    """
    Qwen2-VL 骨干 + 多标签头：输入为 processor 产出的 pixel_values、image_grid_thw，
    通过 get_image_features 得到 (B, hidden_size)，再经线性层得到 (B, num_factors) logits。
    训练目标仍为 BCE 多标签损失，数据格式与 run_demo 一致（图片 + factor_id / multi_hot）。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        num_factors: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        device: str = "cpu",
        torch_dtype: Optional["torch.dtype"] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("torch and transformers are required for Qwen2VLFactorModel")
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self._processor = None
        model, processor = _load_qwen2vl_and_processor(
            model_name,
            device=device,
            torch_dtype=torch_dtype,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.backbone = model
        self._processor = processor
        # Qwen2-VL 7B hidden_size=3584, 2B 等可能不同
        hidden_size = getattr(
            self.backbone.config,
            "hidden_size",
            getattr(self.backbone.config.text_config, "hidden_size", 3584),
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.factor_head = nn.Linear(hidden_dim, num_factors)

    @property
    def processor(self):
        return self._processor

    def forward(
        self,
        pixel_values: "torch.Tensor",
        image_grid_thw: "torch.Tensor",
        **kwargs: Any,
    ) -> Dict[str, "torch.Tensor"]:
        """
        pixel_values / image_grid_thw 由 processor(images=..., text=...) 得到。
        返回 {"factor_logits": (B, num_factors)}。
        """
        if self.backbone is None:
            raise RuntimeError("backbone not loaded")
        # 使用 get_image_features（Qwen2VLForConditionalGeneration 或内部 .model 上）
        backbone = self.backbone
        if hasattr(backbone, "get_image_features"):
            out = backbone.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        elif hasattr(backbone, "model") and hasattr(backbone.model, "get_image_features"):
            out = backbone.model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        else:
            # 兼容：用 model forward 取 last_hidden_state 最后一维
            inner = getattr(backbone, "model", backbone)
            out = inner(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
            feats = out.last_hidden_state[:, -1, :]
            h = self.head(feats)
            logits = self.factor_head(h)
            return {"factor_logits": logits}
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        else:
            feats = out.last_hidden_state[:, 0, :]
        h = self.head(feats)
        logits = self.factor_head(h)
        return {"factor_logits": logits}

    def predict_probs(
        self,
        pixel_values: "torch.Tensor",
        image_grid_thw: "torch.Tensor",
        **kwargs: Any,
    ) -> "torch.Tensor":
        """返回 (B, num_factors) 的 sigmoid 概率。"""
        out = self.forward(pixel_values, image_grid_thw, **kwargs)
        return torch.sigmoid(out["factor_logits"])


def build_qwen2vl_collate_fn(processor: Any, prompt_text: str = "What filter factors apply to this image?"):
    """
    返回一个 collate_fn，将 list[(image_path, multi_hot)] 转为
    dict(pixel_values, image_grid_thw, labels)，供 Qwen2VLFactorModel 与 DataLoader 使用。
    使用 processor.apply_chat_template 以符合 Qwen2-VL 的对话格式。
    """

    def collate_fn(batch: List[tuple]) -> Dict[str, "torch.Tensor"]:
        from PIL import Image
        image_paths = [x[0] for x in batch]
        multi_hots = [x[1] for x in batch]
        images = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            images.append(img)
        texts = [prompt_text] * len(images)
        inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
        labels = torch.tensor(multi_hots, dtype=torch.float32)
        return {
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "labels": labels,
        }

    return collate_fn
