"""
冻结或部分冻结的视觉编码器：提取图像特征供因子预测头使用。
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


class VisionEncoder(nn.Module if HAS_TORCH else object):  # type: ignore[misc]
    """
    基于 torchvision ResNet 的视觉编码器，可冻结前若干层。
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        num_frozen_layers: int = 3,
        output_dim: Optional[int] = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("torch and torchvision are required for VisionEncoder")
        super().__init__()
        from torchvision.models import resnet18, ResNet
        from torchvision.models.resnet import BasicBlock

        if backbone_name == "resnet18":
            backbone = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        else:
            backbone = resnet18(weights=None)

        # 去掉最后的 fc
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        # ResNet18 最后一层输出 512
        self._feature_dim = 512
        self._num_frozen_layers = num_frozen_layers
        self._freeze_layers(num_frozen_layers)

        self._output_dim = output_dim or self._feature_dim
        if output_dim is not None and output_dim != self._feature_dim:
            self.proj = nn.Linear(self._feature_dim, output_dim)
        else:
            self.proj = nn.Identity()

    def _freeze_layers(self, num_layers: int) -> None:
        if num_layers <= 0:
            return
        # 按子模块顺序冻结前 num_layers 个
        children = list(self.backbone.children())
        for i, child in enumerate(children):
            if i >= num_layers:
                break
            for p in child.parameters():
                p.requires_grad = False

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        x: (B, 3, H, W)
        out: (B, output_dim)
        """
        out = self.backbone(x)
        out = out.flatten(1)
        return self.proj(out)

    @property
    def feature_dim(self) -> int:
        return self._output_dim
