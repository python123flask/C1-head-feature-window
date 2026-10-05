"""模型定义：可拆分为特征提取器 φ 与分类头 (W2, b2) 的网络。

所有模型遵守同一接口：
    model.features(x) -> phi        # 特征提取器 φ
    model.head(phi)    -> logits     # 分类头 (W2, b2)
    model(x)           -> logits
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """主实验 CNN（32×32 RGB 输入；width_mult=1 时恰为 113,738 参数）。

    width_mult 用于档 C 的宽度敏感性（P16）：改变特征通道数，
    检验“特征损伤幅度随特征图冗余度变化”这一预测。
    """

    feat_dim = 2048          # width_mult=1.0 的默认值；__init__ 会用实例属性覆盖

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        c1, c2, c3 = (max(8, int(round(w * float(width_mult)))) for w in (32, 64, 128))
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1),   # 32x32xc1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 16x16xc1
            nn.Conv2d(c1, c2, 3, padding=1),  # 16x16xc2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 8x8xc2
            nn.Conv2d(c2, c3, 3, padding=1),  # 8x8xc3
            nn.ReLU(inplace=True),
            # 32x32 输入下恰为 8x8 -> 4x4：与 AdaptiveAvgPool2d(4) 前向**逐位等价**，
            # 但其反向不走 adaptive_avg_pool2d_backward_cuda（原子累加、非确定），
            # 因此可支持 P10 的确定性核复现。
            nn.AvgPool2d(2, 2),                # 4x4xc3
            nn.Flatten(),                      # 16*c3
        )
        self.head = nn.Linear(c3 * 16, num_classes)
        self.feat_dim = c3 * 16
        self.width_mult = float(width_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ResNet18Split(nn.Module):
    """ResNet-18（扩展实验）：features = 除 fc 外所有层，head = fc。"""

    feat_dim = 512

    def __init__(self, num_classes: int = 10):
        super().__init__()
        import torchvision

        net = torchvision.models.resnet18(weights=None)
        self.features = nn.Sequential(*list(net.children())[:-1])  # 含 avgpool
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x).flatten(1)
        return self.head(f)


class SmallMLP(nn.Module):
    """传感器数据用 MLP（约 100K 参数）。"""

    feat_dim = 256

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(spec: str, num_classes: int, width_mult: float = 1.0) -> nn.Module:
    if spec == "smallcnn":
        return SmallCNN(num_classes, width_mult=width_mult)
    if spec == "resnet18":
        return ResNet18Split(num_classes)
    if spec == "smallmlp":
        return SmallMLP(num_classes=num_classes)
    raise ValueError(f"unknown model spec: {spec}")


def reinit_head(model: nn.Module) -> None:
    """按 PyTorch 默认初始化重置分类头（用于 abrupt / headonly 对照）。"""
    model.head.reset_parameters()


def freeze_features(model: nn.Module) -> None:
    for p in model.features.parameters():
        p.requires_grad_(False)


def unfreeze_features(model: nn.Module) -> None:
    for p in model.features.parameters():
        p.requires_grad_(True)


def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def count_params(model: nn.Module) -> dict:
    n_feat = sum(p.numel() for p in model.features.parameters())
    n_head = sum(p.numel() for p in model.head.parameters())
    return {"features": n_feat, "head": n_head, "total": n_feat + n_head}
