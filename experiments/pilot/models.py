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
    """约 2.1M 参数的主实验 CNN（32×32 RGB 输入）。"""

    feat_dim = 2048

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # 32x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 16x16x32
            nn.Conv2d(32, 64, 3, padding=1),  # 16x16x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 8x8x64
            nn.Conv2d(64, 128, 3, padding=1),  # 8x8x128
            nn.ReLU(inplace=True),
            # 32x32 输入下恰为 8x8 -> 4x4：与 AdaptiveAvgPool2d(4) 前向**逐位等价**，
            # 但其反向不走 adaptive_avg_pool2d_backward_cuda（原子累加、非确定），
            # 因此可支持 P10 的确定性核复现。
            nn.AvgPool2d(2, 2),                # 4x4x128
            nn.Flatten(),                      # 2048
        )
        self.head = nn.Linear(2048, num_classes)

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


def build_model(spec: str, num_classes: int) -> nn.Module:
    if spec == "smallcnn":
        return SmallCNN(num_classes)
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
