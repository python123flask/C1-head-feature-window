"""诊断：Phase A 各类测试精度（判断是否为任务本身难度/数据问题）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "pilot"))

import metrics as M                      # noqa: E402
from data_loader import TrainStream, make_cifar_tasks  # noqa: E402
from models import build_model            # noqa: E402
from utils import set_seed                # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else "smallcnn"
    set_seed(42)
    old, _ = make_cifar_tasks(seed=0, head_dim=10)
    model = build_model(spec, 10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    stream = TrainStream(old, 128, seed=42 * 31 + 1, augment=True)
    for _ in range(5000):
        xb, yb = stream.next(DEVICE)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
        opt.step()
    model.eval()
    logits, labels = M.head_forward(model, old, DEVICE)
    pred = logits.argmax(1)
    print(f"{spec}: overall test acc = {M.accuracy(logits, labels):.3f}")
    for c in range(5):
        m = labels == c
        acc = float((pred[m] == labels[m]).float().mean())
        print(f"  class {c} {CIFAR_NAMES[c]:10s} acc={acc:.3f} n={int(m.sum())}")
    # 训练集精度（无增强）
    tl, tyl = M.head_forward(model, old, DEVICE, split="train")
    tp = tl.argmax(1)
    print("  train per-class:", [f"{float((tp[tys == c] == c).float().mean()):.3f}"
                                 for c in range(5) for tys in [tyl]])


if __name__ == "__main__":
    main()
