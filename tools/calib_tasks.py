"""任务构造校准：类别划分 × 训练样本量 × 模型 对 end_A 精度/探针的影响。

目标（A1 质控门）：end_A 测试精度中位 >= 0.85，且探针有足够余量（>= 0.80）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "pilot"))

import metrics as M                      # noqa: E402
from data_loader import TrainStream, make_cifar_tasks  # noqa: E402
from models import build_model            # noqa: E402
from utils import set_seed                # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EASY = [0, 1, 8, 9, 6]      # airplane, automobile, ship, truck, frog
HARD = [2, 3, 4, 5, 7]      # bird, cat, deer, dog, horse
ORIG = [0, 1, 2, 3, 4]


def run(tag, model_spec="smallcnn", old_classes=ORIG, per_class=1000,
        steps=5000, lr=1e-3, seed=42, batch=128):
    set_seed(seed)
    new_classes = [c for c in range(10) if c not in old_classes]
    old, new = make_cifar_tasks(seed=0, per_class_train=per_class, head_dim=10,
                                old_classes=old_classes, new_classes=new_classes)
    model = build_model(model_spec, 10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    stream = TrainStream(old, batch, seed=seed * 31 + 1, augment=True)
    for _ in range(steps):
        xb, yb = stream.next(DEVICE)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
        opt.step()
    acc = M.head_accuracy(model, old, DEVICE)
    probe = M.ridge_probe_acc(model, old, DEVICE, max_train=2048)
    # 新任务探针上限（特征可读出性参考）
    nprobe = M.ridge_probe_acc(model, new, DEVICE, max_train=2048)
    print(f"{tag:34s} model={model_spec:9s} per_class={per_class:5d} steps={steps:5d} | "
          f"endA_test={acc:.3f} probe_old={probe:.3f} probe_new={nprobe:.3f}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {
        "a": [("orig 1000/cls", dict(old_classes=ORIG, per_class=1000, steps=5000)),
              ("orig 5000/cls", dict(old_classes=ORIG, per_class=5000, steps=5000)),
              ("orig 5000/cls 10k", dict(old_classes=ORIG, per_class=5000, steps=10000)),
              ("easy 1000/cls", dict(old_classes=EASY, per_class=1000, steps=5000)),
              ("easy 5000/cls", dict(old_classes=EASY, per_class=5000, steps=5000)),
              ("hard 1000/cls", dict(old_classes=HARD, per_class=1000, steps=5000))],
        "b": [("easy 1000/cls resnet", dict(old_classes=EASY, per_class=1000, steps=5000)),
              ("easy 5000/cls resnet", dict(old_classes=EASY, per_class=5000, steps=5000)),
              ("easy 5000/cls resnet 10k", dict(old_classes=EASY, per_class=5000, steps=10000)),
              ("orig 5000/cls resnet", dict(old_classes=ORIG, per_class=5000, steps=5000))],
    }
    for tag, kw in jobs.get(which, jobs["a"]):
        model = kw.pop("model", "smallcnn") if False else ("resnet18" if "resnet" in tag else "smallcnn")
        run(tag, model_spec=model, **kw)
