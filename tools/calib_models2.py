"""Phase A 校准 v2：优化器/正则化/步数 变体，目标 end_A 测试精度 >= 0.85。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "pilot"))

import metrics as M                    # noqa: E402
from data_loader import TrainStream, make_cifar_tasks  # noqa: E402
from models import build_model          # noqa: E402
from utils import set_seed              # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(model_spec: str, opt_kind: str = "adam", steps: int = 5000,
        lr: float = 1e-3, wd: float = 1e-4, seed: int = 42, batch: int = 128,
        extra: str = ""):
    set_seed(seed)
    old, _ = make_cifar_tasks(seed=0, head_dim=10)
    model = build_model(model_spec, 10).to(DEVICE)
    if opt_kind == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    elif opt_kind == "adamconst":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        sched = None
    elif opt_kind == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    else:
        raise ValueError(opt_kind)
    stream = TrainStream(old, batch, seed=seed * 31 + 1, augment=True)
    for step in range(steps):
        xb, yb = stream.next(DEVICE)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        if sched:
            sched.step()
    model.eval()
    logits, labels = M.head_forward(model, old, DEVICE)
    acc = M.accuracy(logits, labels)
    tl, tyl = M.head_forward(model, old, DEVICE, split="train")
    tr_acc = float((tl.argmax(1) == tyl).float().mean())
    probe = M.ridge_probe_acc(model, old, DEVICE, max_train=2048)
    print(f"{model_spec:9s} {opt_kind:9s} steps={steps} lr={lr} wd={wd} {extra} | "
          f"test={acc:.3f} train={tr_acc:.3f} probe={probe:.3f}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {
        "a": [("smallcnn", "adam", 5000, 1e-3, 1e-4, "baseline"),
              ("smallcnn", "adam", 10000, 1e-3, 1e-4, "2x steps"),
              ("smallcnn", "adam", 5000, 1e-3, 1e-3, "wd1e-3"),
              ("smallcnn", "sgd", 5000, 0.1, 5e-4, "sgd+cos")],
        "b": [("resnet18", "adam", 10000, 1e-3, 1e-4, "2x steps"),
              ("resnet18", "adam", 5000, 1e-3, 1e-3, "wd1e-3"),
              ("resnet18", "sgd", 5000, 0.1, 5e-4, "sgd+cos"),
              ("resnet18", "sgd", 10000, 0.1, 5e-4, "sgd+cos 2x")],
    }
    for spec, kind, steps, lr, wd, note in jobs.get(which, jobs["a"]):
        run(spec, kind, steps, lr, wd, extra=note)
