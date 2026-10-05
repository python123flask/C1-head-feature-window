"""Phase A 质量诊断：模型 × 增强 × 头槽位对 end_A 精度/探针的影响。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "pilot"))

import metrics as M          # noqa: E402
import sim                   # noqa: E402
from data_loader import TrainStream, make_cifar_tasks  # noqa: E402
from models import build_model  # noqa: E402
from utils import set_seed     # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(model_spec: str, steps: int = 5000, lr: float = 1e-3, aug: bool = True,
        seed: int = 42, batch: int = 128, decay: bool = False):
    set_seed(seed)
    old, new = make_cifar_tasks(seed=0, head_dim=10)
    model = build_model(model_spec, 10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps) if decay else None
    stream = TrainStream(old, batch, seed=seed * 31 + 1, augment=aug)
    losses = []
    for step in range(steps):
        xb, yb = stream.next(DEVICE)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        losses.append(float(loss))
    model.eval()
    logits, labels = M.head_forward(model, old, DEVICE)
    acc_all = M.accuracy(logits, labels)
    acc_slots = float((logits[:, :5].argmax(1) == labels).float().mean())
    train_logits, train_labels = M.head_forward(model, old, DEVICE, split="train")
    train_acc = float((train_logits[:, :5].argmax(1) == train_labels).float().mean())
    probe = M.ridge_probe_acc(model, old, DEVICE, max_train=2048)
    unused_win = float((logits.argmax(1) >= 5).float().mean())
    print(f"{model_spec:9s} steps={steps} lr={lr} aug={aug} decay={decay} | "
          f"acc_all={acc_all:.3f} acc_slots={acc_slots:.3f} train={train_acc:.3f} "
          f"probe={probe:.3f} unused_slot_win={unused_win:.3f} "
          f"loss0={losses[0]:.3f} lossN={np.mean(losses[-20:]):.3f}", flush=True)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    jobs = [("smallcnn", 5000, 1e-3, True, False),
            ("smallcnn", 5000, 1e-3, False, False),
            ("smallcnn", 5000, 1e-3, True, True),
            ("resnet18", 5000, 1e-3, True, False),
            ("resnet18", 5000, 1e-3, True, True)]
    for spec, steps, lr, aug, decay in jobs:
        if only and only not in spec:
            continue
        run(spec, steps, lr, aug, decay=decay)
