"""评测指标：头指标、特征探针、特征漂移、结果指标。

约定（排除 E4 测量伪影）：
- 探针用 train 划分拟合、test 划分评估；
- 探针为岭回归闭式解（GPU），带特征标准化与 λ 正则；
- 置换检验：打乱 train 标签重拟合，期望接近随机水平。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- 特征提取
@torch.no_grad()
def extract_features(model, x: torch.Tensor) -> torch.Tensor:
    return model.features(x)


def batched(fn, n: int, batch: int = 2048):
    """分批执行 fn(i0:i1)，返回拼接张量。"""
    outs = []
    for i in range(0, n, batch):
        outs.append(fn(i, min(i + batch, n)))
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------- 头指标
@torch.no_grad()
def head_forward(model, task, device, split: str = "test", bs: int = 1024):
    """返回 (logits, labels)，分批前向。"""
    model.eval()
    x = task.x_test if split == "test" else task.x_train
    y = task.y_test if split == "test" else task.y_train
    logits, labels = [], []
    for i in range(0, len(y), bs):
        xb = task.prepare(x[i:i + bs], device, augment=False)
        logits.append(model.head(model.features(xb).flatten(1)))
        labels.append(y[i:i + bs].to(device))
    return torch.cat(logits), torch.cat(labels)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(1) == labels).float().mean().item())


def head_accuracy(model, task, device, split: str = "test") -> float:
    logits, labels = head_forward(model, task, device, split)
    return accuracy(logits, labels)


def margin(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """真实类 logit 与最强竞争 logit 之差的均值。"""
    true = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, labels.view(-1, 1), False)
    other = logits.masked_fill(~mask, float("-inf")).max(1).values
    return float((true - other).mean().item())


def head_weight_norm(model) -> float:
    return float(model.head.weight.norm(p=2).item())


def nll(model, task, device, split: str = "test", bs: int = 1024) -> float:
    logits, labels = head_forward(model, task, device, split, bs)
    return float(F.cross_entropy(logits, labels).item())


# ---------------------------------------------------------------- 线性探针（岭回归）
def ridge_probe_acc(model, task, device, lam: float = 1e-1,
                    max_train: int | None = None, shuffle_labels: bool = False,
                    bs: int = 2048) -> float:
    """特征 -> 线性探针准确率（train 拟合 / test 评估）。

    max_train: 为控制评测耗时可对 probe 训练集做固定子采样（在 CPU 上按 seed 决定，run 内不变）。
    """
    model.eval()
    # --- train 特征
    idx = torch.arange(task.n_train)
    if max_train is not None and task.n_train > max_train:
        g = torch.Generator().manual_seed(12345)
        idx = idx[torch.randperm(task.n_train, generator=g)[:max_train]]
    tr_x = task.x_train[idx]
    tr_y = task.y_train[idx]
    feats_tr = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            xb = task.prepare(tr_x[i:i + bs], device, augment=False)
            feats_tr.append(model.features(xb).flatten(1))
    Ftr = torch.cat(feats_tr)
    ytr = tr_y.to(device)
    # --- test 特征
    feats_te = []
    with torch.no_grad():
        for i in range(0, task.n_test, bs):
            xb = task.prepare(task.x_test[i:i + bs], device, augment=False)
            feats_te.append(model.features(xb).flatten(1))
    Fte = torch.cat(feats_te)
    yte = task.y_test.to(device)
    return float(_ridge_acc(Ftr, ytr, Fte, yte, lam, shuffle_labels))


def _ridge_acc(Ftr, ytr, Fte, yte, lam, shuffle_labels=False) -> float:
    n, d = Ftr.shape
    C = int(yte.max().item()) + 1
    C = max(C, int(ytr.max().item()) + 1, 2)
    mu = Ftr.mean(0, keepdim=True)
    sd = Ftr.std(0, keepdim=True) + 1e-6
    A = (Ftr - mu) / sd
    B = (Fte - mu) / sd
    if shuffle_labels:
        g = torch.Generator(device=Ftr.device).manual_seed(20240101)
        ytr = ytr[torch.randperm(n, generator=g, device=Ftr.device)]
    Y = F.one_hot(ytr, C).float()
    G = A.T @ A
    G.diagonal().add_(lam * n)
    W = torch.linalg.solve(G, A.T @ Y)
    return float((B @ W).argmax(1).eq(yte).float().mean().item())


def probe_acc(model, task, device, **kw) -> float:
    return ridge_probe_acc(model, task, device, **kw)


# ---------------------------------------------------------------- 特征漂移
def feature_drift(model, task, device, reference: torch.Tensor, bs: int = 2048) -> float:
    """当前特征与 end_A 参考特征在同一批输入上的平均余弦距离。"""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(task.y_test), bs):
            xb = task.prepare(task.x_test[i:i + bs], device, augment=False)
            outs.append(model.features(xb).flatten(1))
    Fnow = torch.cat(outs)
    a = Fnow.flatten(1)
    b = reference.flatten(1)
    cos = F.cosine_similarity(a, b, dim=1)
    return float((1.0 - cos).mean().item())


@torch.no_grad()
def reference_features(model, task, device, bs: int = 2048) -> torch.Tensor:
    """end_A 参考特征（旧任务 test 划分，无增强）。"""
    model.eval()
    outs = []
    for i in range(0, len(task.y_test), bs):
        xb = task.prepare(task.x_test[i:i + bs], device, augment=False)
        outs.append(model.features(xb).flatten(1))
    return torch.cat(outs)


def is_finite(*vals) -> bool:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return False
    return True
