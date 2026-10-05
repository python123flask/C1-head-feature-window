"""C1 · 可塑性恢复窗口 —— 单次 run 主循环（GPU）。

协议（冻结计划 3.4 节 + pilot/DEVIATIONS.md 记录的必要适配）：
    Phase A: 旧任务 CE(ε=0) 训练 phase_A_steps 步
    Phase B: 旧任务 + 标签平滑(ε=eps_B) 暴露 L 步；若 sham_steps>0 先做等量 ε=0 步（等预算对照）
    Phase C: 新任务 CE(ε=0) 适应 phase_C_steps 步
评测：每 eval_interval 步记录头指标 / 特征探针 / 漂移 / 结果指标 -> metrics.jsonl
"""
from __future__ import annotations

import os

# 必须在 CUDA/cuBLAS 首次使用前设置（P10 确定性核复现需要）
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import metrics as M
from data_loader import TrainStream, make_cifar_tasks, make_new_task, make_sensor_tasks
from models import (build_model, count_params, freeze_features, reinit_head,
                    trainable_params, unfreeze_features)
from utils import (LOGS, RUNS, append_jsonl, append_log, gpu_peak_mb, load_checkpoint,
                   peak_rss_mb, read_json, reset_gpu_peak, save_checkpoint, set_seed,
                   write_json)


# ---------------------------------------------------------------- 评测子集
def make_idx(n: int, max_n, seed: int) -> np.ndarray:
    if max_n is None or n <= max_n:
        return np.arange(n)
    g = np.random.default_rng(seed)
    return np.sort(g.choice(n, size=max_n, replace=False))


def _feat_sets(model, task, device, idx_tr, idx_te, bs: int = 2048):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx_tr), bs):
            j = idx_tr[i:i + bs]
            xb = task.prepare(task.x_train[torch.as_tensor(j)], device, augment=False)
            outs.append(model.features(xb).flatten(1))
    Ftr = torch.cat(outs)
    ytr = task.y_train[torch.as_tensor(idx_tr)].to(device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx_te), bs):
            j = idx_te[i:i + bs]
            xb = task.prepare(task.x_test[torch.as_tensor(j)], device, augment=False)
            outs.append(model.features(xb).flatten(1))
    Fte = torch.cat(outs)
    yte = task.y_test[torch.as_tensor(idx_te)].to(device)
    return Ftr, ytr, Fte, yte


def _feat_on(model, task, device, idx, bs: int = 2048) -> torch.Tensor:
    """在固定索引子集上提取特征（与探针评测使用同一批输入）。"""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            xb = task.prepare(task.x_test[torch.as_tensor(j)], device, augment=False)
            outs.append(model.features(xb).flatten(1))
    return torch.cat(outs)


class EvalCtx:
    """一次 run 内固定的评测子集与参考特征。"""

    def __init__(self, config: dict, old_task, new_task):
        self.old_task = old_task
        self.new_task = new_task
        seed = int(config["seed"])
        self.old_tr = make_idx(old_task.n_train, config.get("probe_train_max"), seed + 11) if old_task else None
        self.old_te = make_idx(old_task.n_test, config.get("probe_test_max"), seed + 12) if old_task else None
        self.new_tr = make_idx(new_task.n_train, config.get("probe_train_max"), seed + 21) if new_task else None
        self.new_te = make_idx(new_task.n_test, config.get("probe_test_max"), seed + 22) if new_task else None
        self.ref: torch.Tensor | None = None
        self.perm_done = False
        self.old_probe = bool(config.get("old_probe_metrics", True))
        self.new_probe = bool(config.get("new_probe_metrics", True))
        self.lam = float(config.get("lam", 1e-1))


def evaluate(model, ctx: EvalCtx, device, phase: str, phase_step: int, global_step: int,
             do_permutation: bool = False) -> dict:
    rec = {"phase": phase, "phase_step": phase_step, "global_step": global_step,
           "W2_norm": M.head_weight_norm(model)}

    # ---------- 旧任务侧（头 + 特征探针 + 漂移）
    if ctx.old_task is not None and ctx.old_probe:
        logits, labels = M.head_forward(model, ctx.old_task, device)
        rec["acc_old_cur"] = M.accuracy(logits, labels)
        rec["margin_old"] = M.margin(logits, labels)
        Ftr, ytr, Fte, yte = _feat_sets(model, ctx.old_task, device, ctx.old_tr, ctx.old_te)
        rec["acc_old_probe"] = M._ridge_acc(Ftr, ytr, Fte, yte, ctx.lam)
        if do_permutation and not ctx.perm_done:
            rec["acc_old_probe_perm"] = M._ridge_acc(Ftr, ytr, Fte, yte, ctx.lam, shuffle_labels=True)
            ctx.perm_done = True
        if ctx.ref is not None:
            cos = F.cosine_similarity(Fte.flatten(1), ctx.ref.flatten(1), dim=1)
            rec["feat_drift"] = float((1.0 - cos).mean().item())
        # 无参考特征时（Phase A 全程、移植 run）不写 feat_drift —— 见评审 #6
    elif ctx.old_task is not None:
        logits, labels = M.head_forward(model, ctx.old_task, device)
        rec["acc_old_cur"] = M.accuracy(logits, labels)
        rec["margin_old"] = M.margin(logits, labels)

    # ---------- 新任务侧（Phase C）
    if phase == "C" and ctx.new_task is not None:
        logits, labels = M.head_forward(model, ctx.new_task, device)
        rec["new_task_acc"] = M.accuracy(logits, labels)
        rec["new_task_NLL"] = float(F.cross_entropy(logits, labels).item())
        if ctx.new_probe:
            Ftr, ytr, Fte, yte = _feat_sets(model, ctx.new_task, device, ctx.new_tr, ctx.new_te)
            rec["acc_new_probe"] = M._ridge_acc(Ftr, ytr, Fte, yte, ctx.lam)
    return rec


def _log(run_dir: Path, config: dict, rec: dict) -> None:
    rec = dict(rec)
    rec["run_id"] = config["run_id"]
    append_jsonl(run_dir / "metrics.jsonl", rec)


# ---------------------------------------------------------------- 训练步
def train_step(model, opt, xb, yb, eps: float) -> float:
    model.train()
    opt.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = F.cross_entropy(logits, yb, label_smoothing=eps)
    loss.backward()
    opt.step()
    return float(loss.item())


def _make_opt(model, config, freeze_feat: bool, lr: float | None = None):
    if freeze_feat:
        freeze_features(model)
    else:
        unfreeze_features(model)
    if lr is None:
        lr = float(config.get("lr", 1e-3))
    return torch.optim.Adam(trainable_params(model), lr=lr,
                            weight_decay=float(config.get("weight_decay", 1e-4)))


# ---------------------------------------------------------------- 主循环
def run_experiment(config: dict) -> dict:
    run_id = config["run_id"]
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").unlink(missing_ok=True)
    write_json(run_dir / "config.json", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("deterministic"):
        # 确定性核复现：压低跨进程重跑噪声（P10）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(int(config["seed"]))
    reset_gpu_peak()
    t0 = time.time()

    status, error = "succeeded", None
    summary: dict = {"run_id": run_id, "block": config.get("block"), "mode": config.get("mode"),
                     "seed": config.get("seed"), "L": config.get("L"),
                     "shared": config.get("shared"), "status": status}
    try:
        _run(config, run_dir, device, summary)
    except Exception:
        status = "failed"
        error = traceback.format_exc()
        append_log(LOGS / "batch.log", f"{run_id} FAILED\n{error}")
    wall = time.time() - t0

    summary.update({
        "status": status,
        "error": error,
        "wall_s": round(wall, 2),
        "rss_peak_mb": round(peak_rss_mb(), 1),
        "gpu_peak_mb": round(gpu_peak_mb(), 1),
        "device": str(device),
    })
    write_json(run_dir / "summary.json", summary)
    append_log(LOGS / "batch.log", f"{run_id} {status} wall={wall:.1f}s")
    return summary


def _cifar_pair(config: dict, data_seed: int):
    """按配置构造（旧任务, 高共享新任务）。dataset: 'cifar10'（默认）| 'cifar100'。"""
    if config.get("dataset", "cifar10") == "cifar100":
        from data_loader import make_cifar100_tasks
        return make_cifar100_tasks(
            seed=data_seed, head_dim=int(config.get("head_dim", 25)),
            per_class_train=int(config.get("per_class_train", 500)),
            per_class_test=int(config.get("per_class_test", 100)),
            old_classes=config.get("old_classes"),
            new_classes=config.get("new_classes"))
    return make_cifar_tasks(
        seed=data_seed, head_dim=int(config.get("head_dim", 10)),
        per_class_train=int(config.get("per_class_train", 1000)),
        old_classes=config.get("old_classes"),
        new_classes=config.get("new_classes"))


def _run(config: dict, run_dir: Path, device, summary: dict) -> None:
    mode = config.get("mode", "main")
    head_dim = int(config.get("head_dim", 10))
    seed = int(config["seed"])
    interval = int(config.get("eval_interval", 25))
    bs = int(config.get("batch_size", 128))
    eps_B = float(config.get("eps_B", 1.0))          # 冻结协议值（评审 #5）
    L = int(config.get("L", 0))
    sham = int(config.get("sham_steps", 0))
    A_steps = int(config.get("phase_A_steps", 5000))
    C_steps = int(config.get("phase_C_steps", 2000))
    save_ckpt = bool(config.get("save_ckpt", True))

    # ---------- 任务
    old_task = new_task = None
    data_seed = int(config.get("data_seed", 0))
    if mode == "sensor":
        old_task, new_task = make_sensor_tasks(
            seed=seed, head_dim=head_dim,
            batch_new=int(config.get("sensor_new_batch", 5)))
    elif mode == "transplant":
        new_task = make_new_task(config.get("shared", "high"), seed=data_seed,
                                 head_dim=head_dim, config=config)
        old_task, _ = _cifar_pair(config, data_seed)
    elif mode == "direct":
        new_task = make_new_task(config.get("shared", "high"), seed=data_seed,
                                 head_dim=head_dim, config=config)
        old_task = None
    else:
        old_task, new_hi = _cifar_pair(config, data_seed)
        new_task = new_hi if config.get("shared", "high") == "high" else make_new_task(
            "low", seed=data_seed, head_dim=head_dim, config=config)

    # ---------- 模型
    model_spec = config.get("model", "smallcnn" if mode != "sensor" else "smallmlp")
    model = build_model(model_spec, head_dim).to(device)
    summary["params"] = count_params(model)
    summary["tasks"] = {
        "old": old_task.describe() if old_task else None,
        "new": new_task.describe() if new_task else None,
    }

    ctx = EvalCtx(config, old_task, new_task)
    gstep = 0

    # ---------- 移植组合（P4）
    if mode == "transplant":
        import numpy as _np

        from utils import npz_to_state

        with torch.no_grad():
            # 1) 先装入“特征来源”的完整状态
            load_checkpoint(model, Path(config["feat_from_ckpt"]))
            # 2) 再用“头来源”的 head 参数覆盖（npz 键为 head__weight 形式）
            with _np.load(config["head_from_ckpt"]) as z:
                hs = npz_to_state({k: z[k] for k in z.files})
            head_state = {k[len("head."):]: v for k, v in hs.items()
                          if k.startswith("head.")}
            if not head_state:
                raise KeyError(f"no head.* keys in {config['head_from_ckpt']}")
            model.head.load_state_dict(head_state)
        summary["transplant"] = {"head_from": config["head_from_ckpt"],
                                 "feat_from": config["feat_from_ckpt"]}
        _phase_C(model, ctx, config, run_dir, device, C_steps, interval, bs, gstep, summary)
        summary["final"] = _final_record(run_dir)
        return

    # ---------- Phase A
    if mode != "direct" and A_steps > 0:
        stream = TrainStream(old_task, bs, seed=seed * 31 + 1, augment=(mode != "sensor"))
        opt = _make_opt(model, config, freeze_feat=False)
        losses = []
        for step in range(A_steps):
            xb, yb = stream.next(device)
            losses.append(train_step(model, opt, xb, yb, 0.0))
            gstep += 1
            if (step + 1) % interval == 0 or step == A_steps - 1:
                rec = evaluate(model, ctx, device, "A", step + 1, gstep)
                _log(run_dir, config, rec)
        summary["loss_A"] = float(np.mean(losses))
        if save_ckpt:
            save_checkpoint(model, run_dir / "checkpoints" / "end_A.npz")
        if config.get("reinit_head_after_A"):
            # P14 对照：切换瞬间直接重置分类头（last-layer reset 基线）
            set_seed(int(config["seed"]) + 991)
            reinit_head(model)
            summary["reinit_head_after_A"] = True
        ctx.ref = _feat_on(model, old_task, device, ctx.old_te) if ctx.old_probe else None

    # ---------- Phase B（近均匀标签暴露）
    total_B = sham + L
    if total_B > 0:
        stream = TrainStream(old_task, bs, seed=seed * 31 + 2, augment=(mode != "sensor"))
        opt = _make_opt(model, config, freeze_feat=False,
                        lr=float(config.get("lr_B", 1e-4)))   # 冻结协议值（评审 #5）
        losses, mid_at = [], total_B // 2
        dense_until = int(config.get("B_dense_steps", 400))   # 冻结协议值（评审 #5）
        dense_int = int(config.get("B_dense_interval", 5))
        for step in range(total_B):
            eps = 0.0 if step < sham else eps_B
            xb, yb = stream.next(device)
            losses.append(train_step(model, opt, xb, yb, eps))
            gstep += 1
            phase_step = step + 1
            due = (phase_step % interval == 0 or phase_step == total_B
                   or (phase_step <= dense_until and phase_step % dense_int == 0))
            if due:
                rec = evaluate(model, ctx, device, "B", phase_step, gstep,
                               do_permutation=bool(config.get("permutation_test")) and phase_step == interval)
                rec["eps"] = eps
                _log(run_dir, config, rec)
            if save_ckpt and phase_step == mid_at:
                save_checkpoint(model, run_dir / "checkpoints" / "mid_B.npz")
        summary["loss_B"] = float(np.mean(losses))
        if save_ckpt:
            save_checkpoint(model, run_dir / "checkpoints" / "end_B.npz")

    # ---------- Phase C
    if C_steps > 0:
        _phase_C(model, ctx, config, run_dir, device, C_steps, interval, bs, gstep, summary)

    summary["final"] = _final_record(run_dir)
    if save_ckpt:
        save_checkpoint(model, run_dir / "checkpoints" / "end_C.npz")


def _phase_C(model, ctx, config, run_dir, device, C_steps, interval, bs, gstep, summary):
    mode = config.get("mode", "main")
    new_task = ctx.new_task
    seed = int(config["seed"])
    stream = TrainStream(new_task, bs, seed=seed * 31 + 3, augment=(mode != "sensor"))
    opt = _make_opt(model, config, freeze_feat=bool(config.get("freeze_features_C", False)))
    losses = []
    for step in range(C_steps):
        xb, yb = stream.next(device)
        losses.append(train_step(model, opt, xb, yb, 0.0))
        gstep += 1
        if (step + 1) % interval == 0 or step == C_steps - 1:
            rec = evaluate(model, ctx, device, "C", step + 1, gstep)
            _log(run_dir, config, rec)
    summary["loss_C"] = float(np.mean(losses))
    summary["freeze_features_C"] = bool(config.get("freeze_features_C", False))


def _final_record(run_dir: Path) -> dict:
    from utils import read_jsonl

    rows = read_jsonl(run_dir / "metrics.jsonl")
    out = {"n_points": len(rows)}
    if rows:
        last = rows[-1]
        for k in ("acc_old_cur", "acc_old_probe", "new_task_acc", "new_task_NLL", "W2_norm", "feat_drift"):
            if k in last:
                out[k] = last[k]
    # 终态 = Phase C 最后一点
    c_rows = [r for r in rows if r.get("phase") == "C"]
    if c_rows:
        for k in ("new_task_acc", "new_task_NLL", "acc_new_probe", "acc_old_probe"):
            if k in c_rows[-1]:
                out["end_C_" + k] = c_rows[-1][k]
    b_rows = [r for r in rows if r.get("phase") == "B"]
    if b_rows:
        for k in ("acc_old_probe", "acc_old_cur", "W2_norm", "feat_drift", "margin_old"):
            if k in b_rows[-1]:
                out["end_B_" + k] = b_rows[-1][k]
    a_rows = [r for r in rows if r.get("phase") == "A"]
    if a_rows:
        out["end_A_acc"] = a_rows[-1].get("acc_old_cur")
        out["end_A_probe"] = a_rows[-1].get("acc_old_probe")
    return out
