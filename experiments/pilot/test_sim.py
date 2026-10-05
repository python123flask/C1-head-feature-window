"""S-tests：部署前本机单测（gold 数值核对）。

运行：  .venv/Scripts/python experiments/pilot/test_sim.py [名称子串]
机器门：全部 PASS（exit 0）；科学门：gold 数值核对通过。
"""
from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M                                  # noqa: E402
import sim                                           # noqa: E402
from data_loader import TaskData                     # noqa: E402
from models import build_model, freeze_features, trainable_params  # noqa: E402
from utils import load_checkpoint, save_checkpoint, set_seed       # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- 构造工具
def synthetic_task(n_train=512, n_test=256, n_class=4, seed=0, dim=(3, 32, 32)):
    g = np.random.default_rng(seed)
    def make(n):
        y = g.integers(0, n_class, size=n)
        x = np.empty((n, *dim), dtype=np.uint8)
        # 每类一个基色 + 噪声（可分信号）
        for c in range(n_class):
            m = y == c
            base = np.array([(c * 57 + 20) % 256, (c * 111 + 40) % 256, (c * 199 + 60) % 256], dtype=np.uint8)
            x[m] = base[None, :, None, None]
        x = np.clip(x.astype(np.int16) + g.integers(0, 60, size=x.shape), 0, 255).astype(np.uint8)
        return x, y
    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    return TaskData("synthetic", xtr, ytr, xte, yte,
                    [0.5, 0.5, 0.5], [0.25, 0.25, 0.25],
                    num_classes=n_class, slots=list(range(n_class)), head_dim=n_class)


def tiny_config(**kw):
    cfg = dict(run_id="unit", seed=42, mode="main", shared="high", L=50, sham_steps=0,
               eps_B=0.9, phase_A_steps=120, phase_C_steps=60, eval_interval=25,
               batch_size=64, lr=1e-3, weight_decay=1e-4, probe_train_max=256,
               probe_test_max=128, lam=0.1, head_dim=4, model="smallcnn",
               save_ckpt=True, data_seed=0)
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------- gold 单测
def test_smoothing_gradient_gold():
    """gold：CE(ε) 对 logits 的梯度 = (softmax(z) - [(1-ε)y + ε/C]) / N（mean 归约）。"""
    for eps in (0.0, 0.5, 0.9):
        C, N = 10, 8
        z = torch.randn(N, C, dtype=torch.double, requires_grad=True)
        y = torch.randint(0, C, (N,))
        loss = F.cross_entropy(z, y, label_smoothing=eps)   # reduction='mean'
        loss.backward()
        p = torch.softmax(z.detach(), dim=1)
        target = torch.full((N, C), eps / C, dtype=torch.double)
        target[torch.arange(N), y] += 1.0 - eps
        gold = (p - target) / N
        assert torch.allclose(z.grad, gold, atol=1e-12), f"eps={eps} maxerr={(z.grad - gold).abs().max()}"


def test_smoothing_optimum_gold():
    """gold：软标签交叉熵最小值 = H(target)，当 p = target 时取到。"""
    C, eps = 6, 0.9
    y = 2
    target = torch.full((C,), eps / C, dtype=torch.double)
    target[y] += 1 - eps
    z = torch.log(target).unsqueeze(0)
    loss = F.cross_entropy(z, torch.tensor([y]), label_smoothing=eps)
    H = -(target * target.log()).sum()
    assert abs(float(loss) - float(H)) < 1e-9, (float(loss), float(H))


def test_probe_gold_signal():
    """gold：线性规则 -> 探针接近 100%；打乱标签 -> 接近随机（E4 排除项）。"""
    g = torch.Generator().manual_seed(0)
    # 二分类：y = 1[x0 > 0]（线性可实现，且与贝叶斯边界一致）
    X = torch.randn(4096, 16, generator=g)
    y = (X[:, 0] > 0).long()
    Xte = torch.randn(2048, 16, generator=g)
    yte = (Xte[:, 0] > 0).long()
    acc = M._ridge_acc(X, y, Xte, yte, lam=1e-3)
    assert acc > 0.97, f"binary probe acc {acc}"
    acc_shuf = M._ridge_acc(X, y, Xte, yte, lam=1e-3, shuffle_labels=True)
    assert acc_shuf < 0.6, f"shuffled acc {acc_shuf} (should be ~chance)"
    # 多分类：由线性 logits 生成的标签（真实边界为 argmax 线性）
    W = torch.randn(16, 5, generator=g)
    X = torch.randn(4096, 16, generator=g)
    y = (X @ W).argmax(1)
    Xte = torch.randn(2048, 16, generator=g)
    yte = (Xte @ W).argmax(1)
    acc = M._ridge_acc(X, y, Xte, yte, lam=1e-3)
    assert acc > 0.7, f"multiclass probe acc {acc}"
    acc_shuf = M._ridge_acc(X, y, Xte, yte, lam=1e-3, shuffle_labels=True)
    assert acc_shuf < 0.4, f"shuffled multiclass acc {acc_shuf}"


def test_probe_train_test_split_discipline():
    """探针必须 train 拟合 / test 评估：用测试标签拟合会得到虚高分（此处验证接口一致性）。"""
    task = synthetic_task()
    model = build_model("smallcnn", task.head_dim).to(DEVICE)
    acc = M.ridge_probe_acc(model, task, DEVICE, max_train=256)
    assert 0.0 <= acc <= 1.0 and not math.isnan(acc)


def test_determinism():
    """同 seed 两次初始化/训练 -> 损失序列完全一致。"""
    outs = []
    for rep in range(2):
        set_seed(123)
        model = build_model("smallcnn", 4).to(DEVICE)
        task = synthetic_task(seed=7)
        stream = __import__("data_loader").TrainStream(task, 64, seed=999, augment=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        losses = []
        for _ in range(10):
            xb, yb = stream.next(DEVICE)
            losses.append(sim.train_step(model, opt, xb, yb, 0.0))
        outs.append(losses)
    assert np.allclose(outs[0], outs[1], atol=1e-6), (outs[0], outs[1])


def test_freeze_features_headonly():
    """headonly：特征参数在 Phase C 不更新，头参数更新。"""
    model = build_model("smallcnn", 4).to(DEVICE)
    task = synthetic_task()
    freeze_features(model)
    f_before = [p.detach().clone() for p in model.features.parameters()]
    h_before = [p.detach().clone() for p in model.head.parameters()]
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    stream = __import__("data_loader").TrainStream(task, 64, seed=1, augment=True)
    for _ in range(5):
        xb, yb = stream.next(DEVICE)
        sim.train_step(model, opt, xb, yb, 0.0)
    for a, b in zip(f_before, model.features.parameters()):
        assert torch.equal(a, b.detach()), "features changed despite freeze"
    assert any(not torch.equal(a, b.detach()) for a, b in zip(h_before, model.head.parameters())), \
        "head did not train"


def test_checkpoint_roundtrip():
    set_seed(7)
    m1 = build_model("smallcnn", 4).to(DEVICE)
    task = synthetic_task()
    xb = task.prepare(task.x_test[:32], DEVICE)
    ref = m1(xb).detach()
    path = Path(__file__).resolve().parent / "_tmp_ckpt.npz"
    save_checkpoint(m1, path)
    set_seed(99)
    m2 = build_model("smallcnn", 4).to(DEVICE)
    load_checkpoint(m2, path)
    out = m2(xb)
    assert torch.allclose(ref, out, atol=1e-6)
    path.unlink()


def test_transplant_composition():
    """移植：合成体的输出 = 特征源的 φ + 头源的 W2。"""
    set_seed(1)
    a = build_model("smallcnn", 4).to(DEVICE)
    set_seed(2)
    b = build_model("smallcnn", 4).to(DEVICE)
    task = synthetic_task()
    xb = task.prepare(task.x_test[:16], DEVICE)
    phi = b.features(xb)
    gold = a.head(phi)                      # 头来自 a，特征来自 b
    # 组合 b 的特征 + a 的头
    set_seed(3)
    c = build_model("smallcnn", 4).to(DEVICE)
    c.load_state_dict(b.state_dict())
    c.head.load_state_dict(a.head.state_dict())
    out = c(xb)
    assert torch.allclose(gold, out, atol=1e-6)


def test_transplant_npz_head_loading():
    """P4 移植路径：npz 键名还原 + 头覆盖必须与手工组合逐位一致（回归 B10）。"""
    import numpy as np

    from utils import npz_to_state
    set_seed(1)
    a = build_model("smallcnn", 4).to(DEVICE)     # 头来源
    set_seed(2)
    b = build_model("smallcnn", 4).to(DEVICE)     # 特征来源
    p_dir = Path(__file__).resolve().parent
    p_head, p_feat = p_dir / "_tmp_h.npz", p_dir / "_tmp_f.npz"
    save_checkpoint(a, p_head)
    save_checkpoint(b, p_feat)
    try:
        set_seed(3)
        m = build_model("smallcnn", 4).to(DEVICE)
        load_checkpoint(m, p_feat)                       # 1) 特征来源完整状态
        with np.load(p_head) as z:
            hs = npz_to_state({k: z[k] for k in z.files})  # 2) 头来源
        head_state = {k[len("head."):]: v for k, v in hs.items() if k.startswith("head.")}
        assert sorted(head_state) == ["bias", "weight"], sorted(head_state)
        m.head.load_state_dict(head_state)
        task = synthetic_task()
        xb = task.prepare(task.x_test[:16], DEVICE)
        gold = a.head(b.features(xb))                  # 手工组合
        assert torch.allclose(m(xb), gold, atol=1e-6)
    finally:
        p_head.unlink(missing_ok=True)
        p_feat.unlink(missing_ok=True)


def test_schedule_gold():
    """gold：评测点数下限（A1 ≥ 200）在所有主协议配置下成立。"""
    interval = 25
    for L in (0, 250, 1000, 4000):
        total = 5000 + L + 2000
        pts = (5000 // interval) + (L // interval if L else 0) + (2000 // interval)
        assert pts >= 200, f"L={L} gives {pts} points < 200"


def test_task_counts_with_data():
    """数据构造：CIFAR 旧任务 1000 train/500 test 每类，train/test 不重叠。"""
    from data_loader import DATA
    if not (DATA / "cache" / "cifar_full.npz").exists() and \
            not (DATA / "cifar-10-batches-py").exists() and \
            not (DATA / "cifar-10-python.tar.gz").exists():
        print("      (SKIP: CIFAR 数据未就绪)")
        return
    old, new = __import__("data_loader").make_cifar_tasks(seed=0)
    for t in (old, new):
        tc = np.bincount(t.y_train.numpy(), minlength=5)
        ec = np.bincount(t.y_test.numpy(), minlength=5)
        assert (tc == 1000).all(), tc
        assert (ec == 500).all(), ec
    # train 与 test 图像不共享（按字节校验抽样）
    a = old.x_train[:50].numpy().tobytes()
    b = old.x_test[:50].numpy().tobytes()
    assert a != b
    assert old.n_train == 5000 and old.n_test == 2500


def test_svhn_counts_with_data():
    from data_loader import DATA
    if not (DATA / "cache" / "svhn_full.npz").exists() and not (DATA / "train_32x32.mat").exists():
        print("      (SKIP: SVHN 数据未就绪)")
        return
    t = __import__("data_loader").make_svhn_task(seed=0)
    assert t.n_train == 10000 and t.n_test == 5000, (t.n_train, t.n_test)
    assert t.y_train.min() == 0 and t.y_train.max() == 9
    assert set(np.bincount(t.y_train.numpy()).tolist()) == {1000}
    assert set(np.bincount(t.y_test.numpy()).tolist()) == {500}


def test_gas_sensor_with_data():
    """传感器数据：批次 1 -> 批次 5 构造与归一化。"""
    from data_loader import make_sensor_tasks
    old, new = make_sensor_tasks(seed=0, head_dim=6, batch_new=5)
    assert old.kind == "vector" and old.x_train.shape[1] == 128
    assert old.n_train > 100 and new.n_train > 50, (old.n_train, new.n_train)
    assert old.num_classes == 6
    assert not np.isnan(old.x_train.numpy()).any()
    model = build_model("smallmlp", 6).to(DEVICE)
    acc = M.ridge_probe_acc(model, old, DEVICE, max_train=None)
    assert 0.0 <= acc <= 1.0
    # 训练步可跑通
    from data_loader import TrainStream
    stream = TrainStream(old, 32, seed=1, augment=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xb, yb = stream.next(DEVICE)
    loss = sim.train_step(model, opt, xb, yb, 0.0)
    assert np.isfinite(loss)


def test_evaluate_finite():
    """E4：单点评测输出全部有限且在合理区间。"""
    set_seed(3)
    task = synthetic_task()
    model = build_model("smallcnn", task.head_dim).to(DEVICE)
    cfg = tiny_config()
    ctx = sim.EvalCtx(cfg, task, task)
    rec = sim.evaluate(model, ctx, DEVICE, "A", 0, 0)
    for k, v in rec.items():
        if isinstance(v, float):
            assert not (math.isnan(v) or math.isinf(v)), f"{k}={v}"
    assert 0.0 <= rec["acc_old_cur"] <= 1.0
    assert 0.0 <= rec["acc_old_probe"] <= 1.0
    assert rec["W2_norm"] > 0
    # Phase A 无 end_A 参考 → 不写 feat_drift（而不是伪造 0.0，见评审 B16）
    assert "feat_drift" not in rec, sorted(rec)
    # 建立参考后（Phase B/C）feat_drift 必须存在且 >= 0（参考须取同一固定子集）
    ctx.ref = sim._feat_on(model, task, DEVICE, ctx.old_te)
    rec_b = sim.evaluate(model, ctx, DEVICE, "B", 25, 25)
    assert "feat_drift" in rec_b and rec_b["feat_drift"] >= 0, rec_b.get("feat_drift")
    for k, v in rec_b.items():
        if isinstance(v, float):
            assert not (math.isnan(v) or math.isinf(v)), f"B:{k}={v}"


def test_train_step_loss_decreases():
    set_seed(5)
    task = synthetic_task()
    model = build_model("smallcnn", task.head_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    stream = __import__("data_loader").TrainStream(task, 64, seed=1, augment=True)
    losses = [sim.train_step(model, opt, *stream.next(DEVICE), 0.0) for _ in range(40)]
    assert np.mean(losses[-10:]) < np.mean(losses[:10]), (losses[0], losses[-1])


def test_smoothing_pushes_toward_uniform():
    """机制前提（S-exposure 校准结论，见 pilot/DEVIATIONS.md §E）：
    - ε=1.0 真均匀目标 -> 头失去专化（A2 前提）
    - ε=0.9 输入相关平滑 -> 目标仍以真类为最大，argmax 序被保留，头准确率不降
      （这正是计划原文 ε=0.9 无法达成 A2 判据的机制原因）
    """

    def exposure(eps, steps=200):
        set_seed(11)
        task = synthetic_task()
        model = build_model("smallcnn", task.head_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        stream = __import__("data_loader").TrainStream(task, 64, seed=3, augment=False)
        for _ in range(60):
            sim.train_step(model, opt, *stream.next(DEVICE), 0.0)
        acc_before = M.head_accuracy(model, task, DEVICE)
        for _ in range(steps):
            sim.train_step(model, opt, *stream.next(DEVICE), eps)
        return acc_before, M.head_accuracy(model, task, DEVICE)

    before_u, after_u = exposure(1.0)
    before_s, after_s = exposure(0.9)
    assert before_u > 0.8, before_u
    assert after_u < 0.5, f"uniform exposure should de-specialize the head, got {after_u}"
    assert after_s > 0.6, f"input-dependent smoothing keeps argmax ordering, got {after_s}"
    assert after_u < after_s, (after_u, after_s)


# ---------------------------------------------------------------- runner
def main():
    names = [k for k in globals() if k.startswith("test_")]
    sel = sys.argv[1] if len(sys.argv) > 1 else ""
    names = sorted(n for n in names if sel in n)
    failed = []
    for n in names:
        try:
            globals()[n]()
            print(f"PASS  {n}")
        except Exception:
            failed.append(n)
            print(f"FAIL  {n}")
            traceback.print_exc()
    print(f"\n{len(names) - len(failed)}/{len(names)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
