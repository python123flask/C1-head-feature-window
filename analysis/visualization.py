"""可视化：生成 PDF 矢量图（analysis/figures/*.pdf）。

用法：python analysis/visualization.py [fig1|fig2|...|all]
所有图均为矢量 PDF（matplotlib savefig format='pdf'），供论文直接引用。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

SEEDS5 = [42, 123, 456, 789, 1024]
SEEDS4 = [42, 123, 456, 789]
L_VALUES = [0, 250, 1000, 4000]
STYLE = {"high": dict(color="#1f77b4", marker="o", label="high share (CIFAR)"),
         "low": dict(color="#d62728", marker="s", label="low share (SVHN)")}


def _save(fig, name):
    p = FIGDIR / name
    fig.savefig(p, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


def _traj(run_id, phase, key):
    rows = [r for r in L.phase_rows(run_id, phase) if key in r]
    return np.array([r["phase_step"] for r in rows]), np.array([r[key] for r in rows])


def fig1_trajectories():
    """核心图：Phase B 期间 头准确率 与 特征探针 的时间尺度分离（多 seed 均值±std）。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, shared in zip(axes, ("high", "low")):
        for key, color, name in (("acc_old_cur", "#d62728", "head acc (argmax)"),
                                 ("acc_old_probe", "#1f77b4", "feature probe acc")):
            all_x, all_y = {}, {}
            for s in SEEDS5:
                rid = f"P1_main_{shared}_L4000_s{s}"
                if not L.finished_ok(rid):
                    continue
                x, y = _traj(rid, "B", key)
                for xi, yi in zip(x, y):
                    all_x.setdefault(int(xi), []).append(yi)
            if not all_x:
                continue
            xs = sorted(all_x)
            ys = np.array([all_x[k] for k in xs])
            m, sd = ys.mean(1), ys.std(1)
            ax.plot(xs, m, color=color, label=name)
            ax.fill_between(xs, m - sd, m + sd, color=color, alpha=0.18, linewidth=0)
        ax.axhline(0.50, color="gray", ls="--", lw=0.8)
        ax.axhline(0.70, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("phase-B step (uniform label exposure)")
        ax.set_title(f"{shared} share")
        ax.set_xlim(0, 4000)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("Head de-specializes before features are damaged "
                 "(mean$\\pm$std over 5 seeds)", y=1.02, fontsize=10)
    _save(fig, "fig1_head_feature_trajectories.pdf")


def fig1b_window_zoom():
    """放大窗口：前 400 步的头/探针交叉（A2 判据可视化）。"""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for key, color, name in (("acc_old_cur", "#d62728", "head acc"),
                             ("acc_old_probe", "#1f77b4", "feature probe")):
        acc = {}
        for s in SEEDS5:
            rid = f"P1_main_high_L4000_s{s}"
            if not L.finished_ok(rid):
                continue
            x, y = _traj(rid, "B", key)
            for xi, yi in zip(x, y):
                if xi <= 600:
                    acc.setdefault(int(xi), []).append(yi)
        if not acc:
            continue
        xs = sorted(acc)
        ys = np.array([acc[k] for k in xs], dtype=float)
        m, sd = ys.mean(1), ys.std(1)
        ax.plot(xs, m, color=color, label=name)
        ax.fill_between(xs, m - sd, m + sd, color=color, alpha=0.18, linewidth=0)
    ax.axhline(0.50, color="k", ls="--", lw=0.8, label="A2 head threshold (0.50)")
    ax.axhline(0.70, color="k", ls=":", lw=0.8, label="A2 probe threshold (0.70)")
    ax.set_xlabel("phase-B step")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 600)
    ax.legend(fontsize=8)
    ax.set_title("The de-specialization window (high share)", fontsize=10)
    _save(fig, "fig1b_window_zoom.pdf")


def fig2_outcome_vs_L():
    """终态新任务表现 vs 暴露长度 L，叠加三个对照。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for shared in ("high", "low"):
        accs, errs, nlls, nerrs = [], [], [], []
        for Lv in L_VALUES:
            a, n = [], []
            for s in SEEDS5:
                rid = f"P1_main_{shared}_L{Lv}_s{s}"
                if not L.finished_ok(rid):
                    continue
                c = L.final_c(rid)
                if c.get("new_task_acc") is not None:
                    a.append(c["new_task_acc"])
                    n.append(c["new_task_NLL"])
            accs.append(np.mean(a) if a else np.nan)
            errs.append(np.std(a) if a else 0)
            nlls.append(np.mean(n) if n else np.nan)
            nerrs.append(np.std(n) if n else 0)
        st = STYLE[shared]
        axes[0].errorbar(L_VALUES, accs, yerr=errs, **st, capsize=3)
        axes[1].errorbar(L_VALUES, nlls, yerr=nerrs, **st, capsize=3)
    # 对照（虚线水平参考）
    for name, color, ls in (("abrupt", "#7f7f7f", "-."), ("direct", "#bcbd22", "--"),
                            ("headonly", "#9467bd", ":"), ("headreset", "#8c564b", "-.")):
        a, n = [], []
        for s in SEEDS5:
            rid = f"P2_{name}_s{s}"
            if not L.finished_ok(rid):
                continue
            c = L.final_c(rid)
            if c.get("new_task_acc") is not None:
                a.append(c["new_task_acc"])
                n.append(c["new_task_NLL"])
        if a:
            axes[0].axhline(np.mean(a), color=color, ls=ls, lw=1.2, label=f"control: {name}")
            axes[1].axhline(np.mean(n), color=color, ls=ls, lw=1.2, label=f"control: {name}")
    axes[0].set_xlabel("exposure length L (steps)")
    axes[0].set_ylabel("new-task accuracy (end of phase C)")
    axes[1].set_xlabel("exposure length L (steps)")
    axes[1].set_ylabel("new-task NLL (end of phase C)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[0].set_title("Accuracy: window effect", fontsize=10)
    axes[1].set_title("NLL: window effect", fontsize=10)
    _save(fig, "fig2_outcome_vs_L.pdf")


def fig2b_phase_c():
    """Phase C 早期适应轨迹：窗口效应的“瞬态”形态（4 条 L 曲线）。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    colors = {0: "#7f7f7f", 250: "#1f77b4", 1000: "#2ca02c", 4000: "#d62728"}
    curves = []
    for shared, ls in (("high", "-"), ("low", "--")):
        for Lv in L_VALUES:
            acc = {}
            for s in SEEDS5:
                rid = f"P1_main_{shared}_L{Lv}_s{s}"
                if not L.finished_ok(rid):
                    continue
                for r in L.phase_rows(rid, "C"):
                    if r.get("phase_step", 0) <= 2000 and "new_task_acc" in r:
                        acc.setdefault(int(r["phase_step"]), []).append(r["new_task_acc"])
            if not acc:
                continue
            xs = sorted(acc)
            ys = np.array([acc[k] for k in xs], dtype=float)
            curves.append((shared, Lv, xs, ys.mean(1), ys.std(1), ls))
    for shared, Lv, xs, m, sd, ls in curves:
        lab = f"L={Lv} ({shared})"
        axes[0].plot(xs, m, color=colors[Lv], ls=ls, label=lab)
        axes[0].fill_between(xs, m - sd, m + sd, color=colors[Lv], alpha=0.10, linewidth=0)
        axes[1].plot(xs, m, color=colors[Lv], ls=ls, label=lab)
    axes[0].set_ylabel("new-task accuracy")
    axes[1].set_ylabel("new-task accuracy")
    for ax in axes:
        ax.set_xlabel("phase-C step")
        ax.legend(fontsize=7, ncol=2)
    axes[1].set_xlim(0, 500)
    axes[0].set_title("Adaptation trajectories (mean$\\pm$std, 5 seeds)", fontsize=10)
    _save(fig, "fig2b_phase_c_trajectories.pdf")


def fig3_adaptation_speed():
    """进入 Phase C 后前 1000 步的窗口均值（acc / NLL）vs L —— 窗口效应的主读数。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for shared in ("high", "low"):
        accs, acc_err, nlls, nll_err = [], [], [], []
        for Lv in L_VALUES:
            a, n = [], []
            for s in SEEDS5:
                rid = f"P1_main_{shared}_L{Lv}_s{s}"
                if not L.finished_ok(rid):
                    continue
                sp = L.adaptation_speed(rid, 1000)
                if sp.get("acc_mean") is not None:
                    a.append(sp["acc_mean"])
                    n.append(sp["nll_mean"])
            accs.append(np.mean(a) if a else np.nan)
            acc_err.append(np.std(a) if a else 0)
            nlls.append(np.mean(n) if n else np.nan)
            nll_err.append(np.std(n) if n else 0)
        st = STYLE[shared]
        axes[0].errorbar(L_VALUES, accs, yerr=acc_err, **st, capsize=3)
        axes[1].errorbar(L_VALUES, nlls, yerr=nll_err, **st, capsize=3)
    for ax, yl, ttl in ((axes[0], "mean new-task accuracy (first 1000 steps)",
                         "Adaptation window: accuracy"),
                        (axes[1], "mean new-task NLL (first 1000 steps)",
                         "Adaptation window: NLL")):
        ax.set_xlabel("exposure length L (steps)")
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=8)
    _save(fig, "fig3_adaptation_speed.pdf")


def fig4_transplant():
    """交叉移植：窗口效应跟随头还是特征。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    labels, accs, nlls = [], [], []
    groups = [("native L=250", [f"P1_main_high_L250_s{s}" for s in SEEDS5]),
              ("native L=4000", [f"P1_main_high_L4000_s{s}" for s in SEEDS5]),
              ("short-head + long-feat", [f"P4_headShort_featLong_s{s}" for s in SEEDS5]),
              ("long-head + short-feat", [f"P4_headLong_featShort_s{s}" for s in SEEDS5])]
    for lab, ids in groups:
        a, n = [], []
        for rid in ids:
            if not L.finished_ok(rid):
                continue
            c = L.final_c(rid)
            if c.get("new_task_acc") is not None:
                a.append(c["new_task_acc"])
                n.append(c["new_task_NLL"])
        labels.append(lab)
        accs.append((np.mean(a), np.std(a)) if a else (np.nan, 0))
        nlls.append((np.mean(n), np.std(n)) if n else (np.nan, 0))
    x = np.arange(len(labels))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    axes[0].bar(x, [v[0] for v in accs], yerr=[v[1] for v in accs],
                color=colors, capsize=3, alpha=0.85)
    axes[1].bar(x, [v[0] for v in nlls], yerr=[v[1] for v in nlls],
                color=colors, capsize=3, alpha=0.85)
    for ax, yl in zip(axes, ("new-task accuracy", "new-task NLL")):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        ax.set_ylabel(yl)
    axes[0].set_title("Cross-transplant (causal chain)", fontsize=10)
    _save(fig, "fig4_transplant.pdf")


def fig5_diagnostics():
    """E3 诊断：均匀暴露 vs 续训（ε=0）下的头/特征/范数轨迹。"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = [("acc_old_cur", "head accuracy"), ("acc_old_probe", "feature probe"),
              ("W2_norm", "‖W₂‖ (Frobenius)")]
    for ax, (key, title) in zip(axes, panels):
        for eps, color in ((1.0, "#1f77b4"), (0.0, "#7f7f7f")):
            acc = {}
            for s in SEEDS4:
                rid = f"P5_diag_eps{eps:g}_s{s}"
                if not L.finished_ok(rid):
                    continue
                x, y = _traj(rid, "B", key)
                for xi, yi in zip(x, y):
                    acc.setdefault(int(xi), []).append(yi)
            if not acc:
                continue
            xs = sorted(acc)
            ys = np.array([acc[k] for k in xs])
            m = ys.mean(1)
            lab = ("uniform exposure (ε=1.0)" if eps == 1.0 else "continued CE (ε=0)")
            ax.plot(xs, m, color=color, label=lab)
        ax.set_xlabel("phase-B step")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    _save(fig, "fig5_E3_diagnostics.pdf")


def fig6_sensor():
    """真实传感器漂移（UCI Gas）终态 vs L。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for batch, color, marker in ((5, "#1f77b4", "o"), (10, "#d62728", "s")):
        xs, accs, errs, nlls, nerrs = [], [], [], [], []
        for Lv in (0, 500, 2000):
            a, n = [], []
            for s in SEEDS4:
                rid = f"P6_sensor_L{Lv}_s{s}"
                if batch != 5:
                    rid += f"_b{batch}"
                if not L.finished_ok(rid):
                    continue
                c = L.final_c(rid)
                if c.get("new_task_acc") is not None:
                    a.append(c["new_task_acc"])
                    n.append(c["new_task_NLL"])
            if a:
                xs.append(Lv)
                accs.append(np.mean(a)); errs.append(np.std(a))
                nlls.append(np.mean(n)); nerrs.append(np.std(n))
        if xs:
            axes[0].errorbar(xs, accs, yerr=errs, color=color, marker=marker,
                             capsize=3, label=f"batch 1 → batch {batch}")
            axes[1].errorbar(xs, nlls, yerr=nerrs, color=color, marker=marker,
                             capsize=3, label=f"batch 1 → batch {batch}")
    for ax, yl in zip(axes, ("new-task accuracy", "new-task NLL")):
        ax.set_xlabel("exposure length L (steps)")
        ax.set_ylabel(yl)
        ax.legend(fontsize=8)
    axes[0].set_title("Real sensor-drift transfer", fontsize=10)
    _save(fig, "fig6_sensor_drift.pdf")


def fig7_dose_curve():
    """剂量-响应曲线：L=4000 时头/探针/范数随 ε（目标信息量）的变化。"""
    spec = [(0.0, "P5_diag_eps0_s{s}", SEEDS4),
            (0.9, "P7_dose_eps0.9_L4000_s{s}", [42, 123, 456]),
            (0.95, "P9_eps0.95_L4000_s{s}", [42, 123, 456]),
            (0.99, "P9_eps0.99_L4000_s{s}", [42, 123, 456]),
            (1.0, "P1_main_high_L4000_s{s}", SEEDS5)]
    xs, heads, probes, norms, accs = [], [], [], [], []
    for eps, pat, seeds in spec:
        ids = [pat.format(s=s) for s in seeds]
        def vals(key, getter):
            out = []
            for rid in ids:
                if L.finished_ok(rid):
                    v = getter(rid)
                    if v is not None:
                        out.append(v)
            return out
        eh = vals("h", lambda r: L.end_b(r).get("acc_old_cur"))
        ep = vals("p", lambda r: L.end_b(r).get("acc_old_probe"))
        nw = vals("w", lambda r: L.end_b(r).get("W2_norm"))
        ac = vals("a", lambda r: L.final_c(r).get("new_task_acc"))
        if not eh:
            continue
        xs.append(eps)
        heads.append(np.mean(eh))
        probes.append(np.mean(ep))
        norms.append(np.mean(nw))
        accs.append(np.mean(ac))
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(xs, np.array(probes)[order], "o-", color="#1f77b4", label="feature probe")
    axes[0].plot(xs, np.array(heads)[order], "s-", color="#d62728", label="head acc")
    axes[0].set_ylabel("accuracy at end of exposure")
    axes[0].legend(fontsize=8)
    axes[1].plot(xs, np.array(norms)[order], "d-", color="#555555")
    axes[1].set_ylabel(r"$\|W_2\|$ at end of exposure")
    axes[2].plot(xs, np.array(accs)[order], "^-", color="#2ca02c")
    axes[2].set_ylabel("new-task accuracy (terminal)")
    for ax in axes:
        ax.set_xlabel(r"exposure target $\varepsilon$  (0 = continued CE, 1 = uniform)")
    axes[0].set_title("Information content of the target", fontsize=10)
    _save(fig, "fig7_dose_curve.pdf")


def fig8_budget():
    """适应预算曲线：相对 L=0 的配对增益（左列=窗口均值，右列=端点值）。"""
    import statistics as st
    BUDGETS = [50, 100, 250, 500, 1000, 1500, 2000]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    colors = {250: "#1f77b4", 1000: "#2ca02c", 4000: "#d62728"}

    def deltas(shared, B, Lv, mode):
        out = []
        for s in SEEDS5:
            rid0, rid1 = f"P1_main_{shared}_L0_s{s}", f"P1_main_{shared}_L{Lv}_s{s}"
            if not (L.finished_ok(rid0) and L.finished_ok(rid1)):
                continue
            if mode == "mean":
                f = lambda rid: (st.mean(
                    r["new_task_acc"] for r in L.phase_rows(rid, "C")
                    if r.get("phase_step", 0) <= B and "new_task_acc" in r)
                    if any(r.get("phase_step", 0) <= B for r in L.phase_rows(rid, "C")) else None)
            else:
                f = lambda rid: next((r["new_task_acc"] for r in L.phase_rows(rid, "C")
                                      if r.get("phase_step") == B), None)
            a, b = f(rid0), f(rid1)
            if a is not None and b is not None:
                out.append(b - a)
        return out

    for row, shared in enumerate(("high", "low")):
        for col, mode in enumerate(("mean", "point")):
            ax = axes[row][col]
            for Lv in (250, 1000, 4000):
                ys, es = [], []
                for B in BUDGETS:
                    d = deltas(shared, B, Lv, mode)
                    ys.append(100 * st.mean(d) if d else np.nan)
                    es.append(100 * st.pstdev(d) if d else 0)
                ax.errorbar(BUDGETS, ys, yerr=es, color=colors[Lv], marker="o",
                            capsize=2.5, label=f"L={Lv}")
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_xlabel("adaptation budget B (phase-C steps)")
            kind = "window mean" if mode == "mean" else "value at B"
            ax.set_title(f"{shared} share — {kind}", fontsize=10)
            if col == 0:
                ax.set_ylabel("gain vs $L{=}0$ (accuracy points)")
    axes[0][1].legend(fontsize=8)
    _save(fig, "fig8_budget_gain.pdf")


def fig9_external():
    """外部效度：CIFAR-100 的头/特征轨迹 + 各扩展任务的窗口曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9))

    # (a) CIFAR-100 / SmallCNN：Phase B 轨迹（L=4000，3 seeds）
    for key, color, name in (("acc_old_cur", "#d62728", "head acc"),
                             ("acc_old_probe", "#1f77b4", "feature probe")):
        acc = {}
        for s in (42, 123, 456):
            rid = f"P11_c100_smallcnn_L4000_s{s}"
            if not L.finished_ok(rid):
                continue
            x, y = _traj(rid, "B", key)
            for xi, yi in zip(x, y):
                acc.setdefault(int(xi), []).append(yi)
        if not acc:
            continue
        xs = sorted(acc)
        ys = np.array([acc[k] for k in xs], dtype=float)
        m, sd = ys.mean(1), ys.std(1)
        axes[0].plot(xs, m, color=color, label=name)
        axes[0].fill_between(xs, m - sd, m + sd, color=color, alpha=0.18, linewidth=0)
    axes[0].axhline(0.5, color="k", ls="--", lw=0.8)
    axes[0].axhline(0.7, color="k", ls=":", lw=0.8)
    axes[0].set_xlabel("phase-B step (uniform exposure)")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("CIFAR-100 (25+25 classes), SmallCNN", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0, 4000)

    # (b) 扩展任务的窗口曲线：speed1000 vs L
    combos = [
        ("CIFAR-10 high (5 seeds)", "P1_main_high_L{L}_s{s}", SEEDS5, "o", "#1f77b4"),
        ("CIFAR-100 SmallCNN", "P11_c100_smallcnn_L{L}_s{s}", [42, 123, 456], "s", "#d62728"),
        ("CIFAR-100 ResNet-18", "P11_c100_resnet18_L{L}_s{s}", [42, 123, 456], "^", "#2ca02c"),
        ("→ Fashion-MNIST (low)", "P12_fmnist_L{L}_s{s}", [42, 123, 456], "D", "#9467bd"),
    ]
    for label, pat, seeds, mk, color in combos:
        ys, es, xs = [], [], []
        for Lv in (0, 250, 4000):
            vals = []
            for s in seeds:
                rid = pat.format(L=Lv, s=s)
                if L.finished_ok(rid):
                    v = L.adaptation_speed(rid, 1000).get("acc_mean")
                    if v is not None:
                        vals.append(v)
            if vals:
                xs.append(Lv)
                ys.append(np.mean(vals))
                es.append(np.std(vals))
        if xs:
            axes[1].errorbar(xs, ys, yerr=es, marker=mk, color=color, capsize=3,
                             label=label, lw=1.6)
    axes[1].set_xlabel("exposure length L (steps)")
    axes[1].set_ylabel("mean new-task acc (first 1000 steps)")
    axes[1].set_title("The window across benchmarks", fontsize=10)
    axes[1].legend(fontsize=7.5)
    _save(fig, "fig9_external_validity.pdf")


FIGS = {"fig1": fig1_trajectories, "fig1b": fig1b_window_zoom, "fig2": fig2_outcome_vs_L,
        "fig2b": fig2b_phase_c, "fig3": fig3_adaptation_speed,
        "fig4": fig4_transplant, "fig5": fig5_diagnostics, "fig6": fig6_sensor,
        "fig7": fig7_dose_curve, "fig8": fig8_budget, "fig9": fig9_external}


def main():
    which = sys.argv[1:] or ["all"]
    for name, fn in FIGS.items():
        if "all" in which or name in which:
            try:
                fn()
            except Exception as e:
                print(f"[skip] {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
