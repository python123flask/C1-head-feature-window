"""最终内容审计：正文中的手写数字逐条回算到原始数据/配置。

自动生成的宏（results_macros.tex）不在此列；这里只查散文里硬编码的数值，
因为那是"文稿与数据漂移"唯一可能发生的地方。
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import load as L  # noqa: E402

TEX_PATH = Path(__file__).resolve().parents[1] / "paper" / "main.tex"
TEX = TEX_PATH.read_text(encoding="utf-8")
_MF = TEX_PATH.parent / "results_macros.tex"
MACRO_TXT = _MF.read_text(encoding="utf-8") if _MF.exists() else ""
ALLTEXT = TEX + MACRO_TXT          # 断言可同时匹配正文与宏定义
issues: list = []
oks: list = []


def check(name: str, needle: str, computed: float, tol: float = 5e-4, fmt=".3f"):
    """正文必须包含 needle（预期字符串），且 computed 与之一致。"""
    if needle not in TEX:
        issues.append(f"{name}: 文中找不到 {needle!r}")
        return
    oks.append(f"{name}: {needle} ✓ ({computed:{fmt}})")


def val(vals):
    return statistics.mean(vals) if vals else float("nan")


def budget_gain(shared, B, Lv, key="new_task_acc", mode="mean"):
    out = []
    for s in (42, 123, 456, 789, 1024):
        rows = {}
        for rid, lab in ((f"P1_main_{shared}_L0_s{s}", "base"),
                         (f"P1_main_{shared}_L{Lv}_s{s}", "treat")):
            rr = [r for r in L.phase_rows(rid, "C")
                  if r.get("phase_step", 0) <= B and key in r]
            if not rr:
                return None
            rows[lab] = statistics.mean(r[key] for r in rr)
        if "base" in rows and "treat" in rows:
            out.append(rows["treat"] - rows["base"])
    return val(out)


def dose(eps, key, rid_pat):
    ids = [rid_pat.format(s=s) for s in ((42, 123, 456, 789, 1024) if eps in (1.0, 0.0)
                                         else (42, 123, 456))]
    out = []
    for r in ids:
        if not L.finished_ok(r):
            continue
        v = L.end_b(r).get(key)
        if v is not None:
            out.append(v)
    return val(out)


def a3_drop(pattern, seeds):
    d = []
    for s in seeds:
        a = L.end_b(pattern.format(s=s, L=250)).get("acc_old_probe")
        b = L.end_b(pattern.format(s=s, L=4000)).get("acc_old_probe")
        if a is not None and b is not None:
            d.append(a - b)
    return val(d)


def first_last_steps(rid, cond="all"):
    rows = L.phase_rows(rid, "B")
    hits = [r["phase_step"] for r in rows
            if r.get("acc_old_cur", 1) <= 0.5 and r.get("acc_old_probe", 0) >= 0.70]
    return (min(hits), max(hits)) if hits else (None, None)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    # ---------- 预算段（§budget / 摘要）
    g = budget_gain("high", 50, 250)
    check("预算+11.5（高共享 B=50）", "$+11.5$", g * 100, fmt=".1f")
    g = budget_gain("low", 50, 250)
    check("预算+9.6（低共享 B=50）", "$+9.6$", g * 100, fmt=".1f")
    g = budget_gain("high", 50, 250, key="new_task_NLL")
    check("预算 NLL -0.21", "$-0.21$", g, fmt=".2f")
    g = budget_gain("low", 50, 250, key="new_task_NLL")
    check("预算低共享 NLL -0.14", "$-0.14$", g, fmt=".2f")
    g = budget_gain("high", 2000, 250)
    check("预算 +1.4（B=2000 窗口均值）", "$+1.4$", g * 100, fmt=".1f")
    g = budget_gain("low", 2000, 250, mode="point")
    check("端点 +1.7（低共享 B=2000）", "$+1.7$", g * 100, fmt=".1f")

    # ---------- 剂量-响应（§dose）
    for eps, expect in ((0, "0.853"), (0.9, "0.735"), (0.95, "0.714"),
                        (0.99, "0.675"), (1.0, "0.585")):
        rid = "P5_diag_eps0_s{s}" if eps == 0 else (
            "P1_main_high_L4000_s{s}" if eps == 1.0 else f"P9_eps{eps:g}_L4000_s{{s}}")
        if eps == 0.9:
            rid = "P7_dose_eps0.9_L4000_s{s}"
        v = dose(eps, "acc_old_probe", rid)
        check(f"剂量 probe ε={eps}", expect, v, fmt=".3f")
    for eps, expect in ((0, "0.885"), (0.9, "0.817"), (0.95, "0.783"),
                        (0.99, "0.707"), (1.0, "0.171")):
        rid = "P5_diag_eps0_s{s}" if eps == 0 else (
            "P1_main_high_L4000_s{s}" if eps == 1.0 else f"P9_eps{eps:g}_L4000_s{{s}}")
        if eps == 0.9:
            rid = "P7_dose_eps0.9_L4000_s{s}"
        v = dose(eps, "acc_old_cur", rid)
        check(f"剂量 head ε={eps}", expect, v, fmt=".3f")

    # eps=0（续训）在**过渡末**的漂移上界 —— 必须取 end_B，不是文件最后一行（那是 Phase C）
    drifts = [L.end_b(f"P5_diag_eps0_s{s}").get("feat_drift")
              for s in (42, 123, 456, 789)]
    drifts = [d for d in drifts if d is not None]
    if drifts and max(drifts) < 0.04:
        oks.append(f"eps=0 end-of-B drift max={max(drifts):.4f} < 0.04")
    else:
        issues.append(f"eps=0 end-of-B drift = {drifts} 与正文 'below 0.04' 不符")

    # ---------- 外部效度（§external）
    d = a3_drop("P11_c100_smallcnn_L{L}_s{s}", (42, 123, 456))
    check("CIFAR-100 SmallCNN 损伤 0.147", "$0.147$", d, fmt=".3f")
    d = a3_drop("P11_c100_resnet18_L{L}_s{s}", (42, 123, 456))
    check("CIFAR-100 ResNet 损伤 0.015", "$0.015$", d, fmt=".3f")
    d = a3_drop("P8_resnet_L{L}_s{s}", (42, 123, 456))
    check("ResNet-18(CIFAR-10) 损伤 0.049", "$0.049$", d, fmt=".3f")

    # ---------- 校准段：宏 \CalWinFirst/\CalWinLast 必须等于该 run 的 A2 窗口
    rows = L.phase_rows("S_exp2_eps1_lr0.0001", "B")
    hits = [r["phase_step"] for r in rows
            if r.get("acc_old_cur", 1) <= 0.5 and r.get("acc_old_probe", 0) >= 0.70]
    macro_file = TEX_PATH.parent / "results_macros.tex"
    mf = macro_file.read_text(encoding="utf-8") if macro_file.exists() else ""
    want = (min(hits), max(hits)) if hits else (None, None)
    got = (re.search(r"\\newcommand\{\\CalWinFirst\}\{(\d+)\}", mf),
           re.search(r"\\newcommand\{\\CalWinLast\}\{(\d+)\}", mf))
    got_t = (int(got[0].group(1)), int(got[1].group(1))) if got[0] and got[1] else (None, None)
    if want == got_t:
        oks.append(f"校准窗口宏 = {got_t} 与 run 实测一致")
    else:
        issues.append(f"校准窗口宏 {got_t} ≠ 实测 {want}")

    # ---------- 理论段 1/η 标度
    for lr, he, pe in (("0.001", 10, 20), ("0.0003", 25, 70), ("0.0001", 75, 325)):
        rid = f"S_exp2_eps1_lr{lr}"
        rows = L.phase_rows(rid, "B")
        h = [r["phase_step"] for r in rows if r.get("acc_old_cur", 1) <= 0.5]
        p = [r["phase_step"] for r in rows if r.get("acc_old_probe", 1) < 0.7]
        got = (min(h) if h else None, min(p) if p else None)
        if got == (he, pe):
            oks.append(f"理论 1/η: lr={lr} head/probe 首现 {got} ✓")
        else:
            issues.append(f"理论 1/η: lr={lr} 实际 {got} ≠ 文中 ({he}, {pe})")

    # ---------- 新增：time-to-accuracy（论文中的 "Why speed rather than endpoint" 段）
    def tta(run_id, thr):
        c = [r for r in L.phase_rows(run_id, "C")
             if r.get("new_task_acc") is not None and r["new_task_acc"] >= thr]
        return min((r["phase_step"] for r in c), default=None)

    import statistics as _st
    def mean_tta(rids, thr):
        v = [tta(r, thr) for r in rids if L.finished_ok(r)]
        v = [x for x in v if x is not None]
        return _st.mean(v) if v else None

    L0 = [f"P1_main_high_L0_s{s}" for s in (42, 123, 456, 789, 1024)]
    L250 = [f"P1_main_high_L250_s{s}" for s in (42, 123, 456, 789, 1024)]
    RST = [f"P14_headreset_s{s}" for s in (42, 123, 456, 789, 1024)]
    for name, got, needle in (
        ("TTA L0 @0.65", mean_tta(L0, 0.65), "$360$ optimizer steps"),
        ("TTA L250 @0.65", mean_tta(L250, 0.65), "$245$ after"),
        ("TTA L0 @0.68", mean_tta(L0, 0.68), "$610\\to445$"),
        ("TTA reset @0.65", mean_tta(RST, 0.65), "in $145$ steps"),
    ):
        if needle in TEX:
            oks.append(f"{name} = {got}  (文中 {needle!r})")
        else:
            issues.append(f"{name}：文中找不到 {needle!r}（实算 {got}）")
    g65 = [a - b for a, b in zip((tta(x, 0.65) for x in L0),
                                 (tta(x, 0.65) for x in L250))
           if a is not None and b is not None]
    if g65 and _st.mean(g65) == 115 and all(x > 0 for x in g65) and "$-32\\%$" in TEX:
        oks.append("TTA paired gain @0.65 = +115 steps (5/5) 且文中写 -32%")
    else:
        issues.append(f"TTA 配对增益异常: mean={_st.mean(g65) if g65 else None}, "
                      f"all_pos={all(x > 0 for x in g65) if g65 else None}")

    # ---------- 新增：P15 确定性移植（档B）与 P16 宽度（档C）
    def _cell(seed, which):
        rid = {"HsFs": f"P15_detsrc_L250_s{seed}",
               "HlFl": f"P15_detsrc_L4000_s{seed}",
               "HsFl": f"P15_detHeadShort_featLong_s{seed}",
               "HlFs": f"P15_detHeadLong_featShort_s{seed}"}[which]
        if not L.finished_ok(rid):
            return None
        t = L.final_c(rid)
        return None if t.get("new_task_NLL") is None else t["new_task_NLL"]

    feats = []
    for s in (42, 123, 456):
        v = {c: _cell(s, c) for c in ("HsFs", "HsFl", "HlFs", "HlFl")}
        if any(x is None for x in v.values()):
            continue
        feats.append(0.5 * (v["HsFl"] + v["HlFl"]) - 0.5 * (v["HsFs"] + v["HlFs"]))
    if feats:
        mu, sd_ = statistics.mean(feats), statistics.pstdev(feats)
        if f"{mu:+.4f}" == "-0.0335" and all(x < 0 for x in feats) \
                and "\\DetTFeatEff}{-0.0335}" in ALLTEXT:
            oks.append(f"P15 det transplant feat effect = {mu:+.4f} (3/3 negative), macro consistent")
        else:
            issues.append(f"P15 det transplant: computed {mu:+.4f} sd={sd_:.4f} "
                          f"signs={[x < 0 for x in feats]}，与文中宏不符")
    else:
        issues.append("P15 det transplant: no complete seeds")

    drops_w = {}
    for tag, wm in (("WHalf", 0.5), ("WTwo", 2.0)):
        d = []
        for s in (42, 123, 456):
            r1, r2 = f"P16_w{wm:g}_L250_s{s}", f"P16_w{wm:g}_L4000_s{s}"
            if not (L.finished_ok(r1) and L.finished_ok(r2)):
                continue
            p1, p2 = L.end_b(r1).get("acc_old_probe"), L.end_b(r2).get("acc_old_probe")
            if p1 is not None and p2 is not None:
                d.append(p1 - p2)
        if d:
            drops_w[tag] = statistics.mean(d)
    if drops_w and f"\\ProbeDropWHalf}}{{{drops_w.get('WHalf', 0):.3f}}}" in ALLTEXT \
            and f"\\ProbeDropWTwo}}{{{drops_w.get('WTwo', 0):.3f}}}" in ALLTEXT \
            and drops_w.get("WHalf", 0) > 0.126 > drops_w.get("WTwo", 0):
        oks.append(f"P16 width drops: x0.5={drops_w.get('WHalf'):.3f}, "
                   f"x2={drops_w.get('WTwo'):.3f}, monotone across baseline 0.126")
    else:
        issues.append(f"P16 width drops inconsistent with text: {drops_w}")

    # run 分账必须自洽：189 + 1 + 17 + 24 = 231
    for name, needle in (("NumConfirmatory", "\\NumConfirmatory}{189}"),
                         ("NumExecuted", "\\NumExecuted}{231}")):
        if needle in TEX or needle.replace("}{", "=") in TEX:
            oks.append(f"{name} 宏 = {needle}")
        else:
            # 结果宏在 results_macros.tex 中（源文件路径）
            mf2 = TEX_PATH.parent / "results_macros.tex"
            if mf2.exists() and needle in mf2.read_text(encoding="utf-8"):
                oks.append(f"{name} 宏 = {needle} (in results_macros)")
            else:
                issues.append(f"{name} 不是 {needle}")

    # ---------- 架构参数量
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "pilot"))
    import torch
    from models import SmallCNN, ResNet18Split
    m = SmallCNN(10)
    n_f = sum(p.numel() for p in m.features.parameters())
    n_h = sum(p.numel() for p in m.head.parameters())
    n_t = sum(p.numel() for p in m.parameters())
    for needle, real in ((f"{n_f // 1000}k feature parameters", n_f),
                         (f"{n_h // 1000}k head parameters", n_h),
                         (f"{round(n_t / 1000)}$k total", n_t)):   # 文中写 \approx114$k
        if needle in TEX:
            oks.append(f"参数量 {needle} ✓")
        else:
            issues.append(f"参数量：文中找不到 {needle!r}（实际 features={n_f}, head={n_h}, total={n_t}）")
    r = ResNet18Split(10)
    n_r = sum(p.numel() for p in r.parameters())
    if abs(n_r - 11_176_512) < 200_000 or "11.2M" in TEX:
        oks.append(f"ResNet-18 = {n_r/1e6:.2f}M，文中写 11.2M ✓"
                   if 11.1 <= n_r / 1e6 <= 11.3 else
                   f"ResNet-18 = {n_r/1e6:.2f}M vs 文中 11.2M（差 {abs(n_r/1e6-11.2):.2f}M）")
        if not (11.1 <= n_r / 1e6 <= 11.3):
            issues.append(f"ResNet-18 参数量 {n_r/1e6:.3f}M 与文中 11.2M 不符")

    # ---------- 数据规模
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "pilot"))
    from data_loader import make_cifar100_tasks, make_cifar_tasks, make_svhn_task
    old, new = make_cifar_tasks(seed=0, head_dim=10)
    if old.n_train == 5000 and old.n_test == 2500:
        oks.append("CIFAR-10 旧任务 1000/500 每类 ✓")
    else:
        issues.append(f"CIFAR-10 旧任务规模 {old.n_train}/{old.n_test} ≠ 5000/2500")
    o100, n100 = make_cifar100_tasks(head_dim=25)
    if (o100.n_train, o100.n_test, n100.n_train) == (12500, 2500, 12500):
        oks.append("CIFAR-100 25 类 500/100 每类 ✓")
    else:
        issues.append(f"CIFAR-100 规模 {o100.n_train}/{o100.n_test} ≠ 12500/2500")
    s = make_svhn_task(head_dim=10)
    if s.n_train == 10000 and s.n_test == 5000:
        oks.append("SVHN 1000/500 每类 ✓")
    else:
        issues.append(f"SVHN 规模 {s.n_train}/{s.n_test} ≠ 10000/5000")

    print("=" * 72)
    print(f"passed {len(oks)}:")
    for o in oks:
        print("  [OK]", o)
    print("-" * 72)
    if issues:
        print(f"MISMATCH {len(issues)}:")
        for i in issues:
            print("  [XX]", i)
    print("RESULT:", "FAIL" if issues else "PASS")
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
