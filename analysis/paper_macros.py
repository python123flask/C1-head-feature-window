"""生成 paper/results_macros.tex：把实测数值编译成 LaTeX 宏（论文正文引用）。"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "results_macros.tex"

SEEDS5 = [42, 123, 456, 789, 1024]
SEEDS4 = [42, 123, 456, 789]


def vals(ids, getter):
    out = []
    for rid in ids:
        if L.finished_ok(rid):
            v = getter(rid)
            if v is not None:
                out.append(v)
    return out


def m(xs, fmt=".3f"):
    return f"{statistics.mean(xs):{fmt}}" if xs else "0"


def sd(xs, fmt=".3f"):
    return f"{statistics.pstdev(xs):{fmt}}" if xs else "0"


def a2_window(shared="high", Lv=4000):
    firsts, lasts = [], []
    for s in SEEDS5:
        rid = f"P1_main_{shared}_L{Lv}_s{s}"
        hits = [x["phase_step"] for x in L.phase_rows(rid, "B")
                if x.get("acc_old_cur", 1) <= 0.5 and x.get("acc_old_probe", 0) >= 0.7]
        if hits:
            firsts.append(min(hits))
            lasts.append(max(hits))
    return firsts, lasts


def main():
    macros = []

    def add(name, value):
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def series(shared, key, src="final"):
        out = {}
        for Lv in (0, 250, 1000, 4000):
            ids = [f"P1_main_{shared}_L{Lv}_s{s}" for s in SEEDS5]
            if src == "final":
                v = vals(ids, lambda r: L.final_c(r).get(key))
            else:
                v = vals(ids, lambda r: L.end_b(r).get(key))
            out[Lv] = v
        return out

    for shared, tag in (("high", "Hi"), ("low", "Lo")):
        acc = series(shared, "new_task_acc")
        nll = series(shared, "new_task_NLL")
        spd = {}
        for Lv in (0, 250, 1000, 4000):
            ids = [f"P1_main_{shared}_L{Lv}_s{s}" for s in SEEDS5]
            spd[Lv] = vals(ids, lambda r: L.adaptation_speed(r, 1000).get("acc_mean"))
        for Lv, name in ((0, "Lzero"), (250, "LtwoF"), (1000, "LoneK"), (4000, "LfourK")):
            add(f"Acc{tag}{name}", m(acc[Lv]))
            add(f"Acc{tag}{name}Sd", sd(acc[Lv]))
            add(f"Nll{tag}{name}", m(nll[Lv]))
            add(f"Nll{tag}{name}Sd", sd(nll[Lv]))
            add(f"Spd{tag}{name}", m(spd[Lv]))

    # end_A / end_B 状态
    add("EndAAcc", m(vals([f"P1_main_high_L{Lv}_s{s}" for Lv in (0, 250, 1000, 4000)
                           for s in SEEDS5], lambda r: L.end_a(r).get("acc"))))
    add("EndAProbe", m(vals([f"P1_main_high_L{Lv}_s{s}" for Lv in (0, 250, 1000, 4000)
                             for s in SEEDS5], lambda r: L.end_a(r).get("probe"))))
    add("ProbeAtLtwoF", m(vals([f"P1_main_high_L250_s{s}" for s in SEEDS5],
                               lambda r: L.end_b(r).get("acc_old_probe"))))
    add("ProbeAtLfourK", m(vals([f"P1_main_high_L4000_s{s}" for s in SEEDS5],
                                lambda r: L.end_b(r).get("acc_old_probe"))))
    add("HeadAtLtwoF", m(vals([f"P1_main_high_L250_s{s}" for s in SEEDS5],
                              lambda r: L.end_b(r).get("acc_old_cur"))))
    add("HeadAtLfourK", m(vals([f"P1_main_high_L4000_s{s}" for s in SEEDS5],
                               lambda r: L.end_b(r).get("acc_old_cur"))))
    add("DropA3", m(vals([f"P1_main_high_L250_s{s}" for s in SEEDS5],
                         lambda r: L.end_b(r).get("acc_old_probe")) and
                    [a - b for a, b in zip(
                        vals([f"P1_main_high_L250_s{s}" for s in SEEDS5],
                             lambda r: L.end_b(r).get("acc_old_probe")),
                        vals([f"P1_main_high_L4000_s{s}" for s in SEEDS5],
                             lambda r: L.end_b(r).get("acc_old_probe")))] or [0]))

    # A2 窗口
    firsts, lasts = a2_window()
    if firsts:
        add("WinFirst", f"{min(firsts)}")
        add("WinLast", f"{max(lasts)}")
        add("WinSeeds", f"{len(firsts)}/5")
    else:
        add("WinFirst", "0"); add("WinLast", "0"); add("WinSeeds", "0/5")

    # 对照
    for name, tag in (("abrupt", "Abr"), ("direct", "Dir"), ("headonly", "Hnd")):
        ids = [f"P2_{name}_s{s}" for s in SEEDS5]
        add(f"Acc{tag}", m(vals(ids, lambda r: L.final_c(r).get("new_task_acc"))))
        add(f"Nll{tag}", m(vals(ids, lambda r: L.final_c(r).get("new_task_NLL"))))
    # 等预算
    for Lv, tag in ((0, "Lzero"), (250, "LtwoF"), (1000, "LoneK")):
        ids = [f"P3_fixedc_L{Lv}_s{s}" for s in (42, 123, 456)]
        add(f"AccFix{tag}", m(vals(ids, lambda r: L.final_c(r).get("new_task_acc"))))
        add(f"NllFix{tag}", m(vals(ids, lambda r: L.final_c(r).get("new_task_NLL"))))

    # 移植
    for name, tag in (("headShort_featLong", "SL"), ("headLong_featShort", "LS")):
        ids = [f"P4_{name}_s{s}" for s in SEEDS5]
        add(f"AccP4{tag}", m(vals(ids, lambda r: L.final_c(r).get("new_task_acc"))))
        add(f"NllP4{tag}", m(vals(ids, lambda r: L.final_c(r).get("new_task_NLL"))))

    # 剂量-响应
    ids = [f"P7_dose_eps0.9_L4000_s{s}" for s in (42, 123, 456)]
    add("ProbeEps09K", m(vals(ids, lambda r: L.end_b(r).get("acc_old_probe"))))
    add("AccEps09K", m(vals(ids, lambda r: L.final_c(r).get("new_task_acc"))))
    ids = [f"P5_diag_eps0_s{s}" for s in SEEDS4]
    add("ProbeEps0K", m(vals(ids, lambda r: L.end_b(r).get("acc_old_probe"))))
    add("AccEps0K", m(vals(ids, lambda r: L.final_c(r).get("new_task_acc"))))

    # 传感器
    for Lv, tag in ((0, "Lzero"), (500, "Lfive"), (2000, "LtwoK")):
        ids = [f"P6_sensor_L{Lv}_s{s}" for s in SEEDS4]
        add(f"AccSen{tag}", m(vals(ids, lambda r: L.final_c(r).get("new_task_acc"))))
        add(f"NllSen{tag}", m(vals(ids, lambda r: L.final_c(r).get("new_task_NLL"))))
        add(f"ProbeSen{tag}", m(vals(ids, lambda r: L.end_b(r).get("acc_old_probe"))))

    # ---- P8 ResNet-18 扩展
    for Lv, name in ((0, "Lzero"), (250, "LtwoF"), (4000, "LfourK")):
        ids = [f"P8_resnet_L{Lv}_s{s}" for s in (42, 123, 456)]
        add(f"AccRes{name}", m(vals(ids, lambda r: L.final_c(r).get("new_task_acc"))))
        add(f"NllRes{name}", m(vals(ids, lambda r: L.final_c(r).get("new_task_NLL"))))
        add(f"SpdRes{name}", m(vals(ids, lambda r: L.adaptation_speed(r, 1000).get("acc_mean"))))
        add(f"HeadRes{name}", m(vals(ids, lambda r: L.end_b(r).get("acc_old_cur"))))
        add(f"ProbeRes{name}", m(vals(ids, lambda r: L.end_b(r).get("acc_old_probe"))))
    a2_res = sum(1 for s in (42, 123, 456)
                 for Lv in (250, 4000)
                 if L.finished_ok(f"P8_resnet_L{Lv}_s{s}")
                 and any(x.get("acc_old_cur", 1) <= 0.5 and x.get("acc_old_probe", 0) >= 0.7
                         for x in L.phase_rows(f"P8_resnet_L{Lv}_s{s}", "B")))
    add("A2ResCount", f"{a2_res}/6")

    # ---- P9 ε 连续剂量（L=4000）
    for eps, tag in ((0.95, "E95"), (0.99, "E99")):
        ids = [f"P9_eps{eps:g}_L4000_s{s}" for s in (42, 123, 456)]
        add(f"Head{tag}", m(vals(ids, lambda r: L.end_b(r).get("acc_old_cur"))))
        add(f"Probe{tag}", m(vals(ids, lambda r: L.end_b(r).get("acc_old_probe"))))
        add(f"W2{tag}", m(vals(ids, lambda r: L.end_b(r).get("W2_norm"))))

    # ---- P10 确定性核复现
    max_delta = 0.0
    pairs_checked = 0
    for s in (42, 123):
        for Lv in (0, 250, 4000):
            ra = L.metrics(f"P10_det_L{Lv}_s{s}_a")
            rb = L.metrics(f"P10_det_L{Lv}_s{s}_b")
            if ra and rb and len(ra) == len(rb):
                pairs_checked += 1
                for x, y in zip(ra, rb):
                    for k in x:
                        if isinstance(x.get(k), float) and isinstance(y.get(k), float):
                            max_delta = max(max_delta, abs(x[k] - y[k]))
    add("DetPairs", str(pairs_checked))
    add("DetMaxDelta", f"{max_delta:.1e}")
    for Lv, name in ((0, "Lzero"), (250, "LtwoF"), (4000, "LfourK")):
        ids = [f"P10_det_L{Lv}_s{s}_a" for s in (42, 123, 456, 789, 1024)]
        add(f"AccDet{name}", m(vals(ids, lambda r: L.final_c(r).get("new_task_acc"))))
        add(f"NllDet{name}", m(vals(ids, lambda r: L.final_c(r).get("new_task_NLL"))))
        add(f"SpdDet{name}", m(vals(ids, lambda r: L.adaptation_speed(r, 1000).get("acc_mean"))))

    # ---- P11 CIFAR-100 / P12 Fashion-MNIST / P13 确定性 7 seeds
    def _a2_count(pattern, seeds, Ls=(250, 4000)):
        n = 0
        for s in seeds:
            for Lv in Ls:
                rid = pattern.format(L=Lv, s=s)
                if L.finished_ok(rid) and any(
                        x.get("acc_old_cur", 1) <= 0.5 and x.get("acc_old_probe", 0) >= 0.7
                        for x in L.phase_rows(rid, "B")):
                    n += 1
        return n

    for tag, pattern, seeds in (("CnnC100", "P11_c100_smallcnn_L{L}_s{s}", (42, 123, 456)),
                                ("RnC100", "P11_c100_resnet18_L{L}_s{s}", (42, 123, 456)),
                                ("Fmn", "P12_fmnist_L{L}_s{s}", (42, 123, 456))):
        for Lv, nm in ((0, "Lzero"), (250, "LtwoF"), (4000, "LfourK")):
            ids = [pattern.format(L=Lv, s=s) for s in seeds]
            add(f"Acc{tag}{nm}", m(vals(ids, lambda r: L.final_c(r).get("new_task_acc"))))
            add(f"Nll{tag}{nm}", m(vals(ids, lambda r: L.final_c(r).get("new_task_NLL"))))
            add(f"Spd{tag}{nm}", m(vals(ids, lambda r: L.adaptation_speed(r, 1000).get("acc_mean"))))
            add(f"Head{tag}{nm}", m(vals(ids, lambda r: L.end_b(r).get("acc_old_cur"))))
            add(f"Probe{tag}{nm}", m(vals(ids, lambda r: L.end_b(r).get("acc_old_probe"))))
        add(f"A2{tag}", f"{_a2_count(pattern, seeds)}/6")

    # 确定性平台（10 seeds = P10 的 5 + P13 的 5）汇总
    det_seeds = (42, 123, 456, 789, 1024, 2048, 3003, 4096, 5005, 6006)

    def _det_id(Lv, s):
        return (f"P10_det_L{Lv}_s{s}_a" if s in (42, 123, 456, 789, 1024)
                else f"P13_detseed_L{Lv}_s{s}")

    gains = []
    accs = {0: [], 250: [], 4000: []}
    nlls = {0: [], 250: [], 4000: []}
    for s in det_seeds:
        for Lv in (0, 250, 4000):
            rid = _det_id(Lv, s)
            if not L.finished_ok(rid):
                continue
            v = L.final_c(rid).get("new_task_acc")
            if v is not None:
                accs[Lv].append(v)
                nlls[Lv].append(L.final_c(rid).get("new_task_NLL"))
        a = L.adaptation_speed(_det_id(0, s), 1000).get("acc_mean")
        b = L.adaptation_speed(_det_id(250, s), 1000).get("acc_mean")
        if a is not None and b is not None:
            gains.append(b - a)
    add("DetSeeds", str(len(gains)))
    add("DetWindowGain", f"{statistics.mean(gains):.4f}" if gains else "0")
    add("DetWindowPos", f"{sum(1 for g in gains if g > 0)}/{len(gains)}")
    if all(accs[k] for k in accs):
        rng = max(statistics.mean(v) for v in accs.values()) - \
            min(statistics.mean(v) for v in accs.values())
        add("DetLRrange", f"{rng:.4f}")
        for Lv, nm in ((0, "Lzero"), (250, "LtwoF"), (4000, "LfourK")):
            add(f"AccDet10{nm}", f"{statistics.mean(accs[Lv]):.3f}")
            add(f"NllDet10{nm}", f"{statistics.mean(nlls[Lv]):.3f}")
    else:
        add("DetLRrange", "0")

    # 校准 run 的窗口（S-exposure2, ε=1.0, lr_B=1e-4）—— 校准段专用
    cal = L.phase_rows("S_exp2_eps1_lr0.0001", "B")
    cal_hits = [r["phase_step"] for r in cal
                if r.get("acc_old_cur", 1) <= 0.50 and r.get("acc_old_probe", 0) >= 0.70]
    add("CalWinFirst", str(min(cal_hits)) if cal_hits else "0")
    add("CalWinLast", str(max(cal_hits)) if cal_hits else "0")

    # A2 首现步（跨全部条件/seed 的最小与最大）—— 摘要与正文统一用这组
    firsts = []
    for r in L.grid_runs():
        if r.get("block") != "P1-main":
            continue
        rid = r["run_id"]
        if not L.finished_ok(rid):
            continue
        hits = [x["phase_step"] for x in L.phase_rows(rid, "B")
                if x.get("acc_old_cur", 1) <= 0.50 and x.get("acc_old_probe", 0) >= 0.70]
        if hits:
            firsts.append(min(hits))
    if firsts:
        add("A2FirstMin", str(min(firsts)))
        add("A2FirstMax", str(max(firsts)))
    else:
        add("A2FirstMin", "0")
        add("A2FirstMax", "0")

    # P14 头重置基线（含与各 L 的配对差）
    reset_ids = [f"P14_headreset_s{s}" for s in SEEDS5]
    add("AccReset", m(vals(reset_ids, lambda r: L.final_c(r).get("new_task_acc"))))
    add("NllReset", m(vals(reset_ids, lambda r: L.final_c(r).get("new_task_NLL"))))
    add("SpdReset", m(vals(reset_ids, lambda r: L.adaptation_speed(r, 1000).get("acc_mean"))))

    def paired_reset(base_ids):
        d_spd, d_acc, d_nll = [], [], []
        for s in SEEDS5:
            r1, r0 = f"P14_headreset_s{s}", base_ids.format(s=s)
            if not (L.finished_ok(r1) and L.finished_ok(r0)):
                continue
            c1, c0 = L.final_c(r1), L.final_c(r0)
            s1 = L.adaptation_speed(r1, 1000).get("acc_mean")
            s0 = L.adaptation_speed(r0, 1000).get("acc_mean")
            if None in (c1.get("new_task_acc"), c0.get("new_task_acc")):
                continue
            d_spd.append(s1 - s0)
            d_acc.append(c1["new_task_acc"] - c0["new_task_acc"])
            d_nll.append(c1["new_task_NLL"] - c0["new_task_NLL"])
        return d_spd, d_acc, d_nll

    for tag, base in (("Lzero", "P1_main_high_L0_s{s}"),
                      ("LtwoF", "P1_main_high_L250_s{s}"),
                      ("LfourK", "P1_main_high_L4000_s{s}")):
        d_spd, d_acc, d_nll = paired_reset(base)
        if d_spd:
            add(f"DSpdReset{tag}", f"{statistics.mean(d_spd):+.4f}")
            add(f"DSpdReset{tag}Pos", f"{sum(1 for x in d_spd if x > 0)}/{len(d_spd)}")
            add(f"DAccReset{tag}", f"{statistics.mean(d_acc):+.4f}")
            add(f"DNllReset{tag}", f"{statistics.mean(d_nll):+.4f}")

    # 资源（全部正式块：P0–P10）
    walls, rss, gpu = [], [], []
    for r in L.grid_runs():
        blk = (r.get("block") or "")
        if not blk.startswith("P"):
            continue
        s = L.summary(r["run_id"])
        if s.get("status") != "succeeded":
            continue
        if s.get("wall_s"):
            walls.append(s["wall_s"])
        if s.get("rss_peak_mb") and s["rss_peak_mb"] == s["rss_peak_mb"]:
            rss.append(s["rss_peak_mb"])
        if s.get("gpu_peak_mb"):
            gpu.append(s["gpu_peak_mb"])
    add("NumRuns", str(len(walls)))
    add("WallMean", f"{statistics.mean(walls):.0f}" if walls else "0")
    add("WallMax", f"{max(walls):.0f}" if walls else "0")
    add("PeakRss", f"{max(rss):.0f}" if rss else "0")
    add("PeakGpu", f"{max(gpu):.0f}" if gpu else "0")

    header = ("% auto-generated by analysis/paper_macros.py -- do not edit by hand\n")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(header + "\n".join(macros) + "\n", encoding="utf-8")
    print(f"-> {OUT} ({len(macros)} macros)")


if __name__ == "__main__":
    main()
