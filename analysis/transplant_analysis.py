"""P4 交叉移植的 2×2 因子分解：头效应 / 特征效应 / 交互（配对到 seed）。

单元格来源：
  y(H_short, F_short) = P1 L=250（原生）
  y(H_long,   F_long) = P1 L=4000（原生）
  y(H_short, F_long)  = P4 headShort_featLong
  y(H_long,   F_short) = P4 headLong_featShort
指标：终态（Phase C 最后一点）与窗口均值（Phase C 前 1000 步）。
噪声底线来自 analysis/rerun_noise.py（同协议重跑）。
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402

SEEDS = [42, 123, 456, 789, 1024]


def cell(seed: int, which: str):
    rid = {"HsFs": f"P1_main_high_L250_s{seed}",
           "HlFl": f"P1_main_high_L4000_s{seed}",
           "HsFl": f"P4_headShort_featLong_s{seed}",
           "HlFs": f"P4_headLong_featShort_s{seed}"}[which]
    term = L.final_c(rid)
    win = L.adaptation_speed(rid, 1000)
    return {
        "terminal_acc": term.get("new_task_acc"),
        "terminal_nll": term.get("new_task_NLL"),
        "window_acc": win.get("acc_mean"),
        "window_nll": win.get("nll_mean"),
    }


def main():
    metrics = ["terminal_acc", "terminal_nll", "window_acc", "window_nll"]
    per_metric = {k: {"head": [], "feat": [], "inter": []} for k in metrics}
    cells = {c: {k: [] for k in metrics} for c in ("HsFs", "HsFl", "HlFs", "HlFl")}

    for s in SEEDS:
        v = {}
        ok = True
        for c in ("HsFs", "HsFl", "HlFs", "HlFl"):
            d = cell(s, c)
            if any(d[k] is None for k in metrics):
                ok = False
                break
            v[c] = d
            for k in metrics:
                cells[c][k].append(d[k])
        if not ok:
            continue
        for k in metrics:
            hs = 0.5 * (v["HsFs"][k] + v["HsFl"][k])
            hl = 0.5 * (v["HlFs"][k] + v["HlFl"][k])
            fs = 0.5 * (v["HsFs"][k] + v["HlFs"][k])
            fl = 0.5 * (v["HsFl"][k] + v["HlFl"][k])
            inter = (v["HsFl"][k] - v["HsFs"][k]) - (v["HlFl"][k] - v["HlFs"][k])
            per_metric[k]["head"].append(hs - hl)        # 短头 − 长头
            per_metric[k]["feat"].append(fl - fs)        # 长特征(损伤) − 短特征(完好)
            per_metric[k]["inter"].append(inter)

    print("=== 2x2 factorial (paired over seeds) ===")
    for k in metrics:
        print(f"\n-- {k} --")
        for c in ("HsFs", "HsFl", "HlFs", "HlFl"):
            vals = cells[c][k]
            if vals:
                print(f"   {c}: {statistics.mean(vals):.4f} ± {statistics.pstdev(vals):.4f} "
                      f"(n={len(vals)})")
        for eff in ("head", "feat", "inter"):
            vals = per_metric[k][eff]
            if not vals:
                continue
            mu = statistics.mean(vals)
            sd = statistics.pstdev(vals)
            n_pos = sum(1 for x in vals if x > 0)
            print(f"   {eff:5s} effect = {mu:+.4f} ± {sd:.4f}  "
                  f"(positive in {n_pos}/{len(vals)} seeds)")

    # 方向一致性解读
    print("\n读法：head 效应>0 → 短暴露的头更好（不管配哪个特征）；")
    print("      feat 效应>0 → 长暴露的特征反而更好（与'损伤有害'相反）；")
    print("      两者都接近 0 → 终态差异处于重跑噪声底线内。")


if __name__ == "__main__":
    main()
