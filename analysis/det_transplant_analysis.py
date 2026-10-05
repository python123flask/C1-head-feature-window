"""P15 确定性平台移植（档B）：与 P4（非确定性平台）并排打印 2×2 因子效应。

回应审稿质疑："特征主效应处于重跑噪声边缘"——在确定性平台上，
源 run 与移植 run 的过程噪声为 0（重跑位级一致），效应只剩 seed 方差。
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402

SEEDS = [42, 123, 456, 789, 1024]

PLATFORMS = {
    "P4 platform (baseline, process noise present)": {
        "HsFs": "P1_main_high_L250_s{s}",
        "HlFl": "P1_main_high_L4000_s{s}",
        "HsFl": "P4_headShort_featLong_s{s}",
        "HlFs": "P4_headLong_featShort_s{s}",
    },
    "P15 deterministic platform (process noise = 0)": {
        "HsFs": "P15_detsrc_L250_s{s}",
        "HlFl": "P15_detsrc_L4000_s{s}",
        "HsFl": "P15_detHeadShort_featLong_s{s}",
        "HlFs": "P15_detHeadLong_featShort_s{s}",
    },
}


def cells(tmpl, seed):
    out = {}
    for c, t in tmpl.items():
        rid = t.format(s=seed)
        if not L.finished_ok(rid):
            return None
        term = L.final_c(rid)
        if term.get("new_task_acc") is None or term.get("new_task_NLL") is None:
            return None
        out[c] = {"acc": term["new_task_acc"], "nll": term["new_task_NLL"]}
    return out


def effects(v):
    h = 0.5 * (v["HsFs"]["nll"] + v["HsFl"]["nll"]) - 0.5 * (v["HlFs"]["nll"] + v["HlFl"]["nll"])
    f = 0.5 * (v["HsFl"]["nll"] + v["HlFl"]["nll"]) - 0.5 * (v["HsFs"]["nll"] + v["HlFs"]["nll"])
    ha = 0.5 * (v["HsFs"]["acc"] + v["HsFl"]["acc"]) - 0.5 * (v["HlFs"]["acc"] + v["HlFl"]["acc"])
    fa = 0.5 * (v["HsFl"]["acc"] + v["HlFl"]["acc"]) - 0.5 * (v["HsFs"]["acc"] + v["HlFs"]["acc"])
    return h, f, ha, fa


def report(title, tmpl):
    heads, feats, head_acc, feat_acc = [], [], [], []
    for s in SEEDS:
        v = cells(tmpl, s)
        if v is None:
            continue
        h, f, ha, fa = effects(v)
        heads.append(h)
        feats.append(f)
        head_acc.append(ha)
        feat_acc.append(fa)
    if not feats:
        print(f"\n=== {title}: no complete seeds ===")
        return
    mu, sd_ = statistics.mean(feats), statistics.pstdev(feats)
    d = abs(mu / sd_) if sd_ > 0 else float("inf")
    print(f"\n=== {title} (n={len(feats)}) ===")
    print(f"  feat effect NLL = {mu:+.4f} ± {sd_:.4f}   |d| = {d:.2f}   "
          f"negative (long-exposed features better NLL) in "
          f"{sum(1 for x in feats if x < 0)}/{len(feats)}")
    hs = statistics.pstdev(heads)
    print(f"  head effect NLL = {statistics.mean(heads):+.4f} ± {hs:.4f}"
          + (f"   |d| = {abs(statistics.mean(heads) / hs):.2f}" if hs > 0 else ""))
    print(f"  feat effect acc = {statistics.mean(feat_acc):+.4f} | "
          f"head effect acc = {statistics.mean(head_acc):+.4f}")
    n = len(feats)
    if all(x < 0 for x in feats):
        print(f"  one-sided sign test (all {n} negative): p = 1/{2 ** n} = {0.5 ** n:.3f}")


def main():
    for title, tmpl in PLATFORMS.items():
        report(title, tmpl)


if __name__ == "__main__":
    main()
