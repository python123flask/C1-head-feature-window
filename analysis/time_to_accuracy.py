"""分析专用（只读，不改论文）：time-to-accuracy（达到固定阈值所需步数）。

用途：把"适应更快"这个瞬态收益换算成一个一眼能懂的指标，回答
"快但终点一样，意义何在"。
"""
import json
import statistics
import sys
from pathlib import Path

SEEDS = (42, 123, 456, 789, 1024)
THRESHOLDS = (0.60, 0.65, 0.68)


def rows(run_id):
    p = Path("results") / "runs" / run_id / "metrics.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def tta(run_id, thr):
    c = [r for r in rows(run_id) if r.get("phase") == "C"
         and r.get("new_task_acc") is not None and r["new_task_acc"] >= thr]
    return min((r["phase_step"] for r in c), default=None)


def agg(rids, thr):
    v = [tta(r, thr) for r in rids]
    got = [x for x in v if x is not None]
    miss = len(v) - len(got)
    if not got:
        return None, miss
    return statistics.mean(got), miss


conds = {
    "P1 L=0    ": [f"P1_main_high_L0_s{s}" for s in SEEDS],
    "P1 L=250  ": [f"P1_main_high_L250_s{s}" for s in SEEDS],
    "P1 L=1000 ": [f"P1_main_high_L1000_s{s}" for s in SEEDS],
    "P1 L=4000 ": [f"P1_main_high_L4000_s{s}" for s in SEEDS],
    "P14 reset ": [f"P14_headreset_s{s}" for s in SEEDS],
    "P1 L=0 low ": [f"P1_main_low_L0_s{s}" for s in SEEDS],
    "P1 L=250 low": [f"P1_main_low_L250_s{s}" for s in SEEDS],
}

print("time-to-accuracy：新任务达到阈值的首步（mean over seeds；'x' = 从未达到）")
hdr = "condition     " + "".join(f"acc>={t:<6}" for t in THRESHOLDS)
print(hdr)
for name, rids in conds.items():
    line = f"{name} "
    for thr in THRESHOLDS:
        m, miss = agg(rids, thr)
        line += f"  {('x' if m is None else str(int(round(m))))}({'5-miss' if miss else '5/5'})  "
    print(line)

# 配对：L250 vs L0 的 TTA 提前量
print("\n配对 TTA 提前量（L0 减 L250，正 = L250 更快）")
for thr in THRESHOLDS:
    gains = []
    for s in SEEDS:
        a = tta(f"P1_main_high_L0_s{s}", thr)
        b = tta(f"P1_main_high_L250_s{s}", thr)
        if a is not None and b is not None:
            gains.append(a - b)
    if gains:
        pos = sum(1 for g in gains if g > 0)
        print(f"  acc>={thr}: mean={statistics.mean(gains):+7.1f} steps "
              f"({pos}/{len(gains)} seeds faster), values={gains}")
