"""重跑噪声底线：P1-L0 与 P2-abrupt 协议完全相同，差异 = 纯重跑不确定性。"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402

SEEDS = [42, 123, 456, 789, 1024]


def main():
    diffs_a, diffs_p, diffs_acc, diffs_nll = [], [], [], []
    for s in SEEDS:
        a = L.end_a(f"P1_main_high_L0_s{s}")
        b = L.end_a(f"P2_abrupt_s{s}")
        ca, cb = L.final_c(f"P1_main_high_L0_s{s}"), L.final_c(f"P2_abrupt_s{s}")
        if not (a.get("acc") is not None and b.get("acc") is not None):
            continue
        da = abs(a["acc"] - b["acc"])
        dp = abs((a.get("probe") or 0) - (b.get("probe") or 0))
        diffs_a.append(da)
        diffs_p.append(dp)
        if ca.get("new_task_acc") is not None and cb.get("new_task_acc") is not None:
            diffs_acc.append(abs(ca["new_task_acc"] - cb["new_task_acc"]))
            diffs_nll.append(abs(ca["new_task_NLL"] - cb["new_task_NLL"]))
    if not diffs_a:
        print("no paired runs yet")
        return
    print(f"identical-protocol rerun pairs: {len(diffs_a)}")
    print(f"  |d end_A acc|      mean={statistics.mean(diffs_a):.4f} max={max(diffs_a):.4f}")
    print(f"  |d end_A probe|    mean={statistics.mean(diffs_p):.4f} max={max(diffs_p):.4f}")
    if diffs_acc:
        print(f"  |d terminal acc|   mean={statistics.mean(diffs_acc):.4f} max={max(diffs_acc):.4f}")
        print(f"  |d terminal NLL|   mean={statistics.mean(diffs_nll):.4f} max={max(diffs_nll):.4f}")


if __name__ == "__main__":
    main()
