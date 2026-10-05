"""窗口效应分析：不同 L 在 Phase C 各时间窗内的配对表现（相对 L=0）。"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402

SEEDS5 = [42, 123, 456, 789, 1024]
WINDOWS = [50, 100, 250, 500, 1000, 1500, 2000]


def window_metric(run_id: str, upto: int, key: str):
    rows = [r for r in L.phase_rows(run_id, "C")
            if r.get("phase_step", 0) <= upto and key in r]
    if not rows:
        return None
    return statistics.mean(r[key] for r in rows)


def paired(shared: str, upto: int, key: str):
    base, deltas = {}, {}
    for s in SEEDS5:
        rid0 = f"P1_main_{shared}_L0_s{s}"
        v0 = window_metric(rid0, upto, key)
        if v0 is None:
            continue
        base[s] = v0
        for Lv in (250, 1000, 4000):
            rid = f"P1_main_{shared}_L{Lv}_s{s}"
            v = window_metric(rid, upto, key)
            if v is not None:
                deltas.setdefault(Lv, []).append(v - v0)
    return base, deltas


def report(shared: str):
    print(f"\n=== {shared} share : mean metric over phase-C window (paired vs L=0) ===")
    print(f"{'window':>7s} | {'L0 base':>17s} | " + " | ".join(
        f"L{Lv:<16d}" for Lv in (250, 1000, 4000)))
    for key, unit in (("new_task_acc", "acc"), ("new_task_NLL", "nll")):
        print(f"--- {unit} ---")
        for upto in WINDOWS:
            base, deltas = paired(shared, upto, key)
            if not base:
                continue
            cells = []
            for Lv in (250, 1000, 4000):
                d = deltas.get(Lv, [])
                if d:
                    star = "*" if (statistics.mean(d) < -1e-9 if unit == "nll"
                                   else statistics.mean(d) > 1e-9) else " "
                    cells.append(f"{statistics.mean(d):+.3f}{star} (n={len(d)})")
                else:
                    cells.append("   -   ")
            bmean = statistics.mean(base.values())
            print(f"  C<={upto:4d} | {bmean:.3f} (n={len(base)})   | " + " | ".join(cells))
    print("  (* = direction favouring exposure)")


def direction_consistency(shared: str, upto: int = 1000):
    out = {}
    for Lv in (250, 1000, 4000):
        wins = 0
        for s in SEEDS5:
            a = window_metric(f"P1_main_{shared}_L0_s{s}", upto, "new_task_acc")
            b = window_metric(f"P1_main_{shared}_L{Lv}_s{s}", upto, "new_task_acc")
            if a is not None and b is not None and b > a:
                wins += 1
        out[Lv] = wins
    return out


if __name__ == "__main__":
    for shared in (sys.argv[1:] or ["high", "low"]):
        report(shared)
        print("  seeds where L>0 beats L=0 (acc, C<=1000):",
              direction_consistency(shared), "of 5")
