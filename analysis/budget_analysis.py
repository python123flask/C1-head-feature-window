"""适应预算分析：把"窗口"翻译成可主张的实践命题。

对每个适应预算 B（Phase C 前 B 步的窗口均值），计算各 L 相对 L=0 的配对增益，
按 seed 统计方向一致性。数据全部来自已有 metrics（无需新 run）。
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402

SEEDS5 = [42, 123, 456, 789, 1024]
BUDGETS = [50, 100, 250, 500, 1000, 1500, 2000]


def window_mean(run_id: str, upto: int, key: str):
    rows = [r for r in L.phase_rows(run_id, "C")
            if r.get("phase_step", 0) <= upto and key in r]
    if not rows:
        return None
    return statistics.mean(r[key] for r in rows)


def point_at(run_id: str, step: int, key: str):
    """**恰好**在预算步数处的读数（端点，非窗口均值）——用于对照。"""
    rows = [r for r in L.phase_rows(run_id, "C") if r.get("phase_step") == step]
    if not rows or key not in rows[0]:
        return None
    return rows[0][key]


def deltas(shared: str, B: int, key: str, Lv: int, mode: str = "mean"):
    fn = window_mean if mode == "mean" else point_at
    out = []
    for s in SEEDS5:
        a = fn(f"P1_main_{shared}_L0_s{s}", B, key)
        b = fn(f"P1_main_{shared}_L{Lv}_s{s}", B, key)
        if a is not None and b is not None:
            out.append(b - a)
    return out


def endpoint_table():
    print("\n" + "=" * 84)
    print("端点读数（恰好在预算步 B 处，非窗口均值）：acc 增益 vs L=0")
    for shared in ("high", "low"):
        print(f"  -- {shared} --")
        for B in BUDGETS:
            cells = []
            for lv in (250, 1000, 4000):
                d = deltas(shared, B, "new_task_acc", lv, mode="point")
                if not d:
                    cells.append("    -    ")
                    continue
                mu = statistics.mean(d)
                pos = sum(1 for x in d if x > 0)
                cells.append(f"{mu:+.4f} {pos}/5")
            print(f"    B={B:5d} " + " | ".join(f"L{lv}:{c}" for lv, c in zip((250, 1000, 4000), cells)))


def report():
    for shared in ("high", "low"):
        print("=" * 84)
        print(f"{shared} share — paired gain vs L=0 at adaptation budget B "
              f"(accuracy, mean over seeds, '+' = exposure helps)")
        hdr = "  B   |" + "".join(f"  L={lv:<9d}" for lv in (250, 1000, 4000))
        print(hdr)
        for B in BUDGETS:
            cells = []
            for lv in (250, 1000, 4000):
                d = deltas(shared, B, "new_task_acc", lv)
                if not d:
                    cells.append("   -   ")
                    continue
                mu = statistics.mean(d)
                pos = sum(1 for x in d if x > 0)
                star = "*" if mu > 0 else " "
                cells.append(f"{mu:+.4f}{star} {pos}/5")
            print(f"  {B:4d}|" + "".join(f"  {c:>13s}" for c in cells))
        print("  ('*' = mean gain > 0; 'k/5' = seeds with positive gain)")
        print("\n  NLL gain (negative = exposure helps):")
        print(hdr)
        for B in BUDGETS:
            cells = []
            for lv in (250, 1000, 4000):
                d = deltas(shared, B, "new_task_NLL", lv)
                if not d:
                    cells.append("   -   ")
                    continue
                mu = statistics.mean(d)
                pos = sum(1 for x in d if x < 0)
                star = "*" if mu < 0 else " "
                cells.append(f"{mu:+.4f}{star} {pos}/5")
            print(f"  {B:4d}|" + "".join(f"  {c:>13s}" for c in cells))


def json_payload():
    out = {}
    for shared in ("high", "low"):
        out[shared] = {}
        for B in BUDGETS:
            out[shared][B] = {}
            for lv in (250, 1000, 4000):
                d = deltas(shared, B, "new_task_acc", lv)
                dn = deltas(shared, B, "new_task_NLL", lv)
                out[shared][B][lv] = {
                    "acc_mean": statistics.mean(d) if d else None,
                    "acc_pos_seeds": sum(1 for x in d if x > 0) if d else 0,
                    "nll_mean": statistics.mean(dn) if dn else None,
                    "nll_neg_seeds": sum(1 for x in dn if x < 0) if dn else 0,
                    "n": len(d),
                }
    return out


if __name__ == "__main__":
    report()
    endpoint_table()
    import json
    Path("analysis/budget_analysis.json").write_text(
        json.dumps(json_payload(), indent=2), encoding="utf-8")
    print("\n-> analysis/budget_analysis.json")
