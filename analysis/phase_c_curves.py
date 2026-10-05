"""Phase C 早期适应曲线对比：同 seed 不同 L 的 new_task_acc/NLL 随 C 步数。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402

STEPS = [25, 50, 100, 200, 400, 700, 1000, 1500, 2000]


def main():
    seeds = [int(s) for s in (sys.argv[1:] or ["42"])]
    for s in seeds:
        print(f"=== seed {s} (high share) ===")
        print("   C_step | " + " | ".join(f"L{Lv:<5d}" for Lv in (0, 250, 1000, 4000)))
        for st in STEPS:
            cells = []
            for Lv in (0, 250, 1000, 4000):
                rid = f"P1_main_high_L{Lv}_s{s}"
                rows = [r for r in L.phase_rows(rid, "C") if r.get("phase_step") == st]
                cells.append(f"{rows[0]['new_task_acc']:.3f}" if rows else "  -  ")
            print(f"  {st:7d} | " + " | ".join(cells))
        print("      NLL | " + " | ".join(f"L{Lv:<5d}" for Lv in (0, 250, 1000, 4000)))
        for st in STEPS:
            cells = []
            for Lv in (0, 250, 1000, 4000):
                rid = f"P1_main_high_L{Lv}_s{s}"
                rows = [r for r in L.phase_rows(rid, "C") if r.get("phase_step") == st]
                cells.append(f"{rows[0]['new_task_NLL']:.3f}" if rows else "  -  ")
            print(f"  {st:7d} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
