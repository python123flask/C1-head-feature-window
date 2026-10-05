"""一致性检查：同 seed 不同 L 的 Phase A 末状态是否一致（CUDA 非确定性量化）。"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402

SEEDS = [42, 123, 456, 789, 1024]


def main():
    print("seed | " + " | ".join(f"L{lv}: acc/probe" for lv in (0, 250, 1000, 4000))
          + " | acc range")
    spreads = []
    for s in SEEDS:
        cells, accs = [], []
        for lv in (0, 250, 1000, 4000):
            rid = f"P1_main_high_L{lv}_s{s}"
            a = L.end_a(rid)
            if a.get("acc") is None:
                cells.append("   -   ")
            else:
                cells.append(f"{a['acc']:.4f}/{a['probe']:.4f}")
                accs.append(a["acc"])
        if accs:
            r = max(accs) - min(accs)
            spreads.append(r)
            print(f"{s:4d} | " + " | ".join(cells) + f" | {r:.4f}")
        else:
            print(f"{s:4d} | " + " | ".join(cells))
    if spreads:
        print(f"\nmax within-seed end_A accuracy spread = {max(spreads):.4f} "
              f"(mean {statistics.mean(spreads):.4f})")


if __name__ == "__main__":
    main()
