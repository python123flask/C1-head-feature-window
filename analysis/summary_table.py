"""快速汇总若干 run 的关键读数（end_A / end_B / 终态）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402


def row(rid: str) -> dict:
    s = L.summary(rid)
    if not s:
        return {"run_id": rid, "status": "MISSING"}
    a, b, c = L.end_a(rid), L.end_b(rid), L.final_c(rid)
    sp = L.adaptation_speed(rid)
    return {
        "run_id": rid, "status": s.get("status"), "wall": s.get("wall_s"),
        "pts": s.get("final", {}).get("n_points"),
        "A_acc": a.get("acc"), "A_probe": a.get("probe"),
        "B_head": b.get("acc_old_cur"), "B_probe": b.get("acc_old_probe"),
        "B_W2": b.get("W2_norm"), "B_drift": b.get("feat_drift"),
        "C_acc": c.get("new_task_acc"), "C_nll": c.get("new_task_NLL"),
        "C_probe_old": c.get("acc_old_probe"),
        "speed1000_acc": sp.get("acc_at_window"),
        "rss_mb": s.get("rss_peak_mb"), "gpu_mb": s.get("gpu_peak_mb"),
    }


def main():
    ids = sys.argv[1:] or L.run_ids()
    rows = [row(r) for r in ids]
    cols = ["run_id", "status", "wall", "pts", "A_acc", "A_probe", "B_head",
            "B_probe", "B_W2", "B_drift", "C_acc", "C_nll", "speed1000_acc",
            "rss_mb", "gpu_mb"]
    print(L.table([[("" if r.get(c) is None else
                     (round(r[c], 3) if isinstance(r[c], float) else r[c]))
                    for c in cols] for r in rows], cols))


if __name__ == "__main__":
    main()
