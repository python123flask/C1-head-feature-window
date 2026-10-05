"""S-exposure2 扫描报告：A2/A3 可达性随 (ε, lr_B) 的变化。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402


def head_cross(rows, thr=0.50):
    for r in rows:
        if r.get("acc_old_cur", 1) <= thr:
            return r["phase_step"]
    return None


def probe_cross(rows, thr=0.70):
    for r in rows:
        if r.get("acc_old_probe", 1) < thr:
            return r["phase_step"]
    return None


def a2_points(rows):
    return [r for r in rows if r.get("acc_old_cur", 1) <= 0.50
            and r.get("acc_old_probe", 0) >= 0.70]


def probe_at(rows, step):
    cand = [r for r in rows if r.get("phase_step") == step]
    return cand[0].get("acc_old_probe") if cand else None


def main():
    ids = [r["run_id"] for r in L.grid_runs() if r["block"] == "S-exposure2"]
    print(f"{'run_id':26s} | {'A_acc':>5s} {'A_prb':>5s} | {'head<=.5':>8s} "
          f"{'probe<.7':>8s} | {'A2pts':>5s} {'first':>5s} | {'prb@250':>7s} "
          f"{'prb@4000':>8s} {'drop':>6s} | {'W2@4000':>7s}")
    for rid in ids:
        if not L.summary(rid):
            print(f"{rid:26s} | MISSING")
            continue
        rows = L.phase_rows(rid, "B")
        a = L.end_a(rid)
        a2 = a2_points(rows)
        hc, pc = head_cross(rows), probe_cross(rows)
        p250, p4000 = probe_at(rows, 250), probe_at(rows, 4000)
        w2 = rows[-1].get("W2_norm") if rows else None
        drop = (p250 - p4000) if (p250 is not None and p4000 is not None) else None
        print(f"{rid:26s} | {a.get('acc') or 0:5.3f} {a.get('probe') or 0:5.3f} | "
              f"{str(hc):>8s} {str(pc):>8s} | {len(a2):5d} "
              f"{(a2[0]['phase_step'] if a2 else '-'):>5} | "
              f"{(p250 if p250 is not None else float('nan')):7.3f} "
              f"{(p4000 if p4000 is not None else float('nan')):8.3f} "
              f"{(drop if drop is not None else float('nan')):6.3f} | "
              f"{(w2 or float('nan')):7.2f}")
        # 打印早期轨迹（前 250 步每 5 步）便于观察窗口
        early = [r for r in rows if r.get("phase_step", 0) <= 100
                 and r.get("phase_step", 0) % 10 == 0]
        if early:
            traj = " ".join(f"{r['phase_step']}:{r.get('acc_old_cur', 0):.2f}/"
                            f"{r.get('acc_old_probe', 0):.2f}" for r in early)
            print(f"   early step:head/probe -> {traj}")


if __name__ == "__main__":
    main()
