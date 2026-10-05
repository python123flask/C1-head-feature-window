"""S-exposure 暴露方式诊断报告：判断 Phase B 应采用哪种"近均匀"目标。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402

KEYS = [1, 25, 50, 100, 150, 250, 500, 1000, 2000, 3000, 4000]


def report(run_id: str) -> dict:
    rows = L.phase_rows(run_id, "B")
    a = L.end_a(run_id)
    print(f"=== {run_id}   end_A acc={a.get('acc')} probe={a.get('probe')}  "
          f"B_points={len(rows)}")
    print("   step | head_acc | margin | probe  | W2norm | drift")
    for k in KEYS:
        cand = [r for r in rows if r.get("phase_step") == k]
        if not cand:
            continue
        r = cand[0]
        print(f"   {k:4d} | {r.get('acc_old_cur', float('nan')):8.3f} | "
              f"{r.get('margin_old', float('nan')):6.3f} | "
              f"{r.get('acc_old_probe', float('nan')):6.3f} | "
              f"{r.get('W2_norm', float('nan')):7.2f} | "
              f"{r.get('feat_drift', float('nan')):6.3f}")
    # A2 判据：是否存在 head<=0.50 且 probe>=0.70 的点
    a2 = [r for r in rows if r.get("acc_old_cur", 1) <= 0.50
          and r.get("acc_old_probe", 0) >= 0.70]
    print(f"  A2-like points (head<=0.50 & probe>=0.70): {len(a2)}"
          + (f" first at step {a2[0]['phase_step']}" if a2 else ""))
    if rows:
        print(f"  final B: head={rows[-1].get('acc_old_cur'):.3f} "
              f"probe={rows[-1].get('acc_old_probe'):.3f} "
              f"margin={rows[-1].get('margin_old'):.3f} "
              f"W2={rows[-1].get('W2_norm'):.2f} "
              f"drift={rows[-1].get('feat_drift'):.3f}")
    return {"run_id": run_id, "a2_points": len(a2),
            "final_head": rows[-1].get("acc_old_cur") if rows else None,
            "final_probe": rows[-1].get("acc_old_probe") if rows else None}


def main():
    for rid in ["S_exposure_eps0.9", "S_exposure_eps1", "S_exposure_eps1_L250"]:
        if L.summary(rid):
            report(rid)
        else:
            print(f"=== {rid}: missing")
    # A3 近似：L=4000 过渡末 probe vs L=250 过渡末 probe（同一 seed）
    b1 = L.end_b("S_exposure_eps1")
    b2 = L.end_b("S_exposure_eps1_L250")
    if b1 and b2:
        drop = (b2.get("acc_old_probe") or 0) - (b1.get("acc_old_probe") or 0)
        print(f"\nA3 check (eps=1.0, seed42): probe(L250)={b2.get('acc_old_probe'):.3f} "
              f"probe(L4000)={b1.get('acc_old_probe'):.3f} drop={drop:.3f} "
              f"(需要 >= 0.10)")


if __name__ == "__main__":
    main()
