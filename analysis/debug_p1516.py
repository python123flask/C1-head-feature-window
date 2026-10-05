"""排障：谁的 wall=5930s？width0.5 的 A2=0/3 是头没释放还是探针太低？"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402

print("== wall times > 1500s ==")
for p in sorted(Path("results/runs").glob("*/summary.json")):
    d = json.load(open(p, encoding="utf-8"))
    w = d.get("wall_s") or 0
    if w > 1500:
        print(f"  {d.get('run_id')}  block={d.get('block')}  wall={w:.0f}s "
              f"status={d.get('status')}")

print("\n== width0.5 head/probe at end of B ==")
for s in (42, 123, 456):
    for L_ in (250, 4000):
        rid = f"P16_w0.5_L{L_}_s{s}"
        if not L.finished_ok(rid):
            print(f"  {rid}: not finished")
            continue
        e = L.end_b(rid)
        rows = L.phase_rows(rid, "B")
        hits = [r["phase_step"] for r in rows
                if r.get("acc_old_cur", 1) <= 0.5 and r.get("acc_old_probe", 0) >= 0.7]
        head_only = [r["phase_step"] for r in rows if r.get("acc_old_cur", 1) <= 0.5]
        probe_ok = [r["phase_step"] for r in rows if r.get("acc_old_probe", 0) >= 0.7]
        print(f"  {rid}: endB head={e.get('acc_old_cur'):.3f} probe={e.get('acc_old_probe'):.3f} "
              f"| head<=0.5 first={min(head_only) if head_only else None} "
              f"probe>=0.7 last={max(probe_ok) if probe_ok else None} "
              f"A2joint={min(hits) if hits else None} "
              f"| endA_acc={L.end_a(rid)}")

print("\n== width2.0 for contrast ==")
for s in (42, 123, 456):
    rid = f"P16_w2_L4000_s{s}"
    if L.finished_ok(rid):
        e = L.end_b(rid)
        print(f"  {rid}: endB head={e.get('acc_old_cur'):.3f} probe={e.get('acc_old_probe'):.3f} "
              f"endA_acc={L.end_a(rid)}")
