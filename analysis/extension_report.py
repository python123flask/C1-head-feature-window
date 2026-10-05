"""扩展块报告：P8 ResNet-18（规模）、P9 ε 连续剂量、P10 确定性核复现。"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402

SEEDS3 = [42, 123, 456]
SEEDS5 = [42, 123, 456, 789, 1024]


def _m(vals):
    return f"{statistics.mean(vals):.4f}" if vals else "-"


def _sd(vals):
    return f"{statistics.pstdev(vals):.4f}" if vals else "-"


def _vals(ids, fn):
    out = []
    for rid in ids:
        if L.finished_ok(rid):
            v = fn(rid)
            if v is not None:
                out.append(v)
    return out


def a2_hits(rid):
    return any(r.get("acc_old_cur", 1) <= 0.50 and r.get("acc_old_probe", 0) >= 0.70
               for r in L.phase_rows(rid, "B"))


def first_a2(rid):
    hits = [r["phase_step"] for r in L.phase_rows(rid, "B")
            if r.get("acc_old_cur", 1) <= 0.50 and r.get("acc_old_probe", 0) >= 0.70]
    return min(hits) if hits else None


def block_p8():
    print("=" * 78)
    print("P8 — ResNet-18 (11.2M) 扩展：机制是否随规模成立")
    print(f"{'L':>5s} | {'head@endB':>16s} | {'probe@endB':>16s} | "
          f"{'A2 (n/3)':>8s} | {'first':>6s} | {'speed1000':>16s} | {'end acc':>16s}")
    for Lv in (0, 250, 4000):
        ids = [f"P8_resnet_L{Lv}_s{s}" for s in SEEDS3]
        eh = _vals(ids, lambda r: L.end_b(r).get("acc_old_cur"))
        ep = _vals(ids, lambda r: L.end_b(r).get("acc_old_probe"))
        sp = _vals(ids, lambda r: L.adaptation_speed(r, 1000).get("acc_mean"))
        ac = _vals(ids, lambda r: L.final_c(r).get("new_task_acc"))
        hits = [r for r in ids if L.finished_ok(r) and a2_hits(r)]
        firsts = [first_a2(r) for r in hits if first_a2(r) is not None]
        print(f"{Lv:5d} | {_m(eh):>8s}±{_sd(eh):>6s} | {_m(ep):>8s}±{_sd(ep):>6s} | "
              f"{len(hits):3d}/{len([i for i in ids if L.finished_ok(i)])} | "
              f"{(min(firsts) if firsts else '-'):>6} | "
              f"{_m(sp):>8s}±{_sd(sp):>6s} | {_m(ac):>8s}±{_sd(ac):>6s}")
    # A3 配对
    drops = []
    for s in SEEDS3:
        a = L.end_b(f"P8_resnet_L250_s{s}").get("acc_old_probe")
        b = L.end_b(f"P8_resnet_L4000_s{s}").get("acc_old_probe")
        if a is not None and b is not None:
            drops.append(a - b)
    if drops:
        print(f"A3 drop (L250-L4000 probe): {statistics.mean(drops):.3f} "
              f"± {statistics.pstdev(drops):.3f} (n={len(drops)}, "
              f"median={statistics.median(drops):.3f})")


def block_p9():
    print("=" * 78)
    print("P9 — ε 连续剂量（L=250 / 4000）：头释放与特征损伤随目标信息量")
    print(f"{'eps':>6s} | {'L':>5s} | {'head@endB':>16s} | {'probe@endB':>16s} | "
          f"{'end acc':>16s} | {'speed1000':>16s} | {'A2':>5s}")
    rows = [(1.0, "P1_main_high_L{L}_s{s}", SEEDS5), (0.95, "P9_eps0.95_L{L}_s{s}", SEEDS3),
            (0.99, "P9_eps0.99_L{L}_s{s}", SEEDS3),
            (0.9, "P7_dose_eps0.9_L{L}_s{s}", SEEDS3)]
    for eps, pat, seeds in rows:
        for Lv in (250, 4000):
            if eps == 0.9:
                ids = [pat.format(L=Lv, s=s) for s in seeds]
            else:
                ids = [pat.format(L=Lv, s=s) for s in seeds]
            eh = _vals(ids, lambda r: L.end_b(r).get("acc_old_cur"))
            ep = _vals(ids, lambda r: L.end_b(r).get("acc_old_probe"))
            ac = _vals(ids, lambda r: L.final_c(r).get("new_task_acc"))
            sp = _vals(ids, lambda r: L.adaptation_speed(r, 1000).get("acc_mean"))
            hits = [r for r in ids if L.finished_ok(r) and a2_hits(r)]
            print(f"{eps:6.2f} | {Lv:5d} | {_m(eh):>8s}±{_sd(eh):>6s} | "
                  f"{_m(ep):>8s}±{_sd(ep):>6s} | {_m(ac):>8s}±{_sd(ac):>6s} | "
                  f"{_m(sp):>8s}±{_sd(sp):>6s} | {len(hits):3d}")


def block_p10():
    print("=" * 78)
    print("P10 — 确定性核复现：重跑噪声 = 0，终态 L 效应可判读")
    # 1) 完全相同配置的两份 run 是否逐位一致
    for s in (42, 123):
        for Lv in (0, 250, 4000):
            a = L.metrics(f"P10_det_L{Lv}_s{s}_a")
            b = L.metrics(f"P10_det_L{Lv}_s{s}_b")
            if not a or not b or len(a) != len(b):
                print(f"  pair s{s} L{Lv}: missing/unequal rows "
                      f"({len(a)}/{len(b)})")
                continue
            diffs = []
            for ra, rb in zip(a, b):
                for k in ra:
                    if isinstance(ra.get(k), float) and isinstance(rb.get(k), float):
                        diffs.append(abs(ra[k] - rb.get(k, float("nan"))))
            print(f"  pair s{s} L{Lv}: rows={len(a)} max|Δ| = {max(diffs):.3e} "
                  f"-> {'IDENTICAL' if max(diffs) == 0 else 'NON-DETERMINISTIC'}")
    # 2) 确定性下的 L 效应（配对，同一 seed，零进程噪声）
    print("\n  terminal new-task metrics under deterministic kernels (paired by seed):")
    print(f"  {'seed':>5s} | {'L':>5s} | {'acc':>8s} | {'NLL':>8s} | "
          f"{'speed1000':>10s} | {'endA acc':>8s}")
    acc_by_L = {}
    nll_by_L = {}
    for s in SEEDS5:
        for Lv in (0, 250, 4000):
            rid = f"P10_det_L{Lv}_s{s}_a"
            if not L.finished_ok(rid):
                continue
            c = L.final_c(rid)
            a = L.end_a(rid)
            sp = L.adaptation_speed(rid, 1000).get("acc_mean")
            print(f"  {s:5d} | {Lv:5d} | {c.get('new_task_acc', 0):8.4f} | "
                  f"{c.get('new_task_NLL', 0):8.4f} | {sp or 0:10.4f} | "
                  f"{a.get('acc', 0):8.4f}")
            acc_by_L.setdefault(Lv, []).append(c.get("new_task_acc"))
            nll_by_L.setdefault(Lv, []).append(c.get("new_task_NLL"))
    if acc_by_L:
        print("  mean over seeds:")
        for Lv in sorted(acc_by_L):
            print(f"    L={Lv:5d}  acc {_m(acc_by_L[Lv])}±{_sd(acc_by_L[Lv])}   "
                  f"NLL {_m(nll_by_L[Lv])}±{_sd(nll_by_L[Lv])}  (n={len(acc_by_L[Lv])})")
        acc_rng = max(_m_ (v) for v in acc_by_L.values()) - min(_m_(v) for v in acc_by_L.values())
        nll_rng = max(_m_(v) for v in nll_by_L.values()) - min(_m_(v) for v in nll_by_L.values())
        print(f"  terminal L-range under determinism: acc={acc_rng:.4f} NLL={nll_rng:.4f}")


def _m_(vals):
    return statistics.mean(vals) if vals else float("nan")


def block_p11():
    print("=" * 96)
    print("P11 — CIFAR-100 (25 旧 / 25 新类，官方划分)：外部效度")
    hdr = (f"{'model':>9s} | {'L':>5s} | {'endA':>7s} | {'head@endB':>15s} | "
           f"{'probe@endB':>15s} | {'A2':>5s} | {'speed1000':>15s} | "
           f"{'end acc':>15s} | {'end NLL':>15s}")
    print(hdr)
    for model_spec in ("smallcnn", "resnet18"):
        for Lv in (0, 250, 4000):
            ids = [f"P11_c100_{model_spec}_L{Lv}_s{s}" for s in (42, 123, 456)]
            ea = _vals(ids, lambda r: L.end_a(r).get("acc"))
            eh = _vals(ids, lambda r: L.end_b(r).get("acc_old_cur"))
            ep = _vals(ids, lambda r: L.end_b(r).get("acc_old_probe"))
            sp = _vals(ids, lambda r: L.adaptation_speed(r, 1000).get("acc_mean"))
            ac = _vals(ids, lambda r: L.final_c(r).get("new_task_acc"))
            nl = _vals(ids, lambda r: L.final_c(r).get("new_task_NLL"))
            a2 = sum(1 for r in ids if L.finished_ok(r) and a2_hits(r))
            n_done = sum(1 for r in ids if L.finished_ok(r))
            print(f"{model_spec:>9s} | {Lv:5d} | {_m(ea):>7s} | {_m(eh):>7s}±{_sd(eh):>6s} | "
                  f"{_m(ep):>7s}±{_sd(ep):>6s} | {a2:2d}/{n_done} | "
                  f"{_m(sp):>7s}±{_sd(sp):>6s} | {_m(ac):>7s}±{_sd(ac):>6s} | "
                  f"{_m(nl):>7s}±{_sd(nl):>6s}")
    # A3 配对（SmallCNN / ResNet 各自）
    for model_spec in ("smallcnn", "resnet18"):
        drops = []
        for s in (42, 123, 456):
            a = L.end_b(f"P11_c100_{model_spec}_L250_s{s}").get("acc_old_probe")
            b = L.end_b(f"P11_c100_{model_spec}_L4000_s{s}").get("acc_old_probe")
            if a is not None and b is not None:
                drops.append(a - b)
        if drops:
            print(f"  A3-type drop {model_spec}: mean {statistics.mean(drops):+.3f} "
                  f"median {statistics.median(drops):+.3f} (n={len(drops)})")


def block_p12():
    print("=" * 96)
    print("P12 — 低共享臂第二伙伴：CIFAR-10 → Fashion-MNIST")
    print(f"{'L':>5s} | {'head@endB':>15s} | {'probe@endB':>15s} | {'A2':>5s} | "
          f"{'speed1000':>15s} | {'end acc':>15s} | {'end NLL':>15s}")
    for Lv in (0, 250, 4000):
        ids = [f"P12_fmnist_L{Lv}_s{s}" for s in (42, 123, 456)]
        eh = _vals(ids, lambda r: L.end_b(r).get("acc_old_cur"))
        ep = _vals(ids, lambda r: L.end_b(r).get("acc_old_probe"))
        sp = _vals(ids, lambda r: L.adaptation_speed(r, 1000).get("acc_mean"))
        ac = _vals(ids, lambda r: L.final_c(r).get("new_task_acc"))
        nl = _vals(ids, lambda r: L.final_c(r).get("new_task_NLL"))
        a2 = sum(1 for r in ids if L.finished_ok(r) and a2_hits(r))
        n_done = sum(1 for r in ids if L.finished_ok(r))
        print(f"{Lv:5d} | {_m(eh):>7s}±{_sd(eh):>6s} | {_m(ep):>7s}±{_sd(ep):>6s} | "
              f"{a2:2d}/{n_done} | {_m(sp):>7s}±{_sd(sp):>6s} | "
              f"{_m(ac):>7s}±{_sd(ac):>6s} | {_m(nl):>7s}±{_sd(nl):>6s}")


def block_p13():
    print("=" * 96)
    print("P13 — 确定性平台补 seed（P10 5 seeds + P13 5 seeds = 10 seeds）")
    # 确定性同配置对（P10 的 6 对）
    print("  same-config pairs (must be bitwise identical):")
    n_ident = n_pairs = 0
    for s in (42, 123):
        for Lv in (0, 250, 4000):
            a = L.metrics(f"P10_det_L{Lv}_s{s}_a")
            b = L.metrics(f"P10_det_L{Lv}_s{s}_b")
            if not a or not b or len(a) != len(b):
                continue
            n_pairs += 1
            d = max(abs(x.get(k, 0) - y.get(k, 0)) for x, y in zip(a, b)
                    for k in x if isinstance(x.get(k), float))
            n_ident += (d == 0)
    print(f"    {n_ident}/{n_pairs} pairs bitwise identical")

    seeds = (42, 123, 456, 789, 1024, 2048, 3003, 4096, 5005, 6006)

    def rid(Lv, s):
        return (f"P10_det_L{Lv}_s{s}_a" if s in (42, 123, 456, 789, 1024)
                else f"P13_detseed_L{Lv}_s{s}")

    print(f"  {'seed':>6s} | " + " | ".join(f"L{lv:<8d}" for lv in (0, 250, 4000))
          + " | speed1000 gain (L250-L0)")
    accs = {0: [], 250: [], 4000: []}
    gains = []
    for s in seeds:
        cells, spd = [], []
        for Lv in (0, 250, 4000):
            r = rid(Lv, s)
            c = L.final_c(r) if L.finished_ok(r) else {}
            v = c.get("new_task_acc")
            cells.append(f"{v:.4f}" if v is not None else "   -   ")
            sp = L.adaptation_speed(r, 1000).get("acc_mean") if L.finished_ok(r) else None
            spd.append(sp)
            if v is not None:
                accs[Lv].append(v)
        g = (spd[1] - spd[0]) if (spd[1] is not None and spd[0] is not None) else None
        if g is not None:
            gains.append(g)
        print(f"  {s:6d} | " + " | ".join(cells)
              + f" | {('+%.4f' % g) if g is not None else '-':>12s}")
    if all(accs[k] for k in accs):
        print("  mean±std over seeds:")
        for Lv in (0, 250, 4000):
            print(f"    L={Lv:5d}  acc {_m(accs[Lv])}±{_sd(accs[Lv])} (n={len(accs[Lv])})")
        rng_a = max(_mu(v) for v in accs.values()) - min(_mu(v) for v in accs.values())
        print(f"    terminal L-range under determinism: acc={rng_a:.4f}")
    if gains:
        pos = sum(1 for g in gains if g > 0)
        print(f"  window gain (speed1000, L250 vs L0): mean {statistics.mean(gains):+.4f}, "
              f"positive in {pos}/{len(gains)} seeds")


def _mu(vals):
    return statistics.mean(vals) if vals else float("nan")


if __name__ == "__main__":
    which = sys.argv[1:] or ["p8", "p9", "p10", "p11", "p12", "p13"]
    for name, fn in (("p8", block_p8), ("p9", block_p9), ("p10", block_p10),
                     ("p11", block_p11), ("p12", block_p12), ("p13", block_p13)):
        if name in which:
            fn()
