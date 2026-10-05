"""决策门判定脚本（计划 §2，先导专用）。

读取 results/runs/<run_id>/{metrics.jsonl,summary.json}，
输出 analysis/GATE.json + 控制台报告。

判据（阈值不放宽）：
  A1 质控：无 NaN/traceback；end_A 测试精度中位 >= 0.85；metrics 行数 >= 200
  A2 头先释放：存在 L>=250 的评测点 acc_old_cur<=0.50 且 acc_old_probe>=0.70
  A3 特征后损伤：过渡末 acc_old_probe(L=4000) <= acc_old_probe(L=250) - 0.10
  A4 新任务后果非平凡：各 L 与 abrupt/direct/headonly 终态差异 >=0.05 NLL 或 >=0.03 acc，
     且不能被等预算(P3)/仅头重训(P2 headonly)整体解释
  方向一致性：>= 2/3 seeds 同向
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load as L  # noqa: E402

OUT = Path(__file__).resolve().parent / "GATE.json"


def _finite(x):
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def qc() -> dict:
    ids = L.run_ids("P1-main") + L.run_ids("P2-controls") + L.run_ids("P3-fixedC") \
        + L.run_ids("P4-intervention") + L.run_ids("P5-diagnostics") + L.run_ids("P6-sensor") \
        + L.run_ids("P7-dose") + L.run_ids("P8-resnet") + L.run_ids("P9-eps") \
        + L.run_ids("P10-det") + L.run_ids("P11-c100") + L.run_ids("P12-fmnist") \
        + L.run_ids("P13-detseed") + L.run_ids("P14-headreset")
    done = [i for i in ids if L.finished_ok(i)]
    missing = [i for i in ids if not L.finished_ok(i)]

    def mode_of(rid: str) -> str:
        p = L.RUNS / rid / "config.json"
        return (L.read_json(p).get("mode") if p.exists() else "") or ""

    # 行数下限的适用范围：direct 无 Phase A/B（按设计只有 Phase C 的 80 行）、
    # transplant 只有 Phase C（80 行）——均由协议决定，见 CODE_REVIEW L3。
    def rows_required(rid: str) -> bool:
        return mode_of(rid) not in ("direct", "transplant")

    nan_runs, short_runs, end_a_accs = [], [], []
    for i in done:
        rows = L.metrics(i)
        if not rows:
            short_runs.append(i)
            continue
        bad = any((isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
                  for r in rows for v in r.values() if isinstance(v, float))
        if bad:
            nan_runs.append(i)
        if rows_required(i) and len(rows) < 200:
            short_runs.append(f"{i}({len(rows)})")
        a = L.end_a(i)
        if a.get("acc") is not None:
            end_a_accs.append(a["acc"])
    median_a = statistics.median(end_a_accs) if end_a_accs else float("nan")
    passed = (not nan_runs) and not short_runs and _finite(median_a) and median_a >= 0.85 \
        and len(missing) == 0
    return {"n_expected": len(ids), "n_done": len(done), "missing": missing,
            "nan_runs": nan_runs, "short_runs": short_runs,
            "end_A_median": median_a, "pass": passed}


def a2_gate() -> dict:
    """头先释放：逐 L 统计满足双阈值评测点的 seeds 数。"""
    out = {}
    for shared in ("high", "low"):
        for Lv in (250, 1000, 4000):
            ok_seeds, firsts = [], {}
            for s in (42, 123, 456, 789, 1024):
                rid = f"P1_main_{shared}_L{Lv}_s{s}"
                if not L.finished_ok(rid):
                    continue
                hits = [r for r in L.phase_rows(rid, "B")
                        if r.get("acc_old_cur", 1) <= 0.50 and r.get("acc_old_probe", 0) >= 0.70]
                if hits:
                    ok_seeds.append(s)
                    firsts[s] = hits[0]["phase_step"]
            n = 5
            out[f"{shared}_L{Lv}"] = {"seeds_hit": ok_seeds, "n": n,
                                      "first_step": firsts,
                                      "pass": len(ok_seeds) >= math.ceil(2 / 3 * n)}
    # 计划 A2：存在 L>=250 满足即算通过；同时记录覆盖的共享度（单共享度 → modify 提示）
    covered = sorted({k.split("_")[0] for k, v in out.items() if v["pass"]})
    return {"by_condition": out, "covered_sharedness": covered,
            "pass": any(v["pass"] for v in out.values())}


def a3_gate() -> dict:
    """A3：过渡末 probe(L=250) − probe(L=4000) 的配对中位 ≥ 0.10。
    两个共享度条件都必须满足（评审 #3：避免"任一条件通过"带来的双重机会）。"""
    out = {}
    all_present = True
    for shared in ("high", "low"):
        drops, pairs = [], []
        for s in (42, 123, 456, 789, 1024):
            r250 = f"P1_main_{shared}_L250_s{s}"
            r4000 = f"P1_main_{shared}_L4000_s{s}"
            if not (L.finished_ok(r250) and L.finished_ok(r4000)):
                continue
            p250 = L.end_b(r250).get("acc_old_probe")
            p4000 = L.end_b(r4000).get("acc_old_probe")
            if p250 is None or p4000 is None:
                continue
            drop = p250 - p4000
            drops.append(drop)
            pairs.append({"seed": s, "probe_L250": p250, "probe_L4000": p4000, "drop": drop})
        med = statistics.median(drops) if drops else float("nan")
        complete = len(drops) >= 4          # 评审 #4：配对数不足 → 不通过
        all_present = all_present and complete
        out[shared] = {"pairs": pairs, "median_drop": med, "n_pairs": len(drops),
                       "complete": complete,
                       "pass": bool(complete and _finite(med) and med >= 0.10),
                       "direction_consistent": (sum(d > 0 for d in drops) >= math.ceil(2 / 3 * len(drops)))
                       if drops else False}
    return {"by_condition": out, "pass": all(v["pass"] for v in out.values())}


def _final_metric(rid: str):
    c = L.final_c(rid)
    return c.get("new_task_acc"), c.get("new_task_NLL")


def a4_gate() -> dict:
    """新任务后果非平凡 + 排除 E1/E2。"""
    groups = {}
    for Lv in (0, 250, 1000, 4000):
        accs, nlls = [], []
        for s in (42, 123, 456, 789, 1024):
            rid = f"P1_main_high_L{Lv}_s{s}"
            if not L.finished_ok(rid):
                continue
            a, n = _final_metric(rid)
            if a is not None:
                accs.append(a)
                nlls.append(n)
        if accs:
            groups[f"L{Lv}"] = {"acc_mean": statistics.mean(accs),
                                "nll_mean": statistics.mean(nlls), "n": len(accs)}
    ctrl = {}
    for name, prefix in (("abrupt", "P2_abrupt_s"), ("direct", "P2_direct_s"),
                         ("headonly", "P2_headonly_s")):
        accs, nlls = [], []
        for s in (42, 123, 456, 789, 1024):
            rid = f"{prefix}{s}"
            if not L.finished_ok(rid):
                continue
            a, n = _final_metric(rid)
            if a is not None:
                accs.append(a)
                nlls.append(n)
        if accs:
            ctrl[name] = {"acc_mean": statistics.mean(accs),
                          "nll_mean": statistics.mean(nlls), "n": len(accs)}
    # 等预算对照（P3）
    fixed = {}
    for Lv in (0, 250, 1000):
        accs, nlls = [], []
        for s in (42, 123, 456):
            rid = f"P3_fixedc_L{Lv}_s{s}"
            if not L.finished_ok(rid):
                continue
            a, n = _final_metric(rid)
            if a is not None:
                accs.append(a)
                nlls.append(n)
        if accs:
            fixed[f"L{Lv}"] = {"acc_mean": statistics.mean(accs),
                               "nll_mean": statistics.mean(nlls), "n": len(accs)}

    allgroups = {**groups, **ctrl}
    acc_vals = [v["acc_mean"] for v in allgroups.values()]
    nll_vals = [v["nll_mean"] for v in allgroups.values()]
    acc_range = (max(acc_vals) - min(acc_vals)) if acc_vals else 0.0
    nll_range = (max(nll_vals) - min(nll_vals)) if nll_vals else 0.0
    nontrivial = (acc_range >= 0.03) or (nll_range >= 0.05)

    # ---- 完整性（评审 #4）：组内样本数不足 → 判为不通过，而不是静默跳过
    complete = True
    reasons = []
    for Lv in (0, 250, 1000, 4000):
        g = groups.get(f"L{Lv}")
        if not g or g["n"] < 4:
            complete = False
            reasons.append(f"P1 L{Lv} n={0 if not g else g['n']}<4")
    for name in ("abrupt", "direct", "headonly"):
        c = ctrl.get(name)
        if not c or c["n"] < 4:
            complete = False
            reasons.append(f"control {name} n={0 if not c else c['n']}<4")
    for Lv in (0, 250, 1000):
        f = fixed.get(f"L{Lv}")
        if not f or f["n"] < 3:
            complete = False
            reasons.append(f"P3 L{Lv} n={0 if not f else f['n']}<3")

    # ---- E1（评审 #1）：等预算对照必须真正进入判定。
    # 判定方式：在同一 L 集合 {0,250,1000} 上比较 P1 与 P3 的 L 跨度——
    # 若把总步数拉平后跨度坍缩到不足一半，则差异可由"预算"解释（E1 成立 → A4 不过）。
    def _rng(vals):
        return (max(vals) - min(vals)) if len(vals) >= 2 else None

    p1_L = [f"L{Lv}" for Lv in (0, 250, 1000)]
    p1_acc_r = _rng([groups[k]["acc_mean"] for k in p1_L if k in groups])
    p1_nll_r = _rng([groups[k]["nll_mean"] for k in p1_L if k in groups])
    fixed_acc = [v["acc_mean"] for v in fixed.values()]
    fixed_nll = [v["nll_mean"] for v in fixed.values()]
    fixed_acc_range = _rng(fixed_acc)
    fixed_nll_range = _rng(fixed_nll)

    def _collapsed(p1_r, p3_r):
        if p1_r is None or p3_r is None:
            return True                      # 无法判定 → 按"被解释"处理
        return p3_r < 0.5 * p1_r

    acc_explained_E1 = _collapsed(p1_acc_r, fixed_acc_range)
    nll_explained_E1 = _collapsed(p1_nll_r, fixed_nll_range)
    # 维度层面：至少存在一个维度既非平凡、又未被 E1 解释
    acc_signal = (acc_range >= 0.03) and (not acc_explained_E1)
    nll_signal = (nll_range >= 0.05) and (not nll_explained_E1)
    signal_and_not_E1 = acc_signal or nll_signal

    # ---- E2（评审 #2）：headonly 必须与**每一个** L 条件等效才算"整体解释"
    ho = ctrl.get("headonly")
    if ho and groups:
        headonly_covers = all(
            abs(g["acc_mean"] - ho["acc_mean"]) < 0.01 and
            abs(g["nll_mean"] - ho["nll_mean"]) < 0.01
            for g in groups.values())
    else:
        headonly_covers = False

    return {"P1_by_L": groups, "controls": ctrl, "P3_equal_budget": fixed,
            "complete": complete, "incomplete_reasons": reasons,
            "acc_range": acc_range, "nll_range": nll_range,
            "L_range_P1": {"acc": p1_acc_r, "nll": p1_nll_r},
            "nontrivial": nontrivial,
            "E1_budget_range": {"acc": fixed_acc_range, "nll": fixed_nll_range},
            "E1_explains": {"acc": acc_explained_E1, "nll": nll_explained_E1},
            "signal_not_explained_by_E1": signal_and_not_E1,
            "E2_headonly_covers": headonly_covers,
            "pass": bool(complete and signal_and_not_E1 and not headonly_covers)}


def consistency() -> dict:
    """方向一致性：A3 方向（特征损伤随 L 增大）>= 2/3 seeds。"""
    a3 = a3_gate()
    res = {}
    for k, v in a3["by_condition"].items():
        res[k] = v["direction_consistent"]
    return res


def main():
    report = {"qc": qc(), "A2": a2_gate(), "A3": a3_gate(), "A4": a4_gate(),
              "direction": consistency()}
    a1 = report["qc"]["pass"]
    a2 = report["A2"]["pass"]
    a3 = report["A3"]["pass"]
    a4 = report["A4"]["pass"]
    if a1 and a2 and a3 and a4:
        decision = "advance"
    elif a1:
        decision = "modify"
    else:
        decision = "stop_or_fix_qc"
    report["decision"] = decision
    report["claim_ceiling"] = ("最多：候选机制在合成任务上存在可分离迹象 / 未显示分离；"
                               "禁止论文级主张（先导层级）")
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print(f"A1 quality : {'PASS' if a1 else 'FAIL'}  "
          f"(end_A median={report['qc']['end_A_median']:.3f}, "
          f"done={report['qc']['n_done']}/{report['qc']['n_expected']}, "
          f"nan={len(report['qc']['nan_runs'])}, short={len(report['qc']['short_runs'])})")
    if report["qc"]["missing"]:
        print("   missing:", report["qc"]["missing"][:8],
              "..." if len(report["qc"]["missing"]) > 8 else "")
    print(f"A2 head    : {'PASS' if a2 else 'FAIL'} "
          f"(covered sharedness: {report['A2'].get('covered_sharedness')})")
    for k, v in report["A2"]["by_condition"].items():
        print(f"   {k:14s} seeds_hit={v['seeds_hit']} first_step={v['first_step']}")
    print(f"A3 feature : {'PASS' if a3 else 'FAIL'} (both sharedness conditions required)")
    for k, v in report["A3"]["by_condition"].items():
        print(f"   {k:14s} median_drop={v['median_drop']:.3f} n_pairs={v['n_pairs']} "
              f"complete={v['complete']} "
              f"pairs={[(p['seed'], round(p['drop'], 3)) for p in v['pairs']]}")
    a4r = report["A4"]
    print(f"A4 outcome : {'PASS' if a4 else 'FAIL'}  "
          f"acc_range={a4r['acc_range']:.3f} nll_range={a4r['nll_range']:.3f} "
          f"complete={a4r['complete']}")
    print(f"   E1 equal-budget: P1 L-range acc={a4r['L_range_P1']['acc']} / "
          f"nll={a4r['L_range_P1']['nll']}  vs  P3 L-range "
          f"acc={a4r['E1_budget_range']['acc']} / nll={a4r['E1_budget_range']['nll']}  "
          f"-> explains={a4r['E1_explains']}")
    print(f"   E2 headonly_covers_all_L={a4r['E2_headonly_covers']}  "
          f"signal_not_explained_by_E1={a4r['signal_not_explained_by_E1']}")
    if a4r["incomplete_reasons"]:
        print("   incomplete:", a4r["incomplete_reasons"])
    for k, v in report["A4"]["P1_by_L"].items():
        print(f"   {k:6s} acc={v['acc_mean']:.3f} nll={v['nll_mean']:.3f} (n={v['n']})")
    for k, v in report["A4"]["controls"].items():
        print(f"   {k:9s} acc={v['acc_mean']:.3f} nll={v['nll_mean']:.3f} (n={v['n']})")
    for k, v in report["A4"]["P3_equal_budget"].items():
        print(f"   P3 {k:6s} acc={v['acc_mean']:.3f} nll={v['nll_mean']:.3f} (n={v['n']})")
    print("-" * 72)
    print(f"DECISION: {decision}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
