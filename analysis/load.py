"""分析公共库：读取 run 产物、汇总指标、支持决策门与可视化。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
QUEUE = ROOT / "experiments" / "QUEUE.json"
GRID = ROOT / "experiments" / "pilot" / "grid_v1.json"


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def grid_runs() -> list[dict]:
    if GRID.exists():
        return read_json(GRID)["runs"]
    if QUEUE.exists():
        return read_json(QUEUE)["runs"]
    return []


def run_ids(block: str | None = None) -> list[str]:
    runs = grid_runs()
    if block:
        runs = [r for r in runs if r["block"] == block]
    return [r["run_id"] for r in runs]


def metrics(run_id: str) -> list[dict]:
    p = RUNS / run_id / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summary(run_id: str) -> dict:
    p = RUNS / run_id / "summary.json"
    return read_json(p) if p.exists() else {}


def finished_ok(run_id: str) -> bool:
    return summary(run_id).get("status") == "succeeded"


def phase_rows(run_id: str, phase: str) -> list[dict]:
    return [r for r in metrics(run_id) if r.get("phase") == phase]


def last_of(rows: list[dict], key: str):
    for r in reversed(rows):
        if key in r:
            return r[key]
    return None


def end_b(run_id: str) -> dict:
    rows = phase_rows(run_id, "B")
    if not rows:
        return {}
    r = rows[-1]
    return {k: r.get(k) for k in ("acc_old_probe", "acc_old_cur", "W2_norm",
                                  "feat_drift", "margin_old")}


def end_a(run_id: str) -> dict:
    rows = phase_rows(run_id, "A")
    if not rows:
        return {}
    r = rows[-1]
    return {"acc": r.get("acc_old_cur"), "probe": r.get("acc_old_probe")}


def final_c(run_id: str) -> dict:
    rows = phase_rows(run_id, "C")
    if not rows:
        return {}
    r = rows[-1]
    return {k: r.get(k) for k in ("new_task_acc", "new_task_NLL", "acc_new_probe",
                                  "acc_old_probe", "acc_old_cur", "W2_norm", "feat_drift")}


def adaptation_speed(run_id: str, window: int = 1000) -> dict:
    """Phase C 前 window 步的适应速度：末值 + 平均值（AUC/window）。"""
    rows = [r for r in phase_rows(run_id, "C") if r.get("phase_step", 0) <= window]
    if not rows:
        return {}
    accs = [r["new_task_acc"] for r in rows if "new_task_acc" in r]
    nlls = [r["new_task_NLL"] for r in rows if "new_task_NLL" in r]
    return {
        "acc_at_window": accs[-1] if accs else None,
        "acc_mean": float(np.mean(accs)) if accs else None,
        "nll_at_window": nlls[-1] if nlls else None,
        "nll_mean": float(np.mean(nlls)) if nlls else None,
        "n_points": len(rows),
    }


def table(rows: list[list], header: list[str]) -> str:
    """简易等宽文本表。"""
    cols = list(zip(*([str(c) for c in r] for r in rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]
    head = " | ".join(h.ljust(w) for h, w in zip(header, widths))
    sep = "-+-".join("-" * w for w in widths)
    body = [" | ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join([head, sep] + body)
