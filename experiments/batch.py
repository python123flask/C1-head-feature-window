"""批编排：生成冻结队列 grid_v1.json / QUEUE.json，并按块执行 run。

单写者：只有本 batch 进程写 ORCHESTRATION_STATE.json 与 QUEUE.json。

用法：
  python experiments/batch.py --make-grid
  python experiments/batch.py --run --blocks S-calib,S-smoke,P0-timing
  python experiments/batch.py --run --blocks P1-main --workers 3
  python experiments/batch.py --status
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "pilot"))

from utils import RUNS, read_json, write_json, append_log, LOGS  # noqa: E402

QUEUE_PATH = ROOT / "experiments" / "QUEUE.json"
STATE_PATH = ROOT / "experiments" / "ORCHESTRATION_STATE.json"
GRID_PATH = ROOT / "experiments" / "pilot" / "grid_v1.json"
BATCH_LOG = LOGS / "batch.log"

BLOCK_ORDER = ["S-exposure", "S-exposure2", "S-calib", "S-calib2", "S-smoke", "P0-timing",
               "P1-main", "P2-controls", "P3-fixedC", "P4-intervention", "P5-diagnostics",
               "P7-dose", "P6-sensor", "P8-resnet", "P9-eps", "P10-det",
               "P11-c100", "P12-fmnist", "P13-detseed", "P14-headreset"]

SEEDS5 = [42, 123, 456, 789, 1024]
SEEDS4 = [42, 123, 456, 789]
SEEDS3 = [42, 123, 456]
L_VALUES = [0, 250, 1000, 4000]

# 任务构造（校准后冻结）：旧任务取 CIFAR-10 中较易学习的 5 类，
# 使 end_A 测试精度满足 A1 质控门（>=0.85）并为探针留出余量；新任务为其余 5 类。
OLD_CLASSES = [0, 1, 6, 8, 9]     # airplane, automobile, frog, ship, truck
NEW_CLASSES = [2, 3, 4, 5, 7]     # bird, cat, deer, dog, horse
PER_CLASS_TRAIN = 1000

BASE = dict(
    model="smallcnn", head_dim=10, phase_A_steps=5000, phase_C_steps=2000,
    eval_interval=25, batch_size=128, lr=1e-3, weight_decay=1e-4,
    probe_train_max=2048, probe_test_max=2048, lam=1e-1, data_seed=0,
    # S-exposure2 校准结论：ε=1.0（真均匀目标）+ lr_B=1e-4
    #   - ε=0.9 目标仍依赖输入，A3（特征损伤）在所有 lr 下失败（probe 回升）
    #   - lr_B=1e-4 使 A2 窗口展宽到 75–325 步（21 个评测点），A3 drop=0.135
    eps_B=1.0, lr_B=1e-4, B_dense_steps=400, B_dense_interval=5,
    sham_steps=0, save_ckpt=True, permutation_test=False,
    freeze_features_C=False, old_probe_metrics=True, new_probe_metrics=True,
    old_classes=OLD_CLASSES, new_classes=NEW_CLASSES, per_class_train=PER_CLASS_TRAIN,
)


def cfg(**kw) -> dict:
    c = dict(BASE)
    c.update(kw)
    return c


def build_grid() -> dict:
    runs: list[dict] = []

    # ---- S-exposure2：暴露目标 × 过渡学习率 扫描（A=5000 全长，前 250 步密集评测）
    for eps in (1.0, 0.9):
        for lr_b in (1e-3, 3e-4, 1e-4):
            runs.append(cfg(
                run_id=f"S_exp2_eps{eps:g}_lr{lr_b:g}", block="S-exposure2",
                mode="main", shared="high", L=4000, eps_B=eps, lr_B=lr_b, seed=42,
                phase_C_steps=0, eval_interval=25, B_dense_steps=250,
                B_dense_interval=5, save_ckpt=True))

    # ---- S-exposure：暴露目标诊断（决定 Phase B 采用哪种"近均匀"暴露）
    for eps in (0.9, 1.0):
        runs.append(cfg(run_id=f"S_exposure_eps{eps:g}", block="S-exposure", mode="main",
                        shared="high", L=4000, eps_B=eps, seed=42,
                        phase_A_steps=3000, phase_C_steps=0, eval_interval=25,
                        save_ckpt=True))
    # 追加：250 步暴露（同 eps=1.0）用于看窗口内 A2 是否可达成
    runs.append(cfg(run_id="S_exposure_eps1_L250", block="S-exposure", mode="main",
                    shared="high", L=250, eps_B=1.0, seed=42,
                    phase_A_steps=3000, phase_C_steps=0, eval_interval=25,
                    save_ckpt=True))

    # ---- S-calib：lr 校准（A-only，不计入 86 步正式 run 预算）
    for lr in (1e-3, 3e-3, 3e-4):
        runs.append(cfg(run_id=f"S_calib_lr{lr:g}", block="S-calib", mode="main",
                        shared="high", L=0, seed=42, lr=lr, phase_C_steps=0,
                        eval_interval=50, save_ckpt=False, old_probe_metrics=True))

    # ---- S-smoke：管线冒烟（2000 步子集）
    runs.append(cfg(run_id="S_smoke", block="S-smoke", mode="main", shared="high",
                    L=250, seed=42, phase_A_steps=1200, phase_C_steps=500,
                    eval_interval=25, save_ckpt=True))

    # ---- P0-timing：完整协议资源实测
    runs.append(cfg(run_id="P0_timing", block="P0-timing", mode="main", shared="high",
                    L=4000, seed=42, eval_interval=25))

    # ---- P1-main：L × 共享度 × 5 seeds = 40
    for shared in ("high", "low"):
        for L in L_VALUES:
            for s in SEEDS5:
                runs.append(cfg(run_id=f"P1_main_{shared}_L{L}_s{s}", block="P1-main",
                                mode="main", shared=shared, L=L, seed=s))

    # ---- P2-controls：abrupt / direct / headonly × 5 seeds = 15（高共享）
    for ctrl in ("abrupt", "direct", "headonly"):
        for s in SEEDS5:
            runs.append(cfg(run_id=f"P2_{ctrl}_s{s}", block="P2-controls", mode=ctrl,
                            shared="high", L=0, seed=s,
                            freeze_features_C=(ctrl == "headonly")))

    # ---- P3-fixedC：等总预算对照（sham + L = 4000），3 水平 × 3 seeds = 9
    for L in (0, 250, 1000):
        for s in SEEDS3:
            runs.append(cfg(run_id=f"P3_fixedc_L{L}_s{s}", block="P3-fixedC", mode="main",
                            shared="high", L=L, sham_steps=4000 - L, seed=s))

    # ---- P4-intervention：交叉移植 2 组合 × 5 seeds = 10
    for s in SEEDS5:
        head_short = str(RUNS / f"P1_main_high_L250_s{s}" / "checkpoints" / "end_B.npz")
        head_long = str(RUNS / f"P1_main_high_L4000_s{s}" / "checkpoints" / "end_B.npz")
        runs.append(cfg(run_id=f"P4_headShort_featLong_s{s}", block="P4-intervention",
                        mode="transplant", shared="high", L=1000, seed=s,
                        head_from_ckpt=head_short, feat_from_ckpt=head_long,
                        requires=[head_short, head_long]))
        runs.append(cfg(run_id=f"P4_headLong_featShort_s{s}", block="P4-intervention",
                        mode="transplant", shared="high", L=1000, seed=s,
                        head_from_ckpt=head_long, feat_from_ckpt=head_short,
                        requires=[head_short, head_long]))

    # ---- P5-diagnostics：E3/E4 诊断 2 条件 × 4 seeds = 8
    #   arm1 ε=1.0 + 置换检验（E4）；arm2 ε=0 续训（E3：静态收缩对照）
    for eps, perm in ((1.0, True), (0.0, False)):
        for s in SEEDS4:
            runs.append(cfg(run_id=f"P5_diag_eps{eps:g}_s{s}", block="P5-diagnostics",
                            mode="main", shared="high", L=4000, eps_B=eps, seed=s,
                            permutation_test=perm))

    # ---- P7-dose：暴露强度剂量-响应（ε=0.9 输入相关平滑 vs ε=1.0），6 runs
    for L in (250, 4000):
        for s in SEEDS3:
            runs.append(cfg(run_id=f"P7_dose_eps0.9_L{L}_s{s}", block="P7-dose",
                            mode="main", shared="high", L=L, eps_B=0.9, seed=s))

    # ---- P8-resnet：计划 3.2.2 的 ResNet-18 扩展（11.2M 参数，验证机制随规模）
    for Lv in (0, 250, 4000):
        for s in SEEDS3:
            runs.append(cfg(run_id=f"P8_resnet_L{Lv}_s{s}", block="P8-resnet",
                            mode="main", shared="high", L=Lv, seed=s, model="resnet18",
                            eval_interval=25, B_dense_steps=400, B_dense_interval=5))

    # ---- P9-eps：ε 连续剂量（0.95 / 0.99），检验头释放对目标信息量的阈值行为
    for eps in (0.95, 0.99):
        for Lv in (250, 4000):
            for s in SEEDS3:
                runs.append(cfg(run_id=f"P9_eps{eps:g}_L{Lv}_s{s}", block="P9-eps",
                                mode="main", shared="high", L=Lv, eps_B=eps, seed=s))

    # ---- P10-det：确定性核复现（AdaptiveAvgPool2d→AvgPool2d 前向逐位等价 + cudnn.deterministic）
    #      seed 42/123 各跑 a/b 两份完全相同的 run（证明重跑噪声=0）；其余 seed 单份
    for s in (42, 123):
        for Lv in (0, 250, 4000):
            for rep in ("a", "b"):
                runs.append(cfg(run_id=f"P10_det_L{Lv}_s{s}_{rep}", block="P10-det",
                                mode="main", shared="high", L=Lv, seed=s,
                                deterministic=True, save_ckpt=False))
    for s in (456, 789, 1024):
        for Lv in (0, 250, 4000):
            runs.append(cfg(run_id=f"P10_det_L{Lv}_s{s}_a", block="P10-det",
                            mode="main", shared="high", L=Lv, seed=s,
                            deterministic=True, save_ckpt=False))

    # ---- P6-sensor：真实传感器漂移 3 条件 × 4 seeds = 12
    for L in (0, 500, 2000):
        for s in SEEDS4:
            runs.append(cfg(run_id=f"P6_sensor_L{L}_s{s}", block="P6-sensor", mode="sensor",
                            shared="sensor", L=L, seed=s, model="smallmlp", head_dim=6,
                            batch_size=128, eval_interval=25))

    # ---- S-calib2：CIFAR-100（25 类旧任务）Phase A 质量校准（formal 扩展前冻结阈值）
    for model_spec, steps in (("smallcnn", 5000), ("smallcnn", 10000),
                              ("resnet18", 5000), ("resnet18", 10000)):
        runs.append(cfg(run_id=f"S_calib2_{model_spec}_A{steps}", block="S-calib2",
                        mode="main", shared="high", L=0, seed=42, dataset="cifar100",
                        head_dim=25, model=model_spec, phase_A_steps=steps,
                        phase_C_steps=0, eval_interval=25, save_ckpt=False))

    # ---- P11-CIFAR100：外部效度（25 类旧 / 25 类新，同域高共享）
    for model_spec in ("smallcnn", "resnet18"):
        for Lv in (0, 250, 4000):
            for s in SEEDS3:
                runs.append(cfg(run_id=f"P11_c100_{model_spec}_L{Lv}_s{s}",
                                block="P11-c100", mode="main", shared="high", L=Lv,
                                seed=s, dataset="cifar100", head_dim=25, model=model_spec))

    # ---- P12-FMNIST：低共享臂换成 Fashion-MNIST（跨域新任务 2）
    for Lv in (0, 250, 4000):
        for s in SEEDS3:
            runs.append(cfg(run_id=f"P12_fmnist_L{Lv}_s{s}", block="P12-fmnist",
                            mode="main", shared="low", L=Lv, seed=s, head_dim=10,
                            low_dataset="fmnist"))

    # ---- P13-detseed：确定性平台补 5 个新 seed（终态主张的统计基础）
    for s in (2048, 3003, 4096, 5005, 6006):
        for Lv in (0, 250, 4000):
            runs.append(cfg(run_id=f"P13_detseed_L{Lv}_s{s}", block="P13-detseed",
                            mode="main", shared="high", L=Lv, seed=s,
                            deterministic=True, save_ckpt=False))

    # ---- P14-headreset：切换瞬间重置分类头（last-layer reset 基线，回应"为何不直接重置"）
    for s in SEEDS5:
        runs.append(cfg(run_id=f"P14_headreset_s{s}", block="P14-headreset",
                        mode="main", shared="high", L=0, seed=s,
                        reinit_head_after_A=True))

    ids = [r["run_id"] for r in runs]
    assert len(ids) == len(set(ids)), "duplicate run_id"
    return {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "phase_A_steps": 5000, "phase_C_steps": 2000, "eval_interval": 25,
            "label_smoothing_B": 0.9, "optimizer": "Adam(lr=1e-3, wd=1e-4)",
            "batch_size": 128, "head_dim": 10,
            "note": "评测间隔 25 步（计划 100 步与 ≥200 评测点互斥，按 DEVIATIONS.md 适配）",
        },
        "blocks": {b: [r["run_id"] for r in runs if r["block"] == b] for b in BLOCK_ORDER},
        "runs": runs,
    }


def cmd_make_grid(args):
    grid = build_grid()
    write_json(GRID_PATH, grid)
    queue = {
        "generated_at": grid["frozen_at"],
        "runs": grid["runs"],
        "pending": [r["run_id"] for r in grid["runs"]],
        "running": [], "done": [], "failed": [], "blocked": [],
    }
    write_json(QUEUE_PATH, queue)
    _event("grid_made", f"{len(grid['runs'])} runs")
    counts = {b: len(v) for b, v in grid["blocks"].items()}
    print(json.dumps(counts, indent=2))
    print("total:", len(grid["runs"]))


# ---------------------------------------------------------------- 状态
def _state(**kw):
    st = read_json(STATE_PATH) if STATE_PATH.exists() else {
        "status": "idle", "updated_at": None, "counters": {}, "events": []}
    st.update(kw)
    st["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(STATE_PATH, st)
    return st


def _event(event: str, detail=""):
    st = read_json(STATE_PATH) if STATE_PATH.exists() else {
        "status": "idle", "updated_at": None, "counters": {}, "events": []}
    st.setdefault("events", []).append(
        {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, "detail": detail})
    st["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(STATE_PATH, st)


def _write_queue(queue):
    write_json(QUEUE_PATH, queue)


# ---------------------------------------------------------------- 执行
def _worker(config: dict) -> dict:
    from sim import run_experiment
    for p in config.get("requires", []):
        if not Path(p).exists():
            return {"run_id": config["run_id"], "status": "blocked",
                    "error": f"missing dependency {p}"}
    return run_experiment(config)


def _finished_ok(run_id: str) -> bool:
    p = RUNS / run_id / "summary.json"
    if not p.exists():
        return False
    try:
        return read_json(p).get("status") == "succeeded"
    except Exception:
        return False


def _record(queue, run_id: str, status: str, ok: int, fail: int, blocked: int):
    if status == "succeeded":
        queue["done"].append(run_id)
        ok += 1
    elif status in ("blocked", "skipped"):
        queue.setdefault("blocked", []).append(run_id)
        blocked += 1
    else:
        queue["failed"].append(run_id)
        fail += 1
    if run_id in queue["running"]:
        queue["running"].remove(run_id)
    return ok, fail, blocked


def cmd_run(args):
    global QUEUE_PATH, STATE_PATH
    if args.state_id:
        QUEUE_PATH = QUEUE_PATH.with_name(f"QUEUE_{args.state_id}.json")
        STATE_PATH = STATE_PATH.with_name(f"ORCHESTRATION_STATE_{args.state_id}.json")
    canon = QUEUE_PATH.with_name("QUEUE.json")
    # 运行清单始终以冻结的 canonical QUEUE.json 为准；
    # 分片只维护自己的副本（QUEUE_wX.json）用于记录。
    src = canon if canon.exists() else QUEUE_PATH
    if not src.exists():
        raise SystemExit("QUEUE.json missing; run --make-grid first")
    queue = read_json(src)
    if QUEUE_PATH.exists() and QUEUE_PATH != src:
        prev = read_json(QUEUE_PATH)
        for key in ("done", "failed", "blocked", "running"):
            queue[key] = prev.get(key, [])
    blocks = [b for b in BLOCK_ORDER if not args.blocks or b in args.blocks.split(",")]
    runs = [r for r in queue["runs"] if r["block"] in blocks]
    if args.shard:
        k, n = (int(v) for v in args.shard.split("/"))
        runs = [r for i, r in enumerate(runs) if i % n == k]
    if args.limit:
        runs = runs[:args.limit]
    done_ids = {r["run_id"] for r in runs if _finished_ok(r["run_id"]) and not args.force}
    skipped = len(done_ids)
    runs = [r for r in runs if r["run_id"] not in done_ids]
    print(f"blocks={blocks} shard={args.shard or '-'} to_run={len(runs)} "
          f"skipped_done={skipped} workers={args.workers}", flush=True)
    if not runs:
        return

    _state(status="running", counters={
        **(read_json(STATE_PATH).get("counters") if STATE_PATH.exists() else {}),
        "planned": len(runs)})
    _event("batch_start", f"blocks={blocks} n={len(runs)} shard={args.shard or '-'}")

    queue["running"] = [r["run_id"] for r in runs]
    queue["pending"] = [i for i in queue["pending"] if i not in queue["running"]]
    _write_queue(queue)

    ok = fail = blocked = 0
    t0 = time.time()

    def handle(r, status):
        nonlocal ok, fail, blocked
        ok, fail, blocked = _record(queue, r["run_id"], status, ok, fail, blocked)
        _write_queue(queue)
        _event("run_done", f"{r['run_id']}={status} ok={ok} fail={fail}")
        print(f"[{ok + fail + blocked}/{len(runs)}] {r['run_id']} {status} "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    if args.workers <= 1:
        for r in runs:
            try:
                summ = _worker(r)
                status = summ.get("status", "succeeded")
            except Exception:
                status = "failed"
                append_log(BATCH_LOG, f"{r['run_id']} crash\n{traceback.format_exc()}")
            handle(r, status)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, r): r for r in runs}
            for fut in as_completed(futs):
                r = futs[fut]
                try:
                    summ = fut.result()
                    status = summ.get("status", "succeeded")
                except Exception:
                    status = "failed"
                    append_log(BATCH_LOG, f"{r['run_id']} worker crash\n{traceback.format_exc()}")
                handle(r, status)

    st = read_json(STATE_PATH)
    st.update(status="idle", counters={**st.get("counters", {}),
                                       "succeeded": ok, "failed": fail, "blocked": blocked,
                                       "wall_s": round(time.time() - t0, 1)})
    write_json(STATE_PATH, st)
    _event("batch_end", f"ok={ok} fail={fail} blocked={blocked}")
    print(f"done: ok={ok} fail={fail} blocked={blocked} wall={time.time() - t0:.0f}s",
          flush=True)


def cmd_status(args):
    q = read_json(QUEUE_PATH) if QUEUE_PATH.exists() else {}
    s = read_json(STATE_PATH) if STATE_PATH.exists() else {}
    print("state:", json.dumps({k: s.get(k) for k in ("status", "updated_at", "counters")},
                               ensure_ascii=False))
    print({k: len(q.get(k, [])) for k in ("pending", "running", "done", "failed", "blocked")})
    if q.get("failed"):
        print("failed:", q["failed"])


def main():
    global QUEUE_PATH, STATE_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-grid", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--blocks", default="")
    ap.add_argument("--workers", type=int, default=1,
                    help="1=顺序执行（沙箱内可用）；>1=进程池（需非受限环境）")
    ap.add_argument("--shard", default="", help="k/n 分片，多进程各跑一片")
    ap.add_argument("--state-id", default="", help="状态文件后缀（每写者独立）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.state_id:
        QUEUE_PATH = QUEUE_PATH.with_name(f"QUEUE_{args.state_id}.json")
        STATE_PATH = STATE_PATH.with_name(f"ORCHESTRATION_STATE_{args.state_id}.json")
    if args.make_grid:
        cmd_make_grid(args)
    elif args.run:
        cmd_run(args)
    elif args.status:
        cmd_status(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
