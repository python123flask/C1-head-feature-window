# PLAN.md — C1 · 可塑性恢复窗口（冻结计划的操作性摘要）

> 完整叙述版计划（含文献查新表、架构代码、时间规划等）由用户于 2026-10-04 提供，
> 原文保存在会话记录中；本文件保留**执行所需的冻结要素**，并作为
> `pilot/grid_v1.json`、`experiments/ORCHESTRATION_STATE*.json` 的同级载体。
> 任何相对计划的偏离都记录在 [`DEVIATIONS.md`](DEVIATIONS.md)；运行记录见 [`RECORD.md`](RECORD.md)。

- 状态：frozen_for_execution（冻结于 2026-10-04；运行层级 **pilot**，不得冒充 formal 或论文证据）
- 载体：本文件 + `pilot/grid_v1.json`（队列）+ `experiments/ORCHESTRATION_STATE*.json`（状态）
- 先导用途：(a) 测量资源，(b) 检验机制是否存在可分离迹象，(c) 决定 advance / modify / stop。

## 1. 冻结的主张与竞争解释

**primary_question**：持续训练中出现“有限时近均匀标签暴露”时，分类头去专化与特征受损
是否是可分离的两个过程，并因此改变后续新任务的适应后果？

**candidate_mechanism**：头（W2,b2）先复位/去专化、特征（φ）后发生信息损伤；
存在中间暴露窗口使后续适应收益不降反升。

**competing_explanations**

| id | 名称 | 排除方式 |
|---|---|---|
| E1 | 总更新量/预算 | 等预算对照（P3：sham 步数补足总步数） |
| E2 | 仅头释放 | headonly 对照（P2）+ 移植检验（P4） |
| E3 | 静态标签平滑收缩 | 诊断（P5）+ 剂量-响应（P7）：若头/特征同步单调收缩且无内部窗口 → 判为覆盖 |
| E4 | 测量伪影 | probe 用 train 拟合、test 评估；尺度控制与置换检验 |

**observable_outcomes**：头指标（acc_old_cur、margin_old、‖W2‖）、
特征指标（acc_old_probe、acc_new_probe、feat_drift）、
结果指标（new_task_NLL、new_task_acc、adaptation_speed=相位 C 前 1000 步）。

**claim_ceiling_before_formal_results**：最多“机制存在初步分离迹象 / 未显示分离”；
禁止“支持完整假设”“可用于论文主张”“创新已验证”“可投稿”。

## 2. 预注册决策门（先导专用）

- 读数：`results/runs/<run_id>/metrics.jsonl`、`results/runs/<run_id>/summary.json`
- 质控 A1：无 NaN/traceback；end_A 测试准确率中位 ≥ 0.85；metrics 行数 ≥ 200
- **A2 头先释放**：存在 L≥250 且某评测点 `acc_old_cur ≤ 0.50` 且 `acc_old_probe ≥ 0.70`
- **A3 特征后损伤**：L=4000 过渡末 `acc_old_probe` ≤ L=250 过渡末 − 0.10
- **A4 新任务后果非平凡**：各 L 与三对照（abrupt/direct/headonly）终态差异
  ≥0.05 NLL 或 ≥0.03 acc，且不能被等预算/仅头重训整体解释
- 方向至少 2/3 seeds 一致；`advance` 需 A1–A4 全部满足
- `modify_if`：A1 通过但 A2–A4 部分不满足 → 调整任务构造/尺度/指标后重测（rounds_v2，需留痕）
- `stop_if`：A1 两次修复仍失败；或高低共享下均无任何分离与差异迹象
- `next_claim_ceiling`：advance 后最多“候选机制在合成任务上存在可分离迹象”，
  formal 之前必须再经独立复核

## 3. 实验块与运行顺序（冻结）

| Order | Block | Runs（计划/实际） | 目的 |
|---|---|---|---|
| 0 | S-tests | 本机单测 | 梯度/目标构造/探针/调度/确定性 |
| 1 | S-exposure / S-exposure2 | 3 / 6（新增，见 DEVIATIONS D4） | 暴露形式与过渡学习率校准 |
| 2 | S-calib | 3（不计入正式） | 选 lr（旧任务 ≥0.85 测试精度） |
| 3 | S-smoke | 1 | 管线冒烟（summary=succeeded、指标非 NaN、acc>0.25） |
| 4 | P0-timing | 1 | wall / RSS / GPU memory 实测 |
| 5 | P1-main | 40 | L∈{0,250,1000,4000} × 共享度 × 5 seeds |
| 6 | P2-controls | 15 | abrupt / direct / headonly（E1/E2） |
| 7 | P3-fixedC | 9（计划 5，见 D7） | 固定总步数，排除“挤占效应” |
| 8 | P4-intervention | 10 | 交叉移植因果验证 |
| 9 | P5-diagnostics | 8 | 头/特征同步监测（E3/E4） |
| 10 | P7-dose | 6（新增） | ε=0.9 剂量-响应 |
| 11 | P6-sensor | 12（计划 8/12 不一致，见 D8） | UCI Gas Sensor 真实数据 |

正式对比合计 100 runs（计划 86）。

## 4. 冻结的训练协议（计划 3.4 节 + DEVIATIONS 中的校准选择）

```yaml
phase_A_steps: 5000        # 旧任务 CE(ε=0)
phase_B_steps: L           # L ∈ {0,250,1000,4000}；sham 模式补足到 4000（P3）
phase_C_steps: 2000        # 新任务 CE(ε=0)
label_smoothing_B: 1.0     # 校准选择：真均匀目标（计划原文 0.9，见 D4/E 节）
lr: 0.001                  # Phase A / C
lr_B: 0.0001               # Phase B（校准选择，见 D4/E 节）
optimizer: Adam(weight_decay=1e-4)
batch_size: 128
eval_interval: 25          # 计划 100 步与 ≥200 点互斥，见 D1
B_dense: 每 5 步（前 400 步）  # 解析 A2 窗口
device: cuda
```

## 5. 产物结构

```
experiments/pilot/   sim.py test_sim.py grid_v1.json data_loader.py models.py metrics.py utils.py
experiments/         QUEUE.json QUEUE_w*.json ORCHESTRATION_STATE*.json batch.py
results/runs/<id>/   config.json metrics.jsonl summary.json checkpoints/{end_A,mid_B,end_B,end_C}.npz
results/logs/        attempt.log batch.log data.log
analysis/            load.py decision_gate.py visualization.py make_tables.py paper_macros.py report.md figures/ tables/
pilot/               PLAN.md DEVIATIONS.md RECORD.md CODE_REVIEW.md preflight/resource-snapshot.json
paper/               main.tex results_macros.tex
```
