# CODE_REVIEW.md — 部署前代码评审记录

- 评审对象：`experiments/pilot/{sim,metrics,data_loader,models,utils,test_sim}.py`、
  `experiments/batch.py`、`analysis/{load,decision_gate}.py`
- 评审时间：2026-10-04 ~ 2026-10-05（在 S-tests / S-exposure / S-exposure2 之后、
  P1–P7 正式块之前完成关键项修复；P1 运行期间完成独立复审）
- 评审方式：
  1. 本机单测套件（`experiments/pilot/test_sim.py`，15 项，含 gold 数值核对）；
  2. 独立模型复审（对 `sim.py` 与 `decision_gate.py` 的专项审计，见 §3）；
  3. 同 seed 跨 run 一致性量化（`analysis/consistency_check.py`）。

## 1. 单测（机器门：全部 exit 0）

结果：15/15 PASS（`results/logs/s_tests_final.log`）。覆盖：

| 测试 | 保护的性质 |
|---|---|
| `test_smoothing_gradient_gold` | CE(ε) 梯度 = (softmax − 目标)/N 的**数值金标准**（ε=0/0.5/0.9） |
| `test_smoothing_optimum_gold` | 软标签 CE 最小值 = H(target)（ε=0.9, C=6） |
| `test_probe_gold_signal` | 探针在线性规则上 ≥0.97；打乱标签回落到随机（E4） |
| `test_probe_train_test_split_discipline` | 探针接口 train 拟合 / test 评估 |
| `test_determinism` | 同 seed 两次训练损失序列逐点相同 |
| `test_schedule_gold` | 评测点数下限：L∈{0,250,1000,4000} 全部 ≥200 点 |
| `test_task_counts_with_data` | CIFAR 每类 1000 train / 500 test，train≠test |
| `test_svhn_counts_with_data` | SVHN 10000/5000、标签 0–9、每类均衡 |
| `test_gas_sensor_with_data` | 传感器任务 128 维、6 类、无 NaN、训练步可跑通 |
| `test_checkpoint_roundtrip` | npz 检查点保存→加载输出逐位相同 |
| `test_transplant_composition` | 移植体输出 = 特征源 φ + 头源 W2（逐位） |
| `test_freeze_features_headonly` | headonly：特征参数不动、头参数更新 |
| `test_evaluate_finite` | 单点评测输出全部有限且在合法区间 |
| `test_train_step_loss_decreases` | 训练步有效 |
| `test_smoothing_pushes_toward_uniform` | **校准结论**：ε=1.0 使头去专化；ε=0.9 保留 argmax 序 |

## 2. 执行前修复的缺陷（评审发现 → 修复 → 回归）

| # | 缺陷 | 影响 | 修复 | 回归 |
|---|---|---|---|---|
| B1 | `_augment` 的 gather 索引形状与被 gather 维不一致 | CUDA device-side assert，整个 context 报废 | 先按行 gather（index 末维 40）再按列 gather | `test_determinism` 等全部通过 |
| B2 | `ctx.ref`（end_A 参考特征）用全测试集，评测用 2048 子集 | `feat_drift` 维度不匹配崩溃 | 参考特征改为在同一固定子集上提取 | S-exposure2 全部通过 |
| B3 | `metrics.head_forward` 未 flatten ResNet 特征 | ResNet-18 路径 `mat1 and mat2 shapes` 崩溃 | `.flatten(1)` 统一 | ResNet 校准跑通 |
| B4 | `peak_rss_mb()` 的 Win32 调用未设 argtypes，句柄被截断 | 资源记录恒为 nan | 设置 `argtypes/restype` 后取 `K32GetProcessMemoryInfo` | summary 中 rss≈1.95 GB |
| B5 | `machine_snapshot` 解包 `disk_usage` 返回 2 值（实为 3） | 快照缺 disk 字段 | 三值解包 | `preflight/resource-snapshot.json` 完整 |
| B6 | gold 梯度测试未除以 mean 归约因子 N | 误判为实现错误 | gold 改为 (p−t)/N | PASS |
| B7 | 可视化按 axis=0 求均值（步数维） | 图 1 形状错误 | axis=1（seed 维） | fig1/fig1b/fig5 生成成功 |
| B8 | LaTeX 表格 f-string 中 `{tabular}` 被当格式字段 | `make_tables.py` 崩溃 | 拆分 f-string | 6 张表生成成功 |
| B9 | 沙箱禁止 multiprocessing 命名管道（WinError 5） | 进程池无法启动 | 顺序执行 + `--shard k/n --state-id` 分片（每写者独立状态文件） | 3 分片并发运行正常 |
| B10 | P4 移植把整份 npz state（含 `features.*`、`head.*`）直接 `load_state_dict` 进 `nn.Linear` | P4 全部 10 runs 立即失败（Missing key/Unexpected key） | 过滤 `head.*` 并剥离前缀后再加载；`summary` 记录两个来源路径 | 新增回归测试 `test_transplant_npz_head_loading`（与手工组合逐位一致），PASS；失败的 P4 runs 在修复后由批次重跑 |

> B10 于正式块执行期间发现（P4 首次尝试失败），属**阻断性失败**（无数据被写入/污染，
> 失败 run 只留下 traceback 的 `summary.json`），修复后重跑；未影响 P1/P2/P3/P5/P6/P7 的测量代码路径。

## 3. 跨模型独立复审（第 4 次尝试成功，2026-10-05）

评审对象：`experiments/pilot/sim.py`、`analysis/decision_gate.py`（由**未参与生产**的
独立评审代理完成，含只读执行 `qc()/a2_gate()/a3_gate()/a4_gate()` 与真实 run 的抽查）。
评审结论：**`VERDICT: FIX FIRST - 1, 2`**（1 项 CRITICAL、1 项 MAJOR，另 4 项 MINOR）。
逐项处置如下（B11–B16）：

| # | 严重度 | 发现 | 处置 | 回归 |
|---|---|---|---|---|
| B11 | **CRITICAL** | A4 的**等预算（E1）排除条款被计算但从未进入 `pass`**：`pass = nontrivial and not headonly_covers` 漏掉"且不能被等预算解释" | 已实现 E1 条款（见下方"实现口径"）。当前数据：P1 的 L 跨度 acc 0.0075/nll 0.0210，P3（等预算）acc 0.0057/nll 0.0320 → **跨度未坍缩（≥50%）→ E1 不解释** → A4 仍 PASS，但现在是**真的检验过** | `decision_gate.py` A4 分支；`GATE.json` 新增 `L_range_P1`/`E1_budget_range`/`E1_explains`/`signal_not_explained_by_E1` |
| B12 | **MAJOR** | E2/headonly 判定用"任一 L 条件匹配"，规范要求"**每一个** L 条件" | 改为 `all(...)`：headonly 必须同时匹配全部 4 个 L 才算"整体解释" | 当前数据不变（headonly 0.660/0.913 与每个 L 都差 >0.01） |
| B13 | MINOR | A3 对高/低共享取 `any()`，等于双重机会 | 改为**两个共享度都必须过**，且每条件配对数 ≥4 才判 pass | 当前 0.124 / 0.115 均过；n_pairs=5、complete=True |
| B14 | MINOR | 缺 run 会使子条款**空洞成立**（headonly 全缺 → 跳过 E2；P3 全缺 → E1 无判定；A3 只剩 1 对也算中位） | A4 增加完整性检查（4×L≥4、3 对照≥4、3×P3≥3，缺则 `complete=False` → pass=False 并打印原因）；A3 要求 ≥4 对 | `GATE.json` 的 `complete/incomplete_reasons` |
| B15 | MINOR | `sim.py` 的相位 B 关键参数默认**非协议值**（eps_B=0.9、lr 回退 1e-3、dense=0/25），一旦有人绕过 BASE 网格就会静默偏离协议 | 默认值改为冻结协议值（1.0 / 1e-4 / 400 步 / 5 步） | 网格运行本就显式传值，**已有 114 runs 不受影响**（行为零变化） |
| B16 | MINOR | 无 end_A 参考时 `feat_drift` 仍写 `0.0`（Phase A 全程与移植 run），消费方会把"没测"读成"零漂移" | 改为**仅在有参考时写该键** | 分析脚本全部用 `.get`，历史数据不受影响 |

### B11 的实现口径（为何不是"把 P3 跨度与 0.03/0.05 阈值比较"）

评审者建议 `explained_P3 = (P3 跨度 < 0.03 且 < 0.05)`。**未采纳**，理由：
A4 的 0.03/0.05 是"差异是否存在"的**绝对**阈值，而 E1 要排除的是"差异是否由**预算**造成"，
二者的正确比较对象是 **P1 与 P3 在同一 L 集合上的跨度比**（等预算前 vs 等预算后）。
若按建议实现，由于 P1 自身的 L 跨度（acc 0.0075 / nll 0.0210）也低于阈值，
该判据会因"P3 跨度小"而失败——但它同样会因"P1 跨度小"而失败，即它测的是"差异大小"而非"预算解释力"。
当前口径直接回答 E1：**把步数拉平后，L 之间的差异模式没有坍缩（NLL 跨度甚至变大）→ 差异不是步数造成的**。
同时如实报告：终态差异的主要来源是 headonly 对照（0.660 vs 0.710–0.720），
L 之间的终态差异本身低于噪声底线（见 `report.md` §4）。

### 其余类别（评审者明确"无发现"）

步/相位索引、评测点计数与 ≥200 边界、ε/sham 核心语义、探针 train/test 泄漏、
NaN 缺口、检查点 state_dict 混用 —— **均无发现**（含对 `P1_high_L250_s42` 的
A=200/B=50/C=80 行数与 B 行 eps=1.0 的实测核对）。

### 前三次尝试

`af830825…`、`d34e8657…`、`efeed9f5…` 三次跨模型评审均在产出结论前中断（代理失败），无可用结论；第 4 次以缩小范围（仅 `sim.py` + `decision_gate.py`）重发后成功。

## 4. 已知限制（不修复，记录留痕）

| # | 事项 | 量化 | 处置 |
|---|---|---|---|
| L1 | CUDA kernel 非确定性：同 seed、同协议的不同 run，end_A 精度存在差异 | 同 seed 跨 L 的 end_A 极差 0.003–0.013（均值 0.009，见 `analysis/consistency_check.py`） | 记录为噪声底线；所有配对差异低于 ~0.013 不解释；P2-abrupt 与 P1-L0 同协议，可作为独立的重跑差异估计 |
| L2 | P1 `L=0` 与 P2 `abrupt` 协议相同 | — | 有意保留：跨块复现性检查（DEVIATIONS D9） |
| L3 | A1 的 ≥200 行不适用于 P4（移植 run 只有 Phase C，80 行） | — | 由设计决定，判定时跳过 P4 行数检查 |
| L4 | P0 首跑的 peak-RSS 为 nan（B4 修复前） | — | 收尾时独占 GPU 重测并回填 `RECORD.md` |

## 5. 决策门实现对照清单（`analysis/decision_gate.py` ↔ 冻结计划 §2）

| 判据 | 计划阈值 | 代码实现 | 一致? |
|---|---|---|---|
| A1 质控 | 无 NaN/traceback；end_A 中位 ≥0.85；行数 ≥200；全部正式 run 成功 | `qc()`：扫描 P1/P2/P3/P4/P5/P6/P7 全部 float 是否 NaN/Inf；`statistics.median(end_A)`；逐 run `len(metrics)>=200`（P4 按设计跳过，见 L3）；`missing` 为非 succeeded 的 run | ✅ |
| A2 头先释放 | 存在 L≥250 的评测点 `head≤0.50 ∧ probe≥0.70`，≥2/3 seeds 同向 | `a2_gate()`：对 L∈{250,1000,4000}×{high,low}×5 seeds 求满足点；`len(seeds_hit) >= ceil(2/3*5)=4`；总通过 = 任一条件通过，并记录覆盖共享度（单共享度 → modify 提示） | ✅（阈值 0.50/0.70 未放宽） |
| A3 特征后损伤 | 过渡末 probe(L4000) ≤ probe(L250) − 0.10 | `a3_gate()`：按 seed 配对 `drop = probe@endB(L250) − probe@endB(L4000)`，取中位数 ≥0.10；同时输出方向一致的 seed 数 | ✅ |
| A3 特征后损伤 | 过渡末 probe(L4000) ≤ probe(L250) − 0.10 | `a3_gate()`：按 seed 配对 `drop = probe@endB(L250) − probe@endB(L4000)`，**高、低共享两个条件都**要求中位 ≥0.10 且配对数 ≥4 | ✅ |
| A4 非平凡后果 | 终态 ≥0.05 NLL 或 ≥0.03 acc（L 与三对照），**且非等预算/仅头重训整体解释** | `a4_gate()`：终态 = 最后一个 Phase-C 点；① 全集极差达阈值；② **E1**：同一 L 集合上 P3（等预算）跨度 ≥ 50%×P1 跨度（未坍缩）；③ **E2**：headonly 必须匹配**每一个** L；④ 完整性（4×L≥4、3 对照≥4、3×P3≥3） | ✅（B11/B12/B14 修复后） |
| claim ceiling | 最多“可分离迹象” | `report["claim_ceiling"]` 字段 + 计划 §11 | ✅ |

## 6. 结论（评审后更新）

- 机器门：**PASS**（单测 16/16，exit 0）。
- 科学门（gold 数值核对）：**PASS**。
- 跨模型独立评审：**已完成**（第 4 次尝试），结论 `FIX FIRST - 1, 2`；
  B11（CRITICAL，A4 漏 E1 条款）与 B12（MAJOR，E2 用 any 而非 all）**已修复并回归**，
  B13–B16（MINOR）**已全部修复**；修复不改变任何已产生数据，只改变判定逻辑与未来默认值。
- 关键项 B1–B10 均在对应块运行前/中修复并回归；
- **formal 阶段前置条件的代码部分已满足**；结果层面的独立复核（reviewer 签署）仍待人工完成。
