# RECORD.md — 运行记录区（计划 §8 回填）

> 对应冻结计划的 8.1 / 8.2 / 8.3 节。数值全部来自 `results/runs/*/summary.json`、
> `analysis/GATE.json`，由执行者回填；复核（reviewer）一栏须由未参与生产者复核后才可标 completed。

## 8.1 P0 实测（协议总步数 = 5000 + L + 2000 ≤ 11000，评测 504 点；见 DEVIATIONS D6）

- wall_time_seconds: **85.6**（P0_timing 收尾独占 GPU 重测；3 路并发时单 run 约 150–270 s）
- rss_peak_mb: **1947.5**
- gpu_memory_peak_mb: **605.8**
- batch_size_decision: **128 保持不变**（峰值显存 606 MB / 8.5 GB，余量充足；无需降级）

## 8.2 资源判定

- decision: **go**
- reason: 单 run 85.6 s（独占）、峰值显存 606 MB、峰值 RSS 1.95 GB；
  正式 100 runs（全清单 114）在 3 路并发下数小时内完成，远低于 30 h / 50 GB 预算；
  未触发任何降级策略（未减 seed、未简化模型、未移除 P6）。

## 8.3 决策门判定（`analysis/GATE.json`，由 `analysis/decision_gate.py` 生成）

- A1_quality_control: **pass**（100/100 正式 run 成功；无 NaN；end_A 中位 0.880 ≥ 0.85；评测点 280–504/run ≥ 200）
- A2_head_release: **pass**（L∈{250,1000,4000} × {高,低}共享 共 6 条件全部 5/5 seeds；
  首现步 65–220，高共享 L=4000 窗口 160–380）
- A3_feature_damage: **pass**（L=250 vs L=4000 过渡末 probe 配对下降中位：高 0.124 / 低 0.115 ≥ 0.10，方向一致）
- A4_nontrivial_outcome: **pass**（{4 个 L + 3 对照} 终态极差 acc 0.060 ≥ 0.03、NLL 0.101 ≥ 0.05；
  headonly（0.660/0.913）无法整体解释 → E2 排除）
- overall_decision: **advance**
- reason: 四门全过；机制（头先释放/特征后损伤）在两个共享度、5 seeds 下复现；
  预注册 adaptation_speed 读数出现内部最优窗口（L=250 最优，5/5 seeds）。
  **但**：终态（2000 步）差异处于重跑噪声底线附近（同协议重跑差 acc≤0.024、NLL≤0.046），
  故只主张“适应窗口效应”，不主张终态收益；P6 真实数据只复现头释放成分（特征损伤未出现，记为边界条件）。
- claim_ceiling: 最多“候选机制在合成任务上存在可分离迹象”；禁止论文级/创新性主张。
- reviewer: **部分完成** —— 跨模型代码评审已于 2026-10-05 完成（第 4 次尝试，
  独立评审代理，结论 `FIX FIRST - 1, 2`，B11 CRITICAL + B12 MAJOR 及 4 项 MINOR **全部修复并回归**，
  见 `pilot/CODE_REVIEW.md` §3）；**结果层面的人工独立复核（签署）仍待完成**，
  在此之前不得把状态标为 completed。

## 附：执行时间线（自动事件来自 experiments/ORCHESTRATION_STATE*.json）

| 阶段 | 状态 | 备注 |
|---|---|---|
| S-tests（单测） | ✅ 完成 | 16/16 PASS，exit 0（`results/logs/s_tests_final.log`） |
| S-exposure（ε 诊断） | ✅ 完成 3 runs | ε=0.9 头不掉、probe 回升 → 触发 D4 |
| S-exposure2（ε × lr_B 扫描） | ✅ 完成 6 runs | 选择 ε=1.0 + lr_B=1e-4（DEVIATIONS §E） |
| S-calib（lr 校准） | ✅ 完成 3 runs | 旧类划分 end_A≈0.81 <0.85 → 触发 D3 任务构造适配 |
| S-smoke / P0-timing | ✅ 完成 | smoke 指标合理（acc>0.25、无 NaN）；P0 收尾独占重测 |
| P1-main（40） | ✅ 完成 40/40 | 3 分片并发 |
| P2-controls（15） | ✅ 完成 15/15 | abrupt/direct/headonly |
| P3-fixedC（9） | ✅ 完成 9/9 | 等总步数 sham |
| P4-intervention（10） | ✅ 完成 10/10 | 首次因 B10 失败 → 修复后重跑，全部成功 |
| P5-diagnostics（8） | ✅ 完成 8/8 | ε=1.0 + 置换检验 / ε=0 续训 |
| P7-dose（6） | ✅ 完成 6/6 | ε=0.9 剂量-响应 |
| P6-sensor（12） | ✅ 完成 12/12 | UCI Gas batch 1→5 |
| **P8-resnet（9）** | ✅ 完成 9/9 | ResNet-18 规模扩展（计划 3.2.2）：A2 6/6 命中、首现 40 步 |
| **P9-eps（12）** | ✅ 完成 12/12 | ε=0.95/0.99 连续剂量：头释放阈值型、特征损伤连续型 |
| **P10-det（21）** | ✅ 完成 21/21 | 确定性核：6/6 对 **逐位一致（max\|Δ\|=0）** |
| **S-calib2（4）** | ✅ 完成 4/4 | CIFAR-100 25 类 end_A 0.93/0.91 → 同一 0.85 阈值直接适用 |
| **P11-c100（18）** | ✅ 完成 18/18 | CIFAR-100 外部效度：A2 两模型各 **6/6**；SmallCNN probe 降 0.147、ResNet 仅 0.015 |
| **P12-fmnist（9）** | ✅ 完成 9/9 | 第二个跨域伙伴：A2 **6/6**、probe 0.717→0.595、窗口 0.837→0.849 |
| **P13-detseed（15）** | ✅ 完成 15/15 | 确定性平台 10 seeds：窗口增益 **+0.0263（10/10）**、端点 L 极差 **0.0021** |
| **P14-headreset（5）** | ✅ 完成 5/5 | 头重置基线：适应速度最快（vs L0 **+0.0442 5/5**、vs L250 +0.0184 5/5），终态与 L=4000 噪声内 → 见 D20 |
| 决策门 | ✅ **advance** | **189/189 正式 run**（P1–P14）；A1–A4 全过（含评审修复后的 E1/E2 条款复判） |
| 跨模型评审 | ✅ 完成（第4次尝试） | `FIX FIRST - 1, 2` → B11 CRITICAL + B12 MAJOR + 4 MINOR **全部修复并回归** |
| 分析产物 | ✅ | **10 张 PDF 矢量图 + 10 张 LaTeX 表 + 175 个数值宏** + budget_analysis.json |
| 论文 TeX | ✅ | `paper/main.tex`（静态检查通过：环境配对/宏/引用均无缺失；含理论节、外部效度节、同期工作对比） |
| 查新（第2轮） | ✅ | arXiv+OpenAlex：`label smoothing`×`continual learning`=0 命中、`uniform label`×`plasticity`=0 命中；最近同期工作 arXiv:2609.33620 已入 related work |
| 结果层面独立复核 | ❌ 待人工签署 | 代码级跨模型评审已完成；结果复核仍需非生产者签署后方可标 completed |
