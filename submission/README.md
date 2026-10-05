# submission/ —— Neurocomputing 投稿包（按官方 Guide for authors 构建）

由 `tools/build_submission.py` 从 `paper/main.tex` 生成，校验：`tools/check_submission.py`

## 为什么工作区有多个 .tex，而投稿包只有一个？
工作区是**数据驱动流水线**，分开写是为了杜绝手抄数字：
- `paper/results_macros.tex` —— 194 个数值宏，由 `analysis/paper_macros.py`
  从 207 个 run 的原始 `metrics.jsonl` 自动生成；
- `analysis/tables/tab_*.tex` —— 10 张表，由 `analysis/make_tables.py` 自动生成；
- `paper/main.tex` —— 正文，通过 `\input` 引用上面两者。

**投稿时全部内联**成本包唯一的 `main.tex`（构建脚本会断言无残留 `\input` 与
`../analysis` 路径）。**图必须保持单独文件**——指南原文：*“Submit each image as a
separate file”*，故 `figures/Figure_N.pdf` 不可合并进 tex。

## 文件清单
- `main.tex` —— **自包含单文件**：正文 + 数值宏 +10 张表（elsarticle `review,12pt`）
- `figures/Figure_1.pdf … Figure_11.pdf` —— 矢量图，按正文出现顺序命名
- `references.bib` —— BibTeX 版参考文献（31 条，与正文内嵌 thebibliography 对应；可选）
- `highlights.txt` —— Highlights 单独文件（5 条：OK（均 ≤85 字符））
- `cover_letter.md` —— 投稿信（贡献要点、期刊契合度、声明摘要；转成 PDF/Word 上传）
- `README.md` —— 本说明 + 合规清单

图文件对照表（原始名 → 投稿名）：
| 投稿名 | 原始名 |
|---|---|
| `Figure_1.pdf` | `fig1_head_feature_trajectories.pdf` |
| `Figure_2.pdf` | `fig1b_window_zoom.pdf` |
| `Figure_3.pdf` | `fig8_budget_gain.pdf` |
| `Figure_4.pdf` | `fig2_outcome_vs_L.pdf` |
| `Figure_5.pdf` | `fig2b_phase_c_trajectories.pdf` |
| `Figure_6.pdf` | `fig3_adaptation_speed.pdf` |
| `Figure_7.pdf` | `fig4_transplant.pdf` |
| `Figure_8.pdf` | `fig5_E3_diagnostics.pdf` |
| `Figure_9.pdf` | `fig6_sensor_drift.pdf` |
| `Figure_10.pdf` | `fig7_dose_curve.pdf` |
| `Figure_11.pdf` | `fig9_external_validity.pdf` |

## 编译
```
pdflatex main.tex
pdflatex main.tex          # 两遍以解析交叉引用
```
（指南不强制参考文献格式，录用后期刊会按其样式重排；若系统要求 BibTeX：
删除 thebibliography 环境，改用 `\bibliographystyle{elsarticle-num}` +
`\bibliography{references}` 并跑 pdflatex→bibtex→pdflatex×2。）

## 按 Guide for authors 逐条核对
| 指南要求（原文要点） | 本包状态 |
|---|---|
| 文章类型 Type 1 Regular article（须直接相关于神经网络/学习系统） | ✅ 机制 + 实验，属 learning systems |
| 文件格式：`.tex` 可交、LaTeX 允许双栏 | ✅ elsarticle；**本包仅 1 个 tex（已内联）** |
| 单一匿名评审（single anonymized：审稿人对作者匿名，作者身份可见） | ✅ 按 Elsevier 定义作者信息正常出现；**若投稿系统提示需匿名化，删除姓名/单位行即可** |
| 标题页：题名、作者（名+姓）、单位（含国家）、**通讯作者邮箱** | ✅ 已填：Xianzhe Liu、Hunan University、addressline=Lushan Road, Yuelu District、Changsha 410082, China、`\ead{liuxianzhe@hnu.edu.cn}`。**中文姓名/校名放在注释里**（pdfLaTeX 无法排版中文）；XeLaTeX+xeCJK 可切换注释中的中文版本 |
| **摘要 ≤250 词**、可独立成文、避免引用 | ✅ 242 词、无引用 |
| **关键词 1–7 个**、避免多词含 and/of | ✅ 6 个（`plasticity loss` 已改写避免 of） |
| Highlights：3–5 条、每条 ≤85 字符、**单独文件且文件名含 highlights** | ✅ `highlights.txt` 5 条（74–79 字符）+ 正文 highlights 环境 |
| 公式为可编辑文本、变量斜体、显示公式按序编号 | ✅ 数学模式、单个自动编号公式 |
| 表格为可编辑文本、**无竖线/单元格阴影**、有题注、正文引用 | ✅ booktabs；已内联进 main.tex；自检含竖线 + “未被引用”检查 |
| 图：**每个文件单独提交**、逻辑命名（如 Figure_1）、正文引用、题注、矢量 PDF | ✅ `Figure_1..11.pdf`（矢量），自检“未被引用”已过 |
| 声明：competing interests（**另需在系统填 declarations 工具并上传 .docx**） | ✅ 正文有声明；⚠️ 投稿系统里还需生成 Word 版上传 |
| 声明：Funding（无资助用指南原句） | ✅ 采用指南给的标准句（如获资助见正文注释格式） |
| **声明：生成式 AI 使用（置于参考文献之前的新节）** | ✅ 已**填写完毕**：工具=DeepSeek（DeepSeek Harness），用途=起草/修订文稿与实现调试实验分析绘图代码；并声明结果由代码产出、图由脚本渲染、文献元数据经 arXiv/ACL Anthology/OpenAlex 核对、作者对内容负全责 |
| **Data statement（Option C：入库并引用，或说明原因）** | ✅ 已说明第三方数据以 [dataset] 引用 + 代码/指标将入库并给出 PID |
| CRediT 作者贡献（指南：corresponding authors **are required**） | ✅ 已按**单作者**填好（Xianzhe Liu: Conceptualization… Writing – review and editing）；**若实际有合著者，必须改成每人的真实角色** |
| 参考文献：正文与列表一一对应、须为真实来源、尽量给 DOI | ✅ 自检 32/32 双向一致；未能核实的 DOI 已标注待核（`gasdrift`） |
| 参考文献编号按出现顺序（录用后由期刊重排） | ⚠️ 当前按条目顺序编号；指南注明录用后统一重排，投稿阶段"格式一致即可" |
| 数据引用标注 `[dataset]` | ✅ CIFAR/SVHN/Fashion-MNIST/UCI Gas 四条 |
| Vitae：每位作者 ≤100 词简介 + 照片（单独文件） | ⚠️ **需作者自备**，未包含在本包 |
| 图形摘要（Graphical abstract） | ⚠️ 指南未列为必需；如需请另备 |
| SI、视频、图像 manipulation 声明 | N/A（本研究无原始图像，图均由脚本从测量数据渲染） |
| 语言：美式**或**英式，不混用 | ✅ 美式（analyze/specialization，无 behaviour/modelling 混用） |

## 投稿前必须人工完成
> 已确认：**单作者**（Xianzhe Liu）。姓名/单位/地址/邮箱、CRediT、竞争利益、
> 基金（无资助）、生成式 AI 声明均已填好。正文保留学术惯用的第一人称复数 "we"，
> 如期刊/个人偏好改 "I"，可全局替换（指南无此项要求）。
1. **如实际有合著者**才需要：`\author`/`\affiliation` 加行，CRediT 改成每人真实角色
2. 竞争利益：在投稿系统的 declarations 工具生成 .docx 并上传（正文声明已写好）
3. `gasdrift` 的 DOI 按 UCI 落地页核对（或删除 DOI）
4. 按需准备 Vitae（≤100 词 + 照片）
5. 自己跑 `pdflatex main.tex` 两遍确认编译（我按要求未编译；结构级检查已全过）
6. 投稿时一并上传：`main.tex`、`figures/`（11 个 PDF）、`highlights.txt`、
   `cover_letter.md`、`references.bib`（可选）、declarations(.docx)

## 可复现性材料（代码库内，非投稿包）
- `results/runs/<run_id>/metrics.jsonl`：207 runs 逐点原始读数
- `analysis/decision_gate.py` → `analysis/GATE.json`：门判定（advance）
- `pilot/PLAN.md`、`pilot/DEVIATIONS.md`、`pilot/CODE_REVIEW.md`：预注册/偏离/评审留痕
- 生成脚本：`analysis/visualization.py`、`make_tables.py`、`paper_macros.py`
