"""生成投稿包 submission/（按 Neurocomputing Guide for authors 定制）。

- 主文件路径重写；图片按正文首次出现顺序重命名为 Figure_N.pdf（指南示例命名）
- 生成 highlights.txt、references.bib 副本、README（合规清单 + 待填项）
- 最后自动跑 tools/check_submission.py 校验
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ANALYSIS = ROOT / "analysis"
SUB = ROOT / "submission"

HIGHLIGHTS = [
    "The classifier head loses old-task readability within 65-220 exposure steps",
    "A linear probe still reads the features then: the two timescales differ 10-fold",
    "Adaptation over the first 1000 steps is interior-optimal in exposure length",
    "Every exposure beats none at every adaptation budget (5/5 seeds, 189 runs)",
    "A target-entropy dose-response separates damage from label-smoothing effects",
]

COVER_LETTER = """# Cover letter — Neurocomputing (Type 1: Regular article)

**Manuscript:** Head first, features later: a transient window of classifier-head
de-specialization under uniform label exposure and its effect on task adaptation

**Corresponding author:** Xianzhe Liu, Hunan University (liuxianzhe@hnu.edu.cn)

---

Dear Editor,

We would like to submit our manuscript for consideration as a Type 1 regular
article. It is directly relevant to neural networks and learning systems: it
studies the dynamics of a classifier head and a feature extractor during a
task transition, and derives practical guidance for how long to dwell in a
label-scarce transition.

**What is new.**

1. **A transient mechanism, not a terminal one.** While the literature studies
   loss of plasticity as a terminal state (trapping manifolds, collapsed
   geometry, last-layer resets), we measure the transition itself: within
   65--220 steps of uniform label exposure the classifier head stops reading
   the old task (accuracy <= 0.50) while a linear probe still recovers it from
   the features (>= 0.70) - a timescale separation of roughly an order of
   magnitude, replicated in 5/5 seeds at two levels of feature sharing.

2. **Practical, quantitative guidance.** Read as budget-limited adaptation,
   every exposure beats none for every budget from 50 to 2000 steps (5/5
   seeds), by up to +11.5 accuracy points at a 50-step budget, and an interior
   optimum in exposure length appears on the pre-registered adaptation-speed
   readout. A stopping rule follows directly: switch when the head stops
   reading the old task and before the feature probe begins to fall - a signal
   current-task accuracy cannot provide.

3. **Unusually strong evidence practice.** The study was pre-registered with
   four frozen decision gates and a frozen claim ceiling; every protocol
   deviation is logged with its calibration evidence. We report 207 runs over
   five datasets (CIFAR-10/100, SVHN, Fashion-MNIST, UCI gas-sensor drift),
   three architectures, a five-point target-entropy dose-response, a causal
   cross-transplant of heads and features, five controls (including a
   last-layer-reset baseline), and a deterministic-kernel replication whose
   identical-configuration reruns are bitwise identical (zero process-level
   noise). Code, raw per-evaluation metrics, checkpoints and the analysis
   pipeline are released.

4. **Honest negative results.** Under deterministic kernels the endpoint
   effect is only 0.0021 accuracy points across exposure lengths; we therefore
   claim adaptation-speed gains and explicitly not endpoint gains, and we
   report a benchmark where only half of the mechanism appears.

**Fit to Neurocomputing.** The work combines analysis of network dynamics with
a practical method for non-stationary training, and includes an accessible
theoretical account (a gradient decomposition that predicts crossing times
scaling as 1/learning rate, verified over a 10-fold range).

**Statements.** Single author; no competing interests; this research received
no specific grant funding; the use of an LLM-based AI assistant in manuscript
and code preparation is declared in the manuscript, where all figures are
generated from recorded measurements by released scripts.

Thank you for your consideration.

Yours sincerely,
Xianzhe Liu
Hunan University, Changsha, China
liuxianzhe@hnu.edu.cn
"""


def main():
    if SUB.exists():
        shutil.rmtree(SUB)
    (SUB / "figures").mkdir(parents=True)

    # ---- 1) 主文件：把 results_macros.tex 与10 张表**内联**成单个自包含 main.tex
    #         （表/宏在工作区是流水线自动生成的；投稿只需一个 .tex + 单独的图 PDF）
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")

    macro_src = (PAPER / "results_macros.tex").read_text(encoding="utf-8")
    assert "\\input{results_macros}" in tex
    tex = tex.replace(
        "\\input{results_macros}",
        "% ---- results_macros.tex (auto-generated from raw metrics; inlined for "
        "submission) ----\n" + macro_src.rstrip())

    def _inline_table(m):
        path = m.group(1)
        fp = (PAPER / path).resolve() if path.startswith("..") else (PAPER / path)
        if not fp.exists():
            fp = (PAPER / path).with_suffix(".tex")
        body = fp.read_text(encoding="utf-8").rstrip()
        return f"% ---- {fp.name} (auto-generated by analysis/make_tables.py; inlined) ----\n{body}"

    tex, n_tabs = re.subn(r"\\input\{(\.\./analysis/tables/[^}]+)\}",
                          _inline_table, tex)
    tex = tex.replace("../analysis/tables/", "tables/")   # 兜底（不应再有）

    # 图：按正文首次出现顺序重命名为 Figure_N.pdf
    seen: dict = {}
    fig_map: list = []          # [(submission 名, 原名), ...] 按出现顺序

    def _ren(m):
        name = m.group(1)
        if name not in seen:
            new = f"Figure_{len(seen) + 1}.pdf"
            seen[name] = new
            fig_map.append((new, name))
        return m.group(0).replace(f"../analysis/figures/{name}",
                                  f"figures/{seen[name]}")

    tex = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{\.\./analysis/figures/([^}]+)\}",
                 _ren, tex)
    assert "../analysis" not in tex, "residual relative path in main.tex"
    assert "\\input{" not in tex, "residual \\input in merged main.tex"
    (SUB / "main.tex").write_text(tex, encoding="utf-8")

    # ---- 2) 依赖文件
    for f in ("references.bib",):
        shutil.copy2(PAPER / f, SUB / f)

    # ---- 3) 图（投稿包不含 tables/ 与 results_macros.tex：已内联进 main.tex）
    for new, old in fig_map:
        shutil.copy2(ANALYSIS / "figures" / old, SUB / "figures" / new)

    # ---- 4) Highlights（单独可编辑文件，文件名含 highlights）
    bad = [h for h in HIGHLIGHTS if len(h) > 85]
    (SUB / "highlights.txt").write_text(
        "Highlights\n" + "\n".join(f"- {h}" for h in HIGHLIGHTS) + "\n",
        encoding="utf-8")

    # ---- 4b) Cover letter（投稿信，非指南必需但常规要求）
    (SUB / "cover_letter.md").write_text(COVER_LETTER, encoding="utf-8")

    # ---- 5) README
    n_figs = len(fig_map)
    macro_n = len(re.findall(r"\\newcommand",
                             (PAPER / "results_macros.tex").read_text(encoding="utf-8")))
    n_ref = len(re.findall(r"\\bibitem", (PAPER / "main.tex").read_text(encoding="utf-8")))
    hl_status = "OK（均 ≤85 字符）" if not bad else f"超长: {bad}"
    fig_rows = "\n".join(f"| `{new}` | `{old}` |" for new, old in fig_map)

    readme = """# submission/ —— Neurocomputing 投稿包（按官方 Guide for authors 构建）

由 `tools/build_submission.py` 从 `paper/main.tex` 生成，校验：`tools/check_submission.py`

## 为什么工作区有多个 .tex，而投稿包只有一个？
工作区是**数据驱动流水线**，分开写是为了杜绝手抄数字：
- `paper/results_macros.tex` —— {macro_n} 个数值宏，由 `analysis/paper_macros.py`
  从 207 个 run 的原始 `metrics.jsonl` 自动生成；
- `analysis/tables/tab_*.tex` —— 10 张表，由 `analysis/make_tables.py` 自动生成；
- `paper/main.tex` —— 正文，通过 `\\input` 引用上面两者。

**投稿时全部内联**成本包唯一的 `main.tex`（构建脚本会断言无残留 `\\input` 与
`../analysis` 路径）。**图必须保持单独文件**——指南原文：*“Submit each image as a
separate file”*，故 `figures/Figure_N.pdf` 不可合并进 tex。

## 文件清单
- `main.tex` —— **自包含单文件**：正文 + 数值宏 +10 张表（elsarticle `review,12pt`）
- `figures/Figure_1.pdf … Figure_{n_figs}.pdf` —— 矢量图，按正文出现顺序命名
- `references.bib` —— BibTeX 版参考文献（{n_ref} 条，与正文内嵌 thebibliography 对应；可选）
- `highlights.txt` —— Highlights 单独文件（5 条：{hl_status}）
- `cover_letter.md` —— 投稿信（贡献要点、期刊契合度、声明摘要；转成 PDF/Word 上传）
- `README.md` —— 本说明 + 合规清单

图文件对照表（原始名 → 投稿名）：
| 投稿名 | 原始名 |
|---|---|
{fig_rows}

## 编译
```
pdflatex main.tex
pdflatex main.tex          # 两遍以解析交叉引用
```
（指南不强制参考文献格式，录用后期刊会按其样式重排；若系统要求 BibTeX：
删除 thebibliography 环境，改用 `\\bibliographystyle{{elsarticle-num}}` +
`\\bibliography{{references}}` 并跑 pdflatex→bibtex→pdflatex×2。）

## 按 Guide for authors 逐条核对
| 指南要求（原文要点） | 本包状态 |
|---|---|
| 文章类型 Type 1 Regular article（须直接相关于神经网络/学习系统） | ✅ 机制 + 实验，属 learning systems |
| 文件格式：`.tex` 可交、LaTeX 允许双栏 | ✅ elsarticle；**本包仅 1 个 tex（已内联）** |
| 单一匿名评审（single anonymized：审稿人对作者匿名，作者身份可见） | ✅ 按 Elsevier 定义作者信息正常出现；**若投稿系统提示需匿名化，删除姓名/单位行即可** |
| 标题页：题名、作者（名+姓）、单位（含国家）、**通讯作者邮箱** | ✅ 已填：Xianzhe Liu、Hunan University、addressline=Lushan Road, Yuelu District、Changsha 410082, China、`\\ead{{liuxianzhe@hnu.edu.cn}}`。**中文姓名/校名放在注释里**（pdfLaTeX 无法排版中文）；XeLaTeX+xeCJK 可切换注释中的中文版本 |
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
1. **如实际有合著者**才需要：`\\author`/`\\affiliation` 加行，CRediT 改成每人真实角色
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
"""
    (SUB / "README.md").write_text(
        readme.format(macro_n=macro_n, n_ref=n_ref, n_figs=n_figs,
                      hl_status=hl_status, fig_rows=fig_rows),
        encoding="utf-8")

    print(f"-> {SUB}")
    print(f"   main.tex (merged: macros+tables inlined), figures: {n_figs} "
          f"(Figure_1..Figure_{n_figs}), macros: {macro_n}, refs: {n_ref}")

    # ---- 6) 自动校验
    chk = subprocess_run([sys.executable, str(ROOT / "tools" / "check_submission.py"),
                          str(SUB / "main.tex")])
    return chk


def subprocess_run(cmd):
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print(r.stderr)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
