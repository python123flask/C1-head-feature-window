"""投稿前综合自检：结构、宏、引用、文件完整性、摘要/关键词长度、必需声明。"""
import re
import sys
from collections import Counter
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
MAIN = sys.argv[1] if len(sys.argv) > 1 else str(PAPER / "main.tex")
main = Path(MAIN)
t = main.read_text(encoding="utf-8")
base = main.parent

issues = []
warns = []

# 1 环境配对
b = Counter(re.findall(r"\\begin\{(\w+\*?)\}", t))
e = Counter(re.findall(r"\\end\{(\w+\*?)\}", t))
unbal = {k: (b[k], e[k]) for k in set(b) | set(e) if b[k] != e[k]}
if unbal:
    issues.append(f"unbalanced environments: {unbal}")

# 2 宏定义（白名单：LaTeX 内建数学/文本宏）
LATEX_BUILTIN = {
    "Delta", "Gamma", "Lambda", "Omega", "Phi", "Pi", "Sigma", "Theta",
    "alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta", "iota",
    "kappa", "lambda", "mu", "nu", "pi", "rho", "sigma", "tau", "upsilon",
    "phi", "chi", "psi", "omega", "varphi", "varepsilon", "ell", "infty",
    "times", "cdot", "pm", "mp", "leq", "geq", "neq", "approx", "sim",
    "subseteq", "in", "notin", "to", "rightarrow", "leftarrow", "Rightarrow",
    "Leftrightarrow", "mapsto", "forall", "exists", "emptyset", "nabla",
    "partial", "sum", "prod", "int", "frac", "sqrt", "min", "max", "exp",
    "log", "lim", "sin", "cos", "tan", "top", "bot", "ldots", "cdots",
    "text", "textbf", "textit", "emph", "label", "ref", "cite", "citep",
    "section", "subsection", "paragraph", "footnote", "url", "href",
}
macro_file = base / "results_macros.tex"
macros = set()
if macro_file.exists():
    macros = set(re.findall(r"\\newcommand\{\\(\w+)\}", macro_file.read_text(encoding="utf-8")))
own = set(re.findall(r"\\newcommand\{\\(\w+)\}", t))
used = set(re.findall(r"\\([A-Z][A-Za-z0-9]*)", t)) | \
    set(re.findall(r"\\([a-z]+)", t)) & own
missing_macros = sorted(u for u in used
                        if u not in macros and u not in own and u not in LATEX_BUILTIN)
if missing_macros:
    issues.append(f"possibly undefined macros: {missing_macros}")

# 3b references.bib（若存在）与 \bibitem 键一致性、语法完整性
bibfile = base / "references.bib"
if bibfile.exists():
    bt = bibfile.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bt))
    bibitem_keys = set(re.findall(r"\\bibitem\{([^}]*)\}", t))
    missing_in_bib = sorted(bibitem_keys - bib_keys)
    extra_in_bib = sorted(bib_keys - bibitem_keys)
    if missing_in_bib:
        issues.append(f"references.bib missing entries: {missing_in_bib}")
    if extra_in_bib:
        warns.append(f"references.bib extra entries: {extra_in_bib}")
    if bt.count("{") != bt.count("}"):
        issues.append(f"references.bib braces unbalanced: {{={bt.count('{')} }}={bt.count('}')}")
    entries = re.split(r"(?m)^@", bt)[1:]
    bad_entries = [e.split("{")[0] for e in entries if not e.rstrip().endswith("}")]
    if bad_entries:
        issues.append(f"references.bib malformed entries: {bad_entries}")

# 3 引用 vs 参考文献
cites = set()
for m in re.findall(r"\\cite\{([^}]*)\}", t):
    cites.update(x.strip() for x in m.split(","))
bib = set(re.findall(r"\\bibitem\{([^}]*)\}", t))
no_bib = sorted(cites - bib)
unused_bib = sorted(bib - cites)
if no_bib:
    issues.append(f"cited but NO bibitem: {no_bib}")
if unused_bib:
    warns.append(f"bibitem never cited: {unused_bib}")

# 4 文件完整性（input / includegraphics）
for f in re.findall(r"\\input\{([^}]*)\}", t):
    p = base / f
    if not p.exists() and not (base / (f + ".tex")).exists():
        issues.append(f"missing input: {f}")
for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", t):
    if not (base / f).exists():
        issues.append(f"missing figure: {f}")

# 5 摘要 / 关键词
abs_raw = t.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
abs_words = len(re.sub(r"\\[a-zA-Z]+|[{}$\\]", " ", abs_raw).split())
if abs_words > 250:
    issues.append(f"abstract {abs_words} words > 250 (Neurocomputing Guide for Authors)")
print(f"abstract words : {abs_words}  (limit 250)")
kws_raw = t.split(r"\begin{keyword}")[1].split(r"\end{keyword}")[0]
kws = [x.strip() for x in re.split(r"\\sep|\\and", kws_raw) if x.strip()]
if not 1 <= len(kws) <= 7:
    issues.append(f"keywords={len(kws)} (Neurocomputing 要求 1–7 个)")
multi = [k for k in kws if re.search(r"\b(and|of)\b", k)]
if multi:
    warns.append(f"keywords containing 'and/of' (指南建议避免): {multi}")

# 6 作者/机构留空
if not re.search(r"\\author\[", t):
    issues.append("no \\author block")
if not re.search(r"\\affiliation\[", t):
    issues.append("no \\affiliation block")
if not re.search(r"\\cortext", t):
    warns.append("no corresponding-author marker")

# 6 重复标签（会导致 LaTeX "multiply defined labels" 与重复表格/图）
all_labels = re.findall(r"\\label\{([^}]*)\}", t)
for f in re.findall(r"\\input\{([^}]*)\}", t):
    p = base / f
    fp = p if p.suffix else p.with_suffix(".tex")
    if fp.exists():
        all_labels += re.findall(r"\\label\{([^}]*)\}", fp.read_text(encoding="utf-8"))
dup_labels = [k for k, v in Counter(all_labels).items() if v > 1]
if dup_labels:
    issues.append(f"duplicate \\label (multiply-defined): {dup_labels}")

dup_inputs = [k for k, v in Counter(re.findall(r"\\input\{([^}]*)\}", t)).items() if v > 1]
if dup_inputs:
    issues.append(f"duplicate \\input: {dup_inputs}")

# 7 Elsevier/Neurocomputing 必需声明（依据 Guide for authors 原文）
required = {
    "highlights": r"\\begin\{highlights\}",
    "CRediT author statement": r"CRediT author",
    "declaration of competing interest": r"(?i)competing interest",
    "funding": r"(?i)fund(ing|ed)",
    "data availability": r"(?i)data availability",
    "declaration of generative AI use": r"(?i)Declaration of generative AI",
    "corresponding-author email (\\ead)": r"\\ead\{",
}
for name, pat in required.items():
    if not re.search(pat, t):
        issues.append(f"missing required statement: {name}")

# 7b Highlights 规则：3–5 条、每条 ≤85 字符（含空格）
hl = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", t, re.S)
if hl:
    items = re.findall(r"\\item\s+(.+)", hl.group(1))
    if not 3 <= len(items) <= 5:
        issues.append(f"highlights count {len(items)} not in 3..5")
    long_items = [i for i in items if len(i) > 85]
    if long_items:
        issues.append(f"highlights >85 chars: {[(len(i), i[:40]) for i in long_items]}")
else:
    issues.append("highlights environment not found")

# 7c 图/表必须在正文中被引用（指南：cite all images/tables）
labels_fig = set(re.findall(r"\\label\{(fig:[^}]*)\}", t))
labels_tab = set(re.findall(r"\\label\{(tab:[^}]*)\}", t))
refs_all = set(re.findall(r"\\ref\{([^}]*)\}", t))
uncited_figs = sorted(labels_fig - refs_all)
uncited_tabs = sorted(labels_tab - refs_all)
if uncited_figs:
    issues.append(f"figures never cited in text: {uncited_figs}")
if uncited_tabs:
    issues.append(f"tables never cited in text: {uncited_tabs}")

# 7d 表格不应含竖线/单元格阴影（booktabs 风格）
for f in re.findall(r"\\input\{([^}]*)\}", t):
    if "tables" not in f:
        continue
    fp = base / f
    fp = fp if fp.suffix else fp.with_suffix(".tex")
    if fp.exists():
        spec = re.search(r"\\begin\{tabular\}\{([^}]*)\}", fp.read_text(encoding="utf-8"))
        if spec and "|" in spec.group(1):
            issues.append(f"vertical rules in {f}: {spec.group(1)}")

# 8 文档类与期刊
if not re.search(r"\\documentclass(\[[^\]]*\])?\{elsarticle\}", t):
    issues.append("not elsarticle class")
if not re.search(r"\\journal\{Neurocomputing\}", t):
    warns.append("\\journal not set to Neurocomputing")

# 9 规模统计
n_fig = len(re.findall(r"\\includegraphics", t))
n_tab = (len(re.findall(r"\\input\{[^}]*tables", t))
         or len(re.findall(r"\\begin\{tabular\}", t)))   # 内联合并后按 tabular 计
n_ref = len(bib)
words = len(re.sub(r"\\[a-zA-Z]+|[{}$%]", " ", t).split())

# 10 交叉引用：每个 \ref 必须有对应 \label（含 \input 进来的表格文件）
labels = set(re.findall(r"\\label\{([^}]*)\}", t))
for f in re.findall(r"\\input\{([^}]*)\}", t):
    p = base / f
    fp = p if p.suffix else p.with_suffix(".tex")
    if fp.exists():
        labels |= set(re.findall(r"\\label\{([^}]*)\}", fp.read_text(encoding="utf-8")))
refs = set(re.findall(r"\\ref\{([^}]*)\}", t))
bad_refs = sorted(refs - labels)
if bad_refs:
    issues.append(f"\\ref without \\label: {bad_refs}")

# 11 遗留占位符（注释除外；作者留空是预期的）
t_nocomment = re.sub(r"(?m)^\s*%.*$", "", t)
placeholders = re.findall(r"(?i)(TODO|FIXME|XXX|待填|待定|TBD)", t_nocomment)
if placeholders:
    warns.append(f"placeholder markers in text: {set(placeholders)}")

# 12 图表编号引用
figs_labels = set(re.findall(r"\\label\{(fig:[^}]*)\}", t))
tab_labels = set(re.findall(r"\\label\{(tab:[^}]*)\}", t))

print("=" * 70)
print(f"file           : {main}")
print(f"keywords       : {len(kws)} -> {kws}")
print(f"figures/tables : {n_fig} / {n_tab}")
print(f"references     : {n_ref} cited-uniquely={len(cites)}")
print(f"total words    : ~{words}")
print(f"macros         : {len(macros)} generated + {len(own)} local")
print("-" * 70)
for w in warns:
    print(f"WARN : {w}")
for i in issues:
    print(f"ISSUE: {i}")
print("-" * 70)
print("RESULT:", "FAIL" if issues else ("PASS (with warnings)" if warns else "PASS"))
sys.exit(1 if issues else 0)
