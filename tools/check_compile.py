"""编译级/一致性检查：
1) 正文（非注释）中的非 ASCII 字符 —— pdfLaTeX 会直接报错
2) 花括号 / $ 配对
3) 摘要是否含 \\cite
4) 内联表格数（合并后应为 10）
5) 图窗/首现步等关键数字的宏与文本一致性
"""
import re
import sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/main.tex")
t = p.read_text(encoding="utf-8")
issues = []

# 1) 去掉注释行（但保留行内 % 前内容过于复杂，这里按整行注释处理 + 去掉 \verb 无）
lines = []
for ln in t.split("\n"):
    if ln.lstrip().startswith("%"):
        lines.append("")
    else:
        # 去掉行内注释（简易：无转义的 %）
        i = ln.find("%")
        while i > 0 and ln[i - 1] == "\\":
            i = ln.find("%", i + 1)
        lines.append(ln if i < 0 else ln[:i])
body = "\n".join(lines)

non_ascii = {}
for m in re.finditer(r"[^\x00-\x7F]", body):
    ch = m.group(0)
    s = body.rfind("\n", 0, m.start()) + 1
    e = body.find("\n", m.end())
    ctx = body[s:e if e > 0 else len(body)][:100]
    non_ascii.setdefault(ch, []).append(ctx)
if non_ascii:
    for ch, ctxs in non_ascii.items():
        issues.append(f"non-ASCII U+{ord(ch):04X} {ch!r} x{len(ctxs)} in typeset text: {ctxs[0]!r}")

# 2) 花括号 / $ 配对
if body.count("{") != body.count("}"):
    issues.append(f"braces unbalanced: {{={body.count('{')} }}={body.count('}')}")
dollars = len(re.findall(r"(?<!\\)\$", body))
if dollars % 2:
    issues.append(f"odd number of $ ({dollars})")

# 3) 摘要不得含引用
abstract = body.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
if r"\cite" in abstract:
    issues.append("abstract contains \\cite")

# 4) 内联表格数
n_tab = len(re.findall(r"\\begin\{tabular\}", body))
n_fig = len(re.findall(r"\\includegraphics", body))
print(f"inline tabular: {n_tab} (expect 10)")
print(f"figures       : {n_fig}")

# 5) 关键宏存在性（源文件通过 \input{results_macros} 引入时一并读入）
macros_needed = ["WinFirst", "WinLast", "A2FirstMin", "A2FirstMax", "DropA3",
                 "NumRuns", "DetWindowGain", "DetWindowPos", "DetLRrange",
                 "CalWinFirst", "CalWinLast"]
src = t
if r"\input{results_macros}" in t:
    macro_file = p.parent / "results_macros.tex"
    if macro_file.exists():
        src += "\n" + macro_file.read_text(encoding="utf-8")
for m in macros_needed:
    if rf"\{m}" in body and f"\\newcommand{{\\{m}}}" not in src:
        issues.append(f"macro \\{m} used but not defined")

# 6) 常见 LaTeX 危险
if r"\input{" in body and "tables" in "".join(re.findall(r"\\input\{([^}]*)\}", body)):
    pass
if re.search(r"\\begin\{document\}", t) and not re.search(r"\\end\{document\}", t):
    issues.append("\\begin{document} without \\end")

print("-" * 60)
for i in issues:
    print("ISSUE:", i)
print("RESULT:", "FAIL" if issues else "PASS")
sys.exit(1 if issues else 0)
