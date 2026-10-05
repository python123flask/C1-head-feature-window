"""检查源文件中表格 input 是否重复、引用是否齐全。"""
import collections
import re
from pathlib import Path

t = Path("paper/main.tex").read_text(encoding="utf-8")
inp = re.findall(r"\\input\{[^}]*tables/([^}]+)\}", t)
refs = re.findall(r"\\ref\{(tab:[^}]+)\}", t)
print("inputs :", inp)
print("dup inputs:", [k for k, v in collections.Counter(inp).items() if v > 1])
print("refs   :", sorted(set(refs)))
# 每个 input 的 tab_label 是否被 ref
for name in inp:
    body = Path("analysis/tables") / name
    if not body.exists():
        body = body.with_suffix(".tex")
    lab = re.findall(r"\\label\{(tab:[^}]+)\}", body.read_text(encoding="utf-8"))
    for l in lab:
        print(f"  {name}: label={l} cited={'YES' if l in refs else 'NO'}")
