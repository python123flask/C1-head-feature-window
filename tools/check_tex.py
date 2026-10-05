"""静态检查 paper/main.tex：环境配对、宏定义、引用配对。"""
import re
from collections import Counter
from pathlib import Path

t = Path("paper/main.tex").read_text(encoding="utf-8")
begins = Counter(re.findall(r"\\begin\{(\w+\*?)\}", t))
ends = Counter(re.findall(r"\\end\{(\w+\*?)\}", t))
print("begin:", dict(begins))
print("end  :", dict(ends))
print("unbalanced:", {k: (begins[k], ends[k]) for k in set(begins) | set(ends)
                      if begins[k] != ends[k]})

macros = set(re.findall(r"\\newcommand\{\\(\w+)\}",
                        Path("paper/results_macros.tex").read_text(encoding="utf-8")))
own = set(re.findall(r"\\newcommand\{\\(\w+)\}", t))
used = set(re.findall(r"\\([A-Z][A-Za-z0-9]*)", t))
missing = sorted(u for u in used if u not in macros and u not in own)
print(f"defined macros: {len(macros)} (generated) + {len(own)} (local)")
print("possibly undefined macros:", missing)

cites = set()
for m in re.findall(r"\\cite\{([^}]*)\}", t):
    cites.update(x.strip() for x in m.split(","))
bib = set(re.findall(r"\\bibitem\{([^}]*)\}", t))
print("cited but no bibitem:", sorted(cites - bib))
print("bibitem unused:", sorted(bib - cites))

figs = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]*)\}", t)
for f in figs:
    p = Path("paper") / f
    print(("OK   " if p.exists() else "MISS "), f)

tabs = re.findall(r"\\input\{([^}]*)\}", t)
for f in tabs:
    p = Path("paper") / f
    print(("OK   " if p.exists() else "MISS "), f)
