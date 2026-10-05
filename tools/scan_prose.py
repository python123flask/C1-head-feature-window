"""定位需统一的用词。"""
import re
from pathlib import Path

t = Path("paper/main.tex").read_text(encoding="utf-8")
for w in ["analyse", "analyzes"]:
    print(f"=== {w} ===")
    for m in re.finditer(w, t):
        s = max(0, m.start() - 120)
        print(repr(t[s:m.end() + 120].replace("\n", " ")))
