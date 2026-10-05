"""打包可复现发布物（供 Option C 数据入库 / GitHub+Zenodo 存档用）。

包含：代码与文档 + 逐点原始指标与配置；
排除：.venv、data/（第三方数据集）、checkpoints（2.1 GB，非复现必需）。

用法: python tools/make_release.py
输出: release/ （目录）+ release/C1-head-feature-window-release.zip
"""
from __future__ import annotations

import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "release"
NAME = "C1-head-feature-window-release"

INCLUDE_DIRS = ["experiments", "analysis", "tools", "pilot", "results"]  # 代码+文档+逐点数据
INCLUDE_FILES = ["paper/main.tex", "paper/references.bib",
                 "paper/results_macros.tex"]                          # 稿件源
EXCLUDE_PARTS = {".venv", "data", "__pycache__", ".git", "checkpoints"}

RELEASE_README = """# C1 — Head/feature timescale separation under uniform label exposure

Reproducibility release for the Neurocomputing submission.

## What is inside
- `experiments/pilot/` — simulation code (sim.py), models, data loaders, metrics, unit tests
- `experiments/batch.py` — queue grid generation and (sharded) execution + orchestration state files
- `analysis/` — gate decision (decision_gate.py), figures (visualization.py), tables, macros, all analysis scripts
- `tools/` — dataset fetch/convert, calibration, and the four self-checkers
- `pilot/` — pre-registered plan, deviations log with calibration evidence, code-review record, resource snapshot
- `results/runs/<run_id>/` — for all 207 runs: `config.json`, `metrics.jsonl` (per-evaluation raw readings),
  `summary.json` (status, wall time, peak RSS/GPU). **These are the research data.**
- `results/logs/` — attempt/batch/test logs
- `paper/` — manuscript source (values are injected by `analysis/paper_macros.py` from the metrics above)

## What is NOT inside (and why)
- `data/` — third-party datasets (CIFAR-10/100, SVHN, Fashion-MNIST, UCI Gas Sensor Array Drift);
  fetch scripts are included instead; each dataset is cited as `[dataset]` in the manuscript.
- `checkpoints/*.npz` (2.1 GB) — not required to reproduce any reported number; regenerate on demand.
- `.venv/` — Python environment (requirements: Python 3.11, PyTorch 2.13+cu132, torchvision, numpy, pyarrow, scipy, matplotlib, scikit-learn).

## How to reproduce
```
python tools/hf_to_cache.py                     # convert downloaded datasets to cache
python experiments/pilot/test_sim.py            # 16 unit tests (gold-value checks)
python experiments/batch.py --make-grid         # build the frozen queue
python experiments/batch.py --run --blocks P1-main --workers 1 --shard 0/3 --state-id w0
python analysis/decision_gate.py                # pre-registered gates -> GATE.json
python analysis/visualization.py all            # regenerate all figures (PDF vectors)
python analysis/make_tables.py                  # regenerate all tables
python analysis/paper_macros.py                 # regenerate the numbers in the manuscript
python tools/check_submission.py paper/main.tex # manuscript self-check
```

## Metadata (fill before deposit)
- Title: Head first, features later: a transient window of classifier-head de-specialization ...
- Authors: Xianzhe Liu (Hunan University)
- License: <choose an OSI license, e.g. MIT / Apache-2.0>
- DOI: <fill after Zenodo/OSF deposit, then cite it in the manuscript's Data availability statement>
- Version / date: {date}
"""

SOURCES = [
    ("experiments", ["experiments"]),
    ("analysis", ["analysis"]),
    ("tools", ["tools"]),
    ("pilot", ["pilot"]),
    ("paper", ["paper"]),
    ("results", ["results"]),
]


def keep(p: Path) -> bool:
    parts = set(p.parts)
    if parts & EXCLUDE_PARTS:
        return False
    name = p.name
    if name.endswith(".pyc") or name == "GATE.json.bak":
        return False
    if p.suffix == ".npz" and "checkpoints" in parts:
        return False
    if p.suffix == ".parquet":
        return False
    if name.endswith(".pdf") and "figures" in parts:
        return False          # 位图/矢量图由脚本重新生成
    if name == "COVER_LETTER.md":
        return False          # 投稿材料不属于复现包
    return True


def main():
    if REL.exists():
        shutil.rmtree(REL)
    (REL / NAME).mkdir(parents=True)

    n_files = 0
    n_bytes = 0
    for src in INCLUDE_DIRS + ["paper"]:
        s = ROOT / src
        if not s.exists():
            continue
        for p in s.rglob("*"):
            if not p.is_file() or not keep(p):
                continue
            if src == "paper" and p.name not in ("main.tex", "references.bib",
                                                 "results_macros.tex"):
                continue
            dst = REL / NAME / p.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            n_files += 1
            n_bytes += dst.stat().st_size

    (REL / NAME / "RELEASE_README.md").write_text(
        RELEASE_README.format(date=time.strftime("%Y-%m-%d")), encoding="utf-8")
    n_files += 1

    zpath = REL / f"{NAME}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((REL / NAME).rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(REL))
    print(f"-> {REL / NAME}")
    print(f"   files={n_files} raw={n_bytes / 1e6:.1f} MB  zip={zpath.stat().st_size / 1e6:.1f} MB")
    print("   [x] checkpoints (2.1 GB) excluded; [x] third-party data excluded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
