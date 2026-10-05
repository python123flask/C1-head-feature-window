"""创建 GitHub Release（自动建 tag v1.0 + 写规范 release notes）。

用法: $env:GH_TOKEN='ghp_xxx'; python -X utf8 tools/make_release_tag.py [tag]
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

TOK = os.environ.get("GH_TOKEN") or ""
REPO = "python123flask/C1-head-feature-window"
TAG = sys.argv[1] if len(sys.argv) > 1 else "v1.0"
TITLE = "C1-head-feature-window v1.0"

BODY = """## What this is

Pre-registered pilot study behind the Neurocomputing submission
**"Head first, features later: a transient window of classifier-head
de-specialization under uniform label exposure and its effect on task
adaptation"** (single author: Xianzhe Liu, Hunan University).

## Contents (739 files, 23.7 MB)

- `experiments/` — three-phase simulation code (sim.py), models, data loaders,
  metrics, and 16 unit tests with gold-value checks
- `analysis/` — pre-registered gate decision (`decision_gate.py` -> `GATE.json`),
  figure/table/number generators (all manuscript numbers are injected from raw
  metrics), budget, transplant and consistency analyses
- `results/runs/<run_id>/` — **the research data**: for each of the 207 runs
  `config.json`, `metrics.jsonl` (raw per-evaluation readings), `summary.json`
- `pilot/` — pre-registered plan, deviations log with calibration evidence,
  cross-model code-review record, resource snapshot
- `paper/` — manuscript source; `submission/` — formatted submission package
  (single-file LaTeX, Figure_1..11.pdf, highlights, cover letter)
- `results/logs/` — attempt / batch / test logs

## Excluded on purpose

- `.venv/`, `data/` — third-party datasets (CIFAR-10/100, SVHN, Fashion-MNIST,
  UCI Gas Sensor Array Drift); each is cited as `[dataset]` in the manuscript
  and rebuildable via `tools/hf_to_cache.py`
- `results/runs/*/checkpoints/` (2.1 GB) — not required to reproduce any number
- `release/` — packaging artifact produced by `tools/make_release.py`

## Reproduce

```bash
python experiments/pilot/test_sim.py     # 16 unit tests (gold values)
python analysis/decision_gate.py         # pre-registered gates -> GATE.json
python analysis/visualization.py all     # regenerate all figures (vector PDF)
python analysis/make_tables.py           # regenerate all tables
python analysis/paper_macros.py          # regenerate the manuscript's numbers
```

## Status

Pre-registered pilot; claim ceiling: *"evidence of a separable mechanism on
these benchmarks"*. **207/207 runs succeeded; gates A1-A4 pass**
(`analysis/GATE.json` -> `advance`).

## License

To be specified: add a `LICENSE` file (MIT / Apache-2.0 recommended) before
deposited to Zenodo for a DOI.
"""


def api(method, path, payload=None):
    req = urllib.request.Request(
        "https://api.github.com" + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": f"Bearer {TOK}", "User-Agent": "release-bot",
                 "Accept": "application/vnd.github+json"},
        method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def main():
    if not TOK:
        sys.exit("set $env:GH_TOKEN")
    # 已存在则列出
    rc, rels = api("GET", f"/repos/{REPO}/releases")
    if rc == 200 and rels:
        for r in rels:
            print(f"[exists] tag={r['tag_name']} url={r['html_url']}")
        print("[info] 已有 release；如需新建请换 tag 名")
        return 0
    rc, body = api("POST", f"/repos/{REPO}/releases", {
        "tag_name": TAG,
        "target_commitish": "main",
        "name": TITLE,
        "body": BODY,
        "draft": False,
        "prerelease": False,
    })
    if rc not in (201, 200):
        sys.exit(f"create release failed HTTP {rc}: {body.get('message')}")
    print(f"[ok] {body['html_url']}")
    print(f"     tag={body['tag_name']} target=main assets={len(body.get('assets', []))}")
    print("[action] 请吊销对话中暴露过的 PAT；如需 Zenodo DOI，把这个 release 连到仓库即可")
    return 0


if __name__ == "__main__":
    sys.exit(main())
