"""创建 v2.1 Release（新 run 分账 + P15/P16），然后轮询 Zenodo 取新版本 DOI。

用法: $env:GH_TOKEN='ghp_xxx'; python -X utf8 tools/make_release_v21.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

TOK = os.environ.get("GH_TOKEN") or ""
REPO = "python123flask/C1-head-feature-window"
TAG = "v2.1"

BODY = """What this is
Pre-registered study behind the Neurocomputing submission "Head first, features later: a transient window of classifier-head de-specialization under uniform label exposure and its effect on task adaptation" (single author: Xianzhe Liu, Hunan University).

What is new in v2.1 (838 files, 27.6 MB)
- Post-hoc robustness experiments (DEVIATIONS D21/D22): P15 deterministic-platform cross-transplant (6 source runs + 6 transplant runs, process-level rerun noise = 0 by construction) and P16 SmallCNN width sensitivity (33.8k / 114k / 412k parameters, 12 runs).
- Time-to-accuracy readout added to the manuscript (reaching 0.65 accuracy takes 360 -> 245 optimizer steps after a short exposure, 5/5 seeds).
- Transplant statistics rewritten (paired contrast, d-values, sign tests); claim wording de-piloted; run accounting unified via macros.
- P14 last-layer-reset baseline data (5 runs) and the width/deterministic extensions are now included.

Contents
- experiments/ - three-phase simulation code, models (width_mult option), data loaders, metrics,16 unit tests
- analysis/ - pre-registered gate decision (decision_gate.py -> GATE.json), figure/table/number generators, budget, transplant and deterministic-transplant analyses
- results/runs/<run_id>/ - the research data: config.json, metrics.jsonl (raw per-evaluation readings), summary.json for every run
- pilot/ - pre-registered plan, deviations log (D1-D22) with calibration evidence, cross-model code-review record
- paper/ and submission/ - manuscript source and formatted submission package
- results/logs/ - attempt / batch / test logs

Run accounting (the manuscript quotes 189 confirmatory runs everywhere)
231 runs were executed and all succeeded:
- 189 confirmatory runs (blocks P1-P14) - the set certified by the pre-registered gates A1-A4;
- 1 timing run (P0, resource measurement);
- 17 calibration/smoke runs (S-*, performed before the gates were frozen);
- 24 post-hoc robustness runs outside the pre-registered gate set (P15 deterministic transplant: 6 + 6; P16 width: 12).

Excluded on purpose
- .venv/, data/ - third-party datasets (CIFAR-10/100, SVHN, Fashion-MNIST, UCI Gas Sensor Array Drift); each is cited as [dataset] in the manuscript and rebuildable via tools/hf_to_cache.py
- results/runs/*/checkpoints/ (2.1 GB) - not required to reproduce any reported number
- release/ - packaging artifact produced by tools/make_release.py

Reproduce
    python experiments/pilot/test_sim.py     # 16 unit tests (gold values)
    python analysis/decision_gate.py         # pre-registered gates -> GATE.json
    python analysis/visualization.py all     # regenerate all figures (vector PDF)
    python analysis/make_tables.py           # regenerate all tables
    python analysis/paper_macros.py          # regenerate the manuscript's numbers
    python analysis/det_transplant_analysis.py   # P15 deterministic transplant

Status
Pre-registered study; claim ceiling: "evidence of a separable mechanism on these benchmarks". 231/231 executed runs succeeded (189 confirmatory + 1 timing + 17 calibration + 24 robustness); gates A1-A4 pass (analysis/GATE.json -> advance).
"""


def api(method, url, payload=None):
    h = {"Authorization": f"Bearer {TOK}", "User-Agent": "release",
         "Accept": "application/vnd.github+json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "zenodo-poll",
                                               "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception:
        return {}


def main():
    if not TOK:
        sys.exit("set $env:GH_TOKEN")

    # 已有 v2.1？
    rc, rels = api("GET", f"https://api.github.com/repos/{REPO}/releases")
    existing = next((r for r in (rels or []) if r.get("tag_name") == TAG), None)
    if existing:
        print(f"[github] release {TAG} already exists: {existing['html_url']}")
    else:
        rc, body = api("POST", f"https://api.github.com/repos/{REPO}/releases", {
            "tag_name": TAG, "target_commitish": "main",
            "name": "C1-head-feature-window v2.1", "body": BODY,
            "draft": False, "prerelease": False})
        if rc not in (200, 201):
            sys.exit(f"create release failed HTTP {rc}: {body.get('message')}")
        print(f"[github] release created: {body['html_url']}")

    # 轮询 Zenodo 新版本
    print("[zenodo] polling for the new version (ingest can take a minute)...")
    for i in range(40):
        data = get_json("https://zenodo.org/api/records/23157275/versions")
        hits = [h for h in (data.get("hits", {}).get("hits") or [])
                if (h.get("metadata", {}).get("version") == TAG)]
        if hits:
            rec = hits[0]
            m = rec.get("metadata", {})
            print(f"[zenodo] NEW VERSION FOUND")
            print(f"  version DOI : {m.get('doi')}")
            print(f"  concept DOI : {rec.get('conceptdoi')}")
            print(f"  url         : https://zenodo.org/records/{rec.get('id')}")
            print(f"  files       : {[f.get('key') for f in rec.get('files', [])]}")
            print(f"  creators    : {[c.get('name') for c in m.get('creators', [])]}")
            print(f"  license     : {(m.get('license') or {}).get('id')}")
            print(f"  description has 231 accounting: "
                  f"{'231/231 executed runs succeeded' in (m.get('description') or '')}")
            return 0
        time.sleep(10)
    print("[zenodo] TIMEOUT (400s): release may not have been ingested; "
          "check Zenodo -> GitHub integration is still enabled")
    return 1


if __name__ == "__main__":
    sys.exit(main())
