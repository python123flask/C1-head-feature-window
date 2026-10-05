"""同时更新 Zenodo 记录描述 与 GitHub Release notes（同源文字，run 分账口径统一）。

用法:
  $env:ZENODO_TOKEN='...'; $env:GH_TOKEN='ghp_...'; python -X utf8 tools/update_deposits.py

原则：只改 description/body 两个字段；DOI、tag、creators、license 一律不动，并在结尾回读校验。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

ZTOK = os.environ.get("ZENODO_TOKEN") or ""
GTOK = os.environ.get("GH_TOKEN") or ""
RECORD = "23157275"
REPO = "python123flask/C1-head-feature-window"

# ---------------- 统一口径的新描述（纯文本，给 GitHub） ----------------
PLAIN = """What this is
Pre-registered pilot study behind the Neurocomputing submission "Head first, features later: a transient window of classifier-head de-specialization under uniform label exposure and its effect on task adaptation" (single author: Xianzhe Liu, Hunan University).

Contents (739 files, 23.7 MB)
- experiments/ — three-phase simulation code (sim.py), models, data loaders, metrics, and 16 unit tests with gold-value checks
- analysis/ — pre-registered gate decision (decision_gate.py -> GATE.json), figure/table/number generators (all manuscript numbers are injected from raw metrics), budget, transplant and consistency analyses
- results/runs/<run_id>/ — the research data: for each of the 207 executed runs, config.json, metrics.jsonl (raw per-evaluation readings), summary.json
- pilot/ — pre-registered plan, deviations log with calibration evidence, cross-model code-review record, resource snapshot
- paper/ — manuscript source; submission/ — formatted submission package (single-file LaTeX, Figure_1..11.pdf, highlights, cover letter)
- results/logs/ — attempt / batch / test logs

Run accounting (the manuscript quotes 189 confirmatory runs everywhere)
207 runs were executed and all succeeded, broken down as:
- 189 confirmatory runs (blocks P1-P14) — the set certified by the pre-registered gates A1-A4;
- 1 timing run (P0, resource measurement);
- 17 calibration/smoke runs (S-*, protocol calibration performed before the gates were frozen).

Excluded on purpose
- .venv/, data/ — third-party datasets (CIFAR-10/100, SVHN, Fashion-MNIST, UCI Gas Sensor Array Drift); each is cited as [dataset] in the manuscript and rebuildable via tools/hf_to_cache.py
- results/runs/*/checkpoints/ (2.1 GB) — not required to reproduce any number
- release/ — packaging artifact produced by tools/make_release.py

Reproduce
    python experiments/pilot/test_sim.py     # 16 unit tests (gold values)
    python analysis/decision_gate.py         # pre-registered gates -> GATE.json
    python analysis/visualization.py all     # regenerate all figures (vector PDF)
    python analysis/make_tables.py           # regenerate all tables
    python analysis/paper_macros.py          # regenerate the manuscript's numbers

Status
Pre-registered pilot; claim ceiling: "evidence of a separable mechanism on these benchmarks". 207/207 executed runs succeeded (189 confirmatory + 1 timing + 17 calibration); gates A1-A4 pass (analysis/GATE.json -> advance).
"""

# ---------------- 同一份内容的 HTML（给 Zenodo description） ----------------
def to_html(plain: str) -> str:
    blocks = []
    for seg in plain.strip().split("\n\n"):
        lines = seg.split("\n")
        head, rest = lines[0], lines[1:]
        inner = f"<strong>{head}</strong><br />" + "<br />".join(rest)
        blocks.append(f"<p>{inner}</p>")
    return "\n".join(blocks)


HTML = to_html(PLAIN)


def http(method, url, payload=None, token=None, ctype="application/json"):
    data = None
    if payload is not None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    h = {"User-Agent": "deposit-updater", "Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    if data is not None:
        h["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            body = r.read().decode()
            return r.status, (json.loads(body) if body.strip() else {})
    except urllib.error.HTTPError as e:
        txt = e.read().decode()
        try:
            return e.code, json.loads(txt)
        except Exception:
            return e.code, {"raw": txt[:500]}


def update_zenodo() -> bool:
    base = f"https://zenodo.org/api/records/{RECORD}"
    st, rec = http("GET", base)
    if st != 200:
        print(f"[zenodo] GET record failed {st}: {rec}")
        return False
    meta = rec.get("metadata", {})
    before = {"doi": meta.get("doi"), "title": meta.get("title"),
              "creators": [c.get("name") for c in meta.get("creators", [])],
              "version": meta.get("version"), "license": meta.get("license")}
    print(f"[zenodo] before: {before}")

    new_meta = {k: v for k, v in meta.items()
                if k not in ("doi", "relations")}     # 系统字段不回传
    new_meta["description"] = HTML
    payload = {"metadata": new_meta}

    st, resp = http("PUT", base, payload, ZTOK)
    print(f"[zenodo] PUT /records/{RECORD} -> {st}")
    if st not in (200, 201, 202):
        # 回退：走 draft 流程
        st2, draft = http("GET", base + "/draft")
        print(f"[zenodo] GET /draft -> {st2}")
        if st2 == 404:
            st2, draft = http("POST", base + "/draft", {}, ZTOK)
            print(f"[zenodo] POST /draft -> {st2}")
        if st2 in (200, 201):
            st3, r3 = http("PUT", base + "/draft", payload, ZTOK)
            print(f"[zenodo] PUT /draft -> {st3} {str(r3)[:200]}")
            if st3 in (200, 201, 202):
                st4, r4 = http("POST", base + "/draft/actions/publish", {}, ZTOK)
                print(f"[zenodo] publish -> {st4} {str(r4)[:200]}")
                st = st4
        if st not in (200, 201, 202):
            print(f"[zenodo] FAILED: {str(resp)[:400]}")
            return False

    st, rec2 = http("GET", base)
    m2 = rec2.get("metadata", {})
    after = {"doi": m2.get("doi"), "title": m2.get("title"),
             "creators": [c.get("name") for c in m2.get("creators", [])],
             "version": m2.get("version"), "license": m2.get("license")}
    ok_desc = "189 confirmatory + 1 timing + 17 calibration" in (m2.get("description") or "")
    ok_gone = "To be specified" not in (m2.get("description") or "")
    same_doi = after["doi"] == before["doi"]
    print(f"[zenodo] after : {after}  (revision={rec2.get('revision')})")
    print(f"[zenodo] checks: doi_unchanged={same_desc_doi(same_doi)} "
          f"description_updated={ok_desc} stale_license_text_removed={ok_gone}")
    ok = same_doi and ok_desc and ok_gone
    if not ok:
        print(f"[zenodo] metadata preserved: "
              f"title={after['title'] == before['title']} "
              f"creators={after['creators'] == before['creators']} "
              f"version={after['version'] == before['version']} "
              f"license={after['license'] == before['license']}")
    return ok


def same_desc_doi(x: bool) -> bool:
    return x


def update_github() -> bool:
    base = f"https://api.github.com/repos/{REPO}"
    st, rels = http("GET", base + "/releases", token=GTOK)
    if st != 200:
        print(f"[github] list releases failed {st}: {str(rels)[:200]}")
        return False
    target = next((r for r in rels if r["tag_name"] == "v2.0"), None)
    if target is None:
        print(f"[github] no release with tag v2.0; found: {[r['tag_name'] for r in rels]}")
        return False
    st, r = http("PATCH", f"{base}/releases/{target['id']}", {"body": PLAIN}, GTOK)
    if st != 200:
        print(f"[github] PATCH release failed {st}: {str(r)[:200]}")
        return False
    st, r2 = http("GET", f"{base}/releases/{target['id']}", token=GTOK)
    body = r2.get("body", "")
    ok = "189 confirmatory + 1 timing + 17 calibration" in body and "To be specified" not in body
    print(f"[github] body updated: {ok} (chars={len(body)})")
    print(f"[github] tag={r2.get('tag_name')} target={r2.get('target_commitish')} "
          f"url={r2.get('html_url')}")
    return ok


def main():
    if not ZTOK:
        sys.exit("set $env:ZENODO_TOKEN")
    if not GTOK:
        sys.exit("set $env:GH_TOKEN")
    z = update_zenodo()
    g = update_github()
    print("=" * 70)
    print(f"zenodo={'OK' if z else 'FAIL'}  github={'OK' if g else 'FAIL'}")
    print("[action] 两个 token 已在本次会话使用完毕，用完请到各自后台吊销。")
    return 0 if (z and g) else 1


if __name__ == "__main__":
    sys.exit(main())
