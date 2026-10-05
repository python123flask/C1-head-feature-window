"""把 Zenodo draft 里被清掉的字段按正确 schema 补回，然后发布并校验。

用法: $env:ZENODO_TOKEN='...'; python -X utf8 tools/fix_zenodo_draft.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

TOK = os.environ.get("ZENODO_TOKEN") or ""
REC = "23157275"
BASE = f"https://zenodo.org/api/records/{REC}"
NEW_TEXT = "189 confirmatory + 1 timing + 17 calibration"


def call(method, url, payload=None, accept="application/json"):
    h = {"User-Agent": "draft-fixer", "Accept": accept}
    if TOK:
        h["Authorization"] = f"Bearer {TOK}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=45) as f:
            b = f.read().decode()
            return f.status, (json.loads(b) if b.strip() else {})
    except urllib.error.HTTPError as e:
        t = e.read().decode()
        try:
            return e.code, json.loads(t)
        except Exception:
            return e.code, {"raw": t[:400]}


def draft_meta():
    st, d = call("GET", BASE + "/draft")
    return st, d.get("metadata", {})


def main():
    if not TOK:
        sys.exit("set $env:ZENODO_TOKEN")

    st, m = draft_meta()
    if st != 200:
        sys.exit(f"no draft (HTTP {st}); 先在 UI 点 Edit 创建 draft")
    print(f"[draft] keys={sorted(m.keys())} creators={m.get('creators')}")

    # related_identifiers 的输入键名是 relation_type（不是输出的 relation）
    rid = (m.get("related_identifiers") or [{}])[0]
    fixed_ri = [{
        "identifier": "https://github.com/python123flask/C1-head-feature-window/tree/v2.0",
        "relation_type": "isSupplementTo",
        "resource_type": "software",
        "scheme": "url",
    }]

    candidates = [
        {"name": "Liu, Xianzhe", "affiliation": None},
        {"name": "Liu, Xianzhe"},
        {"name": "Liu, Xianzhe", "type": "personal", "affiliation": None},
        {"name": "Liu, Xianzhe", "type": "personal"},
        {"name": "Liu, Xianzhe", "affiliation": []},
        {"name": "Liu, Xianzhe", "type": "personal", "affiliation": []},
        {"name": "Liu, Xianzhe", "identifiers": []},
    ]

    picked = None
    for i, cr in enumerate(candidates, 1):
        payload = {"metadata": {**m, "creators": [cr],
                                "related_identifiers": fixed_ri}}
        st, r = call("PUT", BASE + "/draft", payload)
        st2, d2 = draft_meta()
        got = (d2.get("creators") or [None])[0]
        got_ri = (d2.get("related_identifiers") or [{}])[0]
        ok_cr = bool(got) and got.get("name") == "Liu, Xianzhe"
        ok_ri = "relation_type" in got_ri
        print(f"  candidate {i} {json.dumps(cr)} -> PUT {st} creators={got} ri_ok={ok_ri}")
        if ok_cr:
            picked = cr
            print(f"  [picked] {json.dumps(cr)}")
            break

    if not picked:
        print("[fail] creators 无法通过 API 落库；改回 UI 修改（Edit → Metadata → "
              "Creators 确认是 Liu, Xianzhe → Save/Publish）")
        return 1

    st, pub = call("POST", BASE + "/draft/actions/publish", {})
    print(f"[publish] -> {st} {str(pub)[:300] if st not in (200, 201) else 'published'}")
    if st not in (200, 201):
        return 1

    st, rec = call("GET", BASE)
    m2 = rec.get("metadata", {})
    checks = {
        "doi": m2.get("doi") == "10.5281/zenodo.23157275",
        "conceptdoi": rec.get("conceptdoi") == "10.5281/zenodo.23157274",
        "creators": [c.get("name") for c in m2.get("creators", [])] == ["Liu, Xianzhe"],
        "title_unchanged": m2.get("title", "").startswith("python123flask/C1-head"),
        "version": m2.get("version") == "v2.0",
        "license": (m2.get("license") or {}).get("id") == "cc-by-4.0",
        "resource_type": (m2.get("resource_type") or {}).get("type") == "software",
        "custom_repo": (m2.get("custom") or {}).get("code:codeRepository", "").endswith(
            "C1-head-feature-window"),
        "new_desc": NEW_TEXT in (m2.get("description") or ""),
        "stale_removed": "To be specified" not in (m2.get("description") or ""),
    }
    print("[verify] revision =", rec.get("revision"))
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'XX '} {k}")
    print("RESULT:", "PASS" if all(checks.values()) else "FAIL")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
