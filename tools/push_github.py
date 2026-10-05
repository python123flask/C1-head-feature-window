"""把仓库推送到 GitHub（token 仅从环境变量读取，绝不写入任何文件）。

用法（PowerShell）:
  $env:GH_TOKEN='ghp_xxx'; python -X utf8 tools/push_github.py [repo_name]

要点：
- .git/config 里不写入 token（用 http.<url>.extraHeader 一次性传认证）
- 输出对 token 做掩码
- 结束后提示吊销 token
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"
DEFAULT_NAME = "C1-head-feature-window"
DEFAULT_DESC = ("Pre-registered pilot: classifier-head / feature timescale separation "
                "under uniform label exposure (207 runs, raw per-evaluation metrics, "
                "gates, figures and manuscript source)")

_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
if not _token:
    sys.exit("ERROR: set $env:GH_TOKEN first")


def mask(s: str) -> str:
    return s.replace(_token, "***TOKEN***")


def api(method: str, path: str, payload: dict | None = None):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": f"Bearer {_token}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "c1-release-push"},
        method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode()
            return r.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def run(cmd: list[str], **kw) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", **kw)
    return p.returncode, mask((p.stdout or "") + (p.stderr or ""))


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NAME

    # 1) git 仓库与身份
    if not (ROOT / ".git").exists():
        rc, out = run(["git", "init", "-b", "main"])
        assert rc == 0, out
        print("[git] initialised on 'main'")
    run(["git", "config", "user.name", "Xianzhe Liu"])
    run(["git", "config", "user.email", "liuxianzhe@hnu.edu.cn"])
    run(["git", "config", "core.quotepath", "false"])

    # 3) 暂存（先 add 再体检，只检查 git 实际会提交的文件）
    run(["git", "add", "-A"])
    rc, out = run(["git", "ls-files", "-z"])
    staged = [f for f in out.split("\0") if f.strip()]
    big = []
    for rel in staged:
        p = ROOT / rel
        if p.is_file() and p.stat().st_size > 90 * 1024 * 1024:
            big.append(f"{rel} ({p.stat().st_size / 1e6:.0f}MB)")
    if big:
        sys.exit("ERROR: GitHub rejects files >100MB: " + ", ".join(big))
    total = sum((ROOT / f).stat().st_size for f in staged if (ROOT / f).is_file())
    print(f"[check] staged={len(staged)} files, {total / 1e6:.1f} MB, none >90MB")

    rc, out = run(["git", "commit", "-m",
                   "Pre-registered pilot: head/feature timescale separation under "
                   "uniform label exposure (207 runs + gates + figures + manuscript)"])
    if rc != 0 and "nothing to commit" not in out:
        sys.exit("commit failed:\n" + out)
    print(f"[commit] staged files: {len(staged)}")

    # 4) 身份与仓库
    rc, body = api("GET", "/user")
    if rc != 200:
        sys.exit(f"token rejected (HTTP {rc}): {body.get('message')}")
    login = body["login"]
    print(f"[auth] authenticated as: {login}")

    rc, body = api("GET", f"/repos/{login}/{repo}")
    if rc == 404:
        rc, body = api("POST", "/user/repos", {
            "name": repo, "description": DEFAULT_DESC,
            "private": False, "auto_init": False, "has_issues": True,
            "has_wiki": False})
        if rc not in (201, 422):
            sys.exit(f"create repo failed (HTTP {rc}): {body.get('message')}")
        print(f"[repo] created: https://github.com/{login}/{repo} (public)")
    elif rc == 200:
        print(f"[repo] exists: https://github.com/{login}/{repo} "
              f"(private={body.get('private')})")
    else:
        sys.exit(f"repo lookup failed (HTTP {rc}): {body.get('message')}")

    # 5) 推送（认证用一次性 header，不写进 .git/config）
    remote = f"https://github.com/{login}/{repo}.git"
    run(["git", "remote", "remove", "origin"])
    run(["git", "remote", "add", "origin", remote])
    cred = base64.b64encode(f"x-access-token:{_token}".encode()).decode()
    rc, out = run(["git", "-c",
                   f"http.https://github.com/.extraHeader=Authorization: Basic {cred}",
                   "push", "-u", "origin", "main"])
    print("[push]", out.strip()[-500:])
    if rc != 0:
        sys.exit("push failed")

    # 6) 校验 + 确认 config 无凭据
    rc, body = api("GET", f"/repos/{login}/{repo}")
    cfg = (ROOT / ".git" / "config").read_text(encoding="utf-8")
    assert _token not in cfg, "token leaked into .git/config!"
    print(f"[ok] default_branch={body.get('default_branch')} "
          f"size={body.get('size')}KB private={body.get('private')}")
    print(f"[ok] url: https://github.com/{login}/{repo}")
    print("[action required] push 用的 Personal Access Token 已出现在对话中，"
          "请立即到 GitHub → Settings → Developer settings → Tokens 删除/吊销它。")


if __name__ == "__main__":
    main()
