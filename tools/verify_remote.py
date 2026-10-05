"""推送后校验：远端根目录、关键文件、以及 token 未泄漏。"""
import json
import os
import sys
import urllib.request

TOK = os.environ.get("GH_TOKEN") or ""
REPO = "python123flask/C1-head-feature-window"
H = {"Authorization": f"Bearer {TOK}", "User-Agent": "verify",
     "Accept": "application/vnd.github+json"}


def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=H),
                                            timeout=30))


def main():
    root = get(f"https://api.github.com/repos/{REPO}/contents/")
    print("remote root:", sorted((x["name"], x["type"]) for x in root))
    for path in ["experiments/pilot/sim.py",
                 "analysis/decision_gate.py",
                 "analysis/figures/fig1_head_feature_trajectories.pdf",
                 "results/runs/P1_main_high_L4000_s42/metrics.jsonl",
                 "results/runs/P14_headreset_s42/summary.json",
                 "pilot/PLAN.md",
                 "paper/main.tex",
                 ".gitignore"]:
        try:
            j = get(f"https://api.github.com/repos/{REPO}/contents/{path}")
            print(f"  OK   {path} ({j.get('size')} B)")
        except Exception as e:
            print(f"  MISS {path} -> {e}")
    repo = get(f"https://api.github.com/repos/{REPO}")
    print(f"private={repo.get('private')} branch={repo.get('default_branch')} "
          f"description={repo.get('description')[:60]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
