# C1 — Head/feature timescale separation under uniform label exposure

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
- Version / date: 2026-10-05
