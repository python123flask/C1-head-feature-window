"""工具函数：种子、日志、检查点、资源测量。

C1 · 可塑性恢复窗口先导实验 (GPU 扩展版)
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parents[2]      # 项目根目录
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
LOGS = RESULTS / "logs"
DATA = ROOT / "data"
for _p in (RUNS, LOGS, DATA):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 种子
def set_seed(seed: int) -> None:
    """固定 python / numpy / torch 的随机源（单进程内确定性）。"""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(base_seed: int, step: int) -> int:
    """为训练流生成派生种子（批洗牌等）。"""
    return (base_seed * 1000003 + step * 7919) % (2**31 - 1)


# ---------------------------------------------------------------- json / jsonl
def write_json(path: Path | str, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def read_json(path: Path | str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not json serializable: {type(o)}")


def append_jsonl(path: Path | str, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path: Path | str) -> list:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_log(path: Path | str, msg: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


# ---------------------------------------------------------------- 检查点 (npz)
def state_to_npz(state: dict) -> dict:
    """state_dict -> {key 中 '.' 替换为 '_' 的 ndarray}。"""
    return {k.replace(".", "__"): v.detach().cpu().numpy() for k, v in state.items()}


def npz_to_state(arrays: dict) -> dict:
    return {k.replace("__", "."): torch.from_numpy(np.asarray(v)) for k, v in arrays.items()}


def save_checkpoint(model, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **state_to_npz(model.state_dict()))


def load_checkpoint(model, path: Path | str) -> None:
    with np.load(path) as z:
        state = npz_to_state({k: z[k] for k in z.files})
    model.load_state_dict(state)


# ---------------------------------------------------------------- 资源测量
def peak_rss_mb() -> float:
    """进程峰值 RSS（MB）：Windows 用 K32GetProcessMemoryInfo，其余用 resource。"""
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        fn = getattr(k32, "K32GetProcessMemoryInfo", None)
        if fn is None:
            fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        ok = fn(k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        if ok:
            return counters.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB->MB on linux
    except Exception:
        return float("nan")


def gpu_peak_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def machine_snapshot() -> dict:
    """preflight 资源快照（pilot/preflight/resource-snapshot.json）。"""
    import platform
    import shutil

    snap = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": {
            "cpu": {"logical_cores": os.cpu_count(), "model": platform.processor() or platform.machine()},
            "memory": {},
            "disk": {},
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }
    try:
        import shutil

        total, used, free = shutil.disk_usage(str(ROOT))
        snap["hardware"]["disk"] = {
            "total_gb": round(total / 1e9, 1),
            "available_gb": round(free / 1e9, 1),
            "used_gb": round(used / 1e9, 1),
        }
    except Exception as e:
        snap["hardware"]["disk"] = {"error": str(e)}
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        snap["hardware"]["memory"] = {
            "total_gb": round(stat.ullTotalPhys / 1e9, 1),
            "available_gb": round(stat.ullAvailPhys / 1e9, 1),
        }
    except Exception:
        pass
    if torch.cuda.is_available():
        snap["hardware"]["gpu"] = {
            "count": torch.cuda.device_count(),
            "model": torch.cuda.get_device_name(0),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        }
    return snap
