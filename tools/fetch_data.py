"""带测速与多线程分片续传的数据抓取（慢网络 / 限速镜像友好）。

用法: python -u tools/fetch_data.py [which...]
which: cifar | svhn_train | svhn_test | all(默认)
"""
from __future__ import annotations

import hashlib
import sys
import threading
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
UA = {"User-Agent": "Mozilla/5.0 (compatible; dataset-fetcher)"}

TARGETS = {
    "cifar": [
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "http://cs231n.stanford.edu/cifar-10-python.tar.gz",
        "https://data.deepai.org/cifar10.zip",
    ],
    "svhn_train": [
        "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
        "https://ufldl.stanford.edu/housenumbers/train_32x32.mat",
    ],
    "svhn_test": [
        "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
        "https://ufldl.stanford.edu/housenumbers/test_32x32.mat",
    ],
    # hf-mirror.com（中国大陆可达；官方 CIFAR-10 / SVHN 仓库）
    "hf_cifar_train": [
        "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
    ],
    "hf_cifar_test": [
        "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
    ],
    "hf_svhn_train": [
        "https://hf-mirror.com/datasets/ufldl-stanford/svhn/resolve/main/cropped_digits/train-00000-of-00001.parquet",
    ],
    "hf_svhn_test": [
        "https://hf-mirror.com/datasets/ufldl-stanford/svhn/resolve/main/cropped_digits/test-00000-of-00001.parquet",
    ],
    "hf_cifar100_train": [
        "https://hf-mirror.com/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet",
    ],
    "hf_cifar100_test": [
        "https://hf-mirror.com/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet",
    ],
    "hf_fmnist_train": [
        "https://hf-mirror.com/datasets/zalando-datasets/fashion_mnist/resolve/main/fashion_mnist/train-00000-of-00001.parquet",
    ],
    "hf_fmnist_test": [
        "https://hf-mirror.com/datasets/zalando-datasets/fashion_mnist/resolve/main/fashion_mnist/test-00000-of-00001.parquet",
    ],
}
NAMES = {"cifar": "cifar-10-python.tar.gz",
         "svhn_train": "train_32x32.mat",
         "svhn_test": "test_32x32.mat",
         "hf_cifar_train": "hf/cifar10_train.parquet",
         "hf_cifar_test": "hf/cifar10_test.parquet",
         "hf_svhn_train": "hf/svhn_train.parquet",
         "hf_svhn_test": "hf/svhn_test.parquet",
         "hf_cifar100_train": "hf/cifar100_train.parquet",
         "hf_cifar100_test": "hf/cifar100_test.parquet",
         "hf_fmnist_train": "hf/fmnist_train.parquet",
         "hf_fmnist_test": "hf/fmnist_test.parquet"}

# 权威大小（字节）；0 = 未知（跳过大小校验，仅校验 >0）
EXPECTED = {"cifar": 170498071, "svhn_train": 182042733, "svhn_test": 64275384,
            "hf_cifar_train": 119705255, "hf_cifar_test": 23940850,
            "hf_svhn_train": 135589380, "hf_svhn_test": 47003111,
            "hf_cifar100_train": 118518617, "hf_cifar100_test": 23772751,
            "hf_fmnist_train": 30931277, "hf_fmnist_test": 5175617}

ALIASES = {
    "hf": ["hf_cifar_train", "hf_cifar_test", "hf_svhn_train", "hf_svhn_test"],
    "ext": ["hf_cifar100_train", "hf_cifar100_test", "hf_fmnist_train", "hf_fmnist_test"],
    "all": ["hf_cifar_train", "hf_cifar_test", "hf_svhn_train", "hf_svhn_test",
            "hf_cifar100_train", "hf_cifar100_test", "hf_fmnist_train", "hf_fmnist_test",
            "cifar", "svhn_train", "svhn_test"],
}


def _req(url: str, rng: str | None = None):
    r = urllib.request.Request(url, headers=UA)
    if rng:
        r.add_header("Range", rng)
    return r


def head_info(url: str, timeout=15):
    try:
        with urllib.request.urlopen(_req(url), timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            accept = r.headers.get("Accept-Ranges", "")
            return (int(n) if n else None), (accept == "bytes")
    except Exception:
        return None, False


def probe_speed(url: str, seconds: float = 8.0) -> tuple[float, int, bool]:
    """返回 (MB/s, total_len, supports_range)。"""
    t0 = time.time()
    n = 0
    total, ranged = None, False
    try:
        with urllib.request.urlopen(_req(url), timeout=15) as r:
            total = r.headers.get("Content-Length")
            total = int(total) if total else None
            ranged = r.headers.get("Accept-Ranges", "") == "bytes" or r.status == 206
            while time.time() - t0 < seconds:
                d = r.read(1 << 20)
                if not d:
                    break
                n += len(d)
    except Exception as e:
        print(f"  probe fail {type(e).__name__}: {e}", flush=True)
        return 0.0, total, ranged
    el = max(time.time() - t0, 1e-3)
    return n / 1e6 / el, total, ranged


def _part(dest: Path, url: str, i: int = 0) -> Path:
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    return dest.parent / f"{dest.name}.{h}.part{i}"


def _stream(url: str, start: int, end: int, part: Path, stop: threading.Event | None = None):
    mode = "ab" if part.exists() else "wb"
    have = part.stat().st_size if part.exists() else 0
    rng = f"bytes={start + have}-{end}" if end >= start + have else None
    if end < start:
        rng = None
    with urllib.request.urlopen(_req(url, rng), timeout=30) as r, open(part, mode) as f:
        while True:
            if stop is not None and stop.is_set():
                return
            try:
                d = r.read(1 << 20)
            except Exception:
                return
            if not d:
                return
            f.write(d)


def download_parallel(url: str, dest: Path, total: int, nstreams: int = 8,
                      seconds: int = 3600, attempts: int = 6) -> bool:
    """支持 Range 时分片并行；整体分批重试直至完成或超时。"""
    for attempt in range(attempts):
        chunks = []
        base = total // nstreams
        for i in range(nstreams):
            s = i * base
            e = (i + 1) * base - 1 if i < nstreams - 1 else total - 1
            chunks.append((s, e, _part(dest, url, i)))
        done0 = sum(p.stat().st_size for _, _, p in chunks if p.exists())
        if done0 >= total:
            break
        t0 = time.time()
        last = [t0, done0]

        def reporter():
            while time.time() - t0 < seconds:
                time.sleep(15)
                d = sum(p.stat().st_size for _, _, p in chunks if p.exists())
                rate = (d - last[1]) / max(time.time() - last[0], 1e-3) / 1e6
                print(f"  {dest.name}: {d/1e6:.1f}/{total/1e6:.1f}MB "
                      f"({100*d/total:.0f}%) {rate:.2f}MB/s", flush=True)
                last[0], last[1] = time.time(), d

        threading.Thread(target=reporter, daemon=True).start()
        th = [threading.Thread(target=_stream, args=(url, s, e, p), daemon=True)
              for s, e, p in chunks]
        for t in th:
            t.start()
        budget = min(seconds, 300)      # 每轮最多 5 分钟，之后检查续传
        for t in th:
            t.join(timeout=budget)
        done = sum(p.stat().st_size for _, _, p in chunks if p.exists())
        print(f"  attempt {attempt + 1}: {done/1e6:.1f}/{total/1e6:.1f}MB", flush=True)
        if done >= total:
            break
    done = sum(p.stat().st_size for _, _, p in chunks if p.exists())
    if done < total:
        print(f"  incomplete {done}/{total}; keeping shards for resume", flush=True)
        return False
    with open(dest, "wb") as out:
        for _, _, p in chunks:
            with open(p, "rb") as f:
                out.write(f.read())
            p.unlink()
    return True


def download_single(url: str, dest: Path, seconds: int = 3600) -> bool:
    part = _part(dest, url, 0)
    _, ranged = head_info(url)
    have = part.stat().st_size if part.exists() else 0
    if have and not ranged:
        part.unlink(missing_ok=True)
        have = 0
    t0 = time.time()
    last = [t0, have]
    try:
        rng = f"bytes={have}-" if have else None
        with urllib.request.urlopen(_req(url, rng), timeout=30) as r, open(part, "ab" if have else "wb") as f:
            while time.time() - t0 < seconds:
                d = r.read(1 << 20)
                if not d:
                    break
                f.write(d)
                now = time.time()
                if now - last[0] >= 15:
                    nown = part.stat().st_size
                    print(f"  {dest.name}: {nown/1e6:.1f}MB "
                          f"{(nown-last[1])/max(now-last[0],1e-3)/1e6:.2f}MB/s", flush=True)
                    last = [now, nown]
    except Exception as e:
        print(f"  single fail {type(e).__name__}: {e}", flush=True)
        return False
    return dest_total_ok(part, url)


def dest_total_ok(part: Path, url: str) -> bool:
    total, _ = head_info(url)
    have = part.stat().st_size if part.exists() else 0
    return bool(total) and have >= total


def finalize(dest: Path, url: str, nstreams: int) -> bool:
    parts = [_part(dest, url, i) for i in range(nstreams)]
    total, _ = head_info(url)
    if not total:
        return False
    have = [p.stat().st_size if p.exists() else 0 for p in parts]
    if sum(have) >= total:
        with open(dest, "wb") as out:
            for p, n in zip(parts, have):
                if not p.exists():
                    continue
                with open(p, "rb") as f:
                    out.write(f.read(n))
                p.unlink()
        return True
    return False


def fetch(which: str) -> bool:
    dest = DATA / NAMES[which]
    exp = EXPECTED[which]                      # 0 = 大小未知，仅要求 >0
    size = dest.stat().st_size if dest.exists() else -1
    if exp == 0:
        if size > 0:
            print(f"[skip] {dest.name} exists ({size} B)", flush=True)
            return True
    elif size == exp:
        print(f"[skip] {dest.name} exists", flush=True)
        return True
    if dest.exists():
        print(f"[warn] {dest.name} size {size} != {exp}; re-download", flush=True)
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    best = None
    for url in TARGETS[which]:
        for trial in range(3):
            speed, total, ranged = probe_speed(url, seconds=6)
            print(f"  {speed:6.2f} MB/s range={ranged} (try {trial + 1}) {url}", flush=True)
            if speed > 0:
                break
            time.sleep(3)
        if speed > 0 and (best is None or speed > best[1]):
            best = (url, speed, total, ranged)
    if best is None:
        # 兜底：取第一个能给出 Content-Length 的 URL
        for url in TARGETS[which]:
            total, _ranged = head_info(url)
            if total:
                best = (url, 0.0, total, False)
                break
    if best is None:
        print(f"[FAIL] no reachable mirror for {which}", flush=True)
        return False
    url, speed, total, ranged = best
    print(f"[use] {url} ({speed:.2f} MB/s)", flush=True)
    if ranged and total:
        ok = download_parallel(url, dest, total, nstreams=8)
        if not ok:
            ok = finalize(dest, url, 8)
    else:
        ok = download_single(url, dest)
        if not ok:
            ok = dest_total_ok(_part(dest, url, 0), url)
            if ok:
                _part(dest, url, 0).rename(dest)
    fsize = dest.stat().st_size if dest.exists() else -1
    if exp == 0:                                # 大小未知：只要求 >0
        ok = fsize > 0
    else:
        if fsize == exp:
            ok = True
        elif not dest.exists():
            ok = finalize(dest, url, 4)
        if dest.exists() and dest.stat().st_size != exp:
            print(f"[warn] final size {dest.stat().st_size} != {exp}", flush=True)
            ok = False
    print(f"[{'ok' if ok else 'FAIL'}] {dest.name}", flush=True)
    return bool(ok)


def main():
    which = sys.argv[1:] or ["hf"]
    out = []
    for w in which:
        out.extend(ALIASES.get(w, [w]))
    ok = True
    for w in out:
        if not fetch(w):
            ok = False
    print("[done] all ok" if ok else "[done] some failed", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
