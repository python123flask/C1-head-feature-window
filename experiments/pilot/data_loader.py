"""数据加载：CIFAR-10（旧/新任务）、SVHN（低共享任务）、UCI Gas Sensor Drift。

设计要点：
- 原始图像以 uint8 存 CPU，归一化/增强在 GPU 上进行（省内存、省 Windows 进程开销）。
- probe 用 train 划分拟合、test 划分评估（排除 E4 测量伪影）。
- head 槽位（输出维）固定 10：高共享新任务复用旧任务槽位 0-4，
  低共享（SVHN 10 类）占用槽位 0-9 —— 见 pilot/DEVIATIONS.md 记录的适配。
"""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils import DATA, append_log, LOGS

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
SVHN_MEAN = [0.4377, 0.4438, 0.4728]
SVHN_STD = [0.1980, 0.2010, 0.1970]

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
SVHN_TRAIN_URL = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
SVHN_TEST_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
GAS_URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    append_log(LOGS / "data.log", f"download {url} -> {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


# ---------------------------------------------------------------- TaskData
class TaskData:
    """一个分类任务：train/test 样本 + 标签 + 归一化统计。

    kind='image'  : x 为 uint8 (N,3,32,32)，归一化用通道均值/方差，可 GPU 增强
    kind='vector' : x 为 float32 (N,D)，归一化用训练集逐维 z-score
    """

    def __init__(self, name, x_train, y_train, x_test, y_test, mean, std,
                 num_classes, slots, head_dim=10):
        self.name = name
        self.kind = "image" if np.asarray(x_train).ndim == 4 else "vector"
        if self.kind == "image":
            self.x_train = torch.as_tensor(np.ascontiguousarray(x_train), dtype=torch.uint8)
            self.x_test = torch.as_tensor(np.ascontiguousarray(x_test), dtype=torch.uint8)
            self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        else:
            self.x_train = torch.as_tensor(np.ascontiguousarray(x_train), dtype=torch.float32)
            self.x_test = torch.as_tensor(np.ascontiguousarray(x_test), dtype=torch.float32)
            if mean is None:      # 训练集 z-score
                mu = self.x_train.mean(0, keepdim=True)
                sd = self.x_train.std(0, keepdim=True) + 1e-6
            else:
                mu = torch.tensor(mean, dtype=torch.float32).view(1, -1)
                sd = torch.tensor(std, dtype=torch.float32).view(1, -1)
            self.mean, self.std = mu, sd
        self.y_train = torch.as_tensor(y_train, dtype=torch.long)
        self.y_test = torch.as_tensor(y_test, dtype=torch.long)
        self.num_classes = num_classes
        self.slots = slots          # 该任务使用的 head 输出槽位（list[int]）
        self.head_dim = head_dim
        assert len(self.x_train) == len(self.y_train)
        assert len(self.x_test) == len(self.y_test)

    # ---- 基本信息
    @property
    def n_train(self):
        return len(self.y_train)

    @property
    def n_test(self):
        return len(self.y_test)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "num_classes": self.num_classes,
            "slots": list(self.slots),
            "head_dim": self.head_dim,
            "train_per_class": np.bincount(self.y_train.numpy(), minlength=self.num_classes).tolist(),
            "test_per_class": np.bincount(self.y_test.numpy(), minlength=self.num_classes).tolist(),
        }

    # ---- GPU 批次准备
    def prepare(self, x_uint8: torch.Tensor, device, augment: bool = False) -> torch.Tensor:
        if self.kind == "vector":
            x = x_uint8.to(device, non_blocking=True).float()
        else:
            x = x_uint8.to(device, non_blocking=True).float().div_(255.0)
            if augment:
                x = _augment(x)
        mean = self.mean.to(device)
        std = self.std.to(device)
        return (x - mean) / std

    def train_batch(self, idx: torch.Tensor, device, augment: bool = False):
        x = self.x_train[idx]
        y = self.y_train[idx]
        return self.prepare(x, device, augment), y.to(device, non_blocking=True)

    def test_tensors(self, device):
        return (
            self.prepare(self.x_test, device, augment=False),
            self.y_test.to(device, non_blocking=True),
        )


def _augment(x: torch.Tensor) -> torch.Tensor:
    """GPU 上的 RandomCrop(32, padding=4, reflect) + RandomHorizontalFlip。"""
    B = x.shape[0]
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")            # (B,3,40,40)
    top = torch.randint(0, 9, (B,), device=x.device)       # 0..8
    left = torch.randint(0, 9, (B,), device=x.device)
    # 先按行裁剪：index 形状需与输入各维一致（除被 gather 的维）
    rows = top.view(B, 1, 1, 1) + torch.arange(32, device=x.device).view(1, 1, 32, 1)
    x = x.gather(2, rows.expand(B, 3, 32, 40))             # (B,3,32,40)
    cols = left.view(B, 1, 1, 1) + torch.arange(32, device=x.device).view(1, 1, 1, 32)
    x = x.gather(3, cols.expand(B, 3, 32, 32))             # (B,3,32,32)
    flip = torch.rand(B, 1, 1, 1, device=x.device) < 0.5
    return torch.where(flip, x.flip(3), x)


# ---------------------------------------------------------------- 训练流
class TrainStream:
    """按 seed 循环洗牌的任务训练流（无 DataLoader/多进程，Windows 下更稳）。"""

    def __init__(self, task: TaskData, batch_size: int, seed: int, augment: bool = True):
        self.task = task
        self.batch_size = batch_size
        self.augment = augment
        self.gen = torch.Generator().manual_seed(seed)
        self._order = torch.randperm(task.n_train, generator=self.gen)
        self._pos = 0

    def next(self, device):
        t = self.task
        if self._pos + self.batch_size > t.n_train:
            self._order = torch.randperm(t.n_train, generator=self.gen)
            self._pos = 0
        idx = self._order[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        return t.train_batch(idx, device, self.augment)


# ---------------------------------------------------------------- 子集缓存
def _cache_file(name: str) -> Path:
    p = DATA / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.npz"


def _cache_get(name: str):
    p = _cache_file(name)
    if p.exists():
        try:
            with np.load(p) as z:
                return {k: z[k] for k in z.files}
        except Exception:
            return None
    return None


def _cache_put(name: str, arrays: dict) -> None:
    np.savez_compressed(_cache_file(name), **arrays)


# ---------------------------------------------------------------- CIFAR-10
def _ensure_cifar() -> Path:
    root = DATA
    full = root / "cache" / "cifar_full.npz"
    if full.exists():
        return root
    tar = root / "cifar-10-python.tar.gz"
    extracted = root / "cifar-10-batches-py"
    if not (extracted / "data_batch_1").exists():
        _download(CIFAR_URL, tar)
        append_log(LOGS / "data.log", "extract cifar-10")
        with tarfile.open(tar, "r:gz") as tf:
            tf.extractall(root)
    return extracted


def _load_cifar_batches(folder: Path):
    """优先读 data/cache/cifar_full.npz（由 tools/hf_to_cache.py 从官方 HF 镜像转换）。"""
    full = DATA / "cache" / "cifar_full.npz"
    if full.exists():
        with np.load(full) as z:
            return z["x_train"], z["y_train"]
    import pickle

    xs, ys = [], []
    files = sorted([p for p in folder.iterdir() if p.name.startswith("data_batch_")])
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        xs.append(d[b"data"])
        ys.append(d[b"labels"])
    x = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.uint8)
    y = np.array(np.concatenate(ys), dtype=np.int64)
    return x, y


def _load_cifar_test(folder: Path):
    full = DATA / "cache" / "cifar_full.npz"
    if full.exists():
        with np.load(full) as z:
            return z["x_test"], z["y_test"]
    import pickle

    with open(folder / "test_batch", "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8)
    y = np.array(d[b"labels"], dtype=np.int64)
    return x, y


def _subset_by_class(x, y, classes, per_class, per_class_test, rng):
    tr_idx, te_idx = [], []
    all_idx = {c: np.where(y == c)[0] for c in classes}
    # 每类：取 per_class_test 作测试、per_class 作训练（不重叠）
    for c in classes:
        idx = rng.permutation(all_idx[c])
        te_pool = idx[:per_class_test]
        tr_idx.append(idx[per_class_test:per_class_test + per_class])
        te_idx.append(te_pool)
    tr = np.concatenate(tr_idx)
    te = np.concatenate(te_idx)
    return x[tr], y[tr], x[te], y[te]


def make_cifar_tasks(seed: int = 0, per_class_train: int = 1000,
                     per_class_test: int = 500, head_dim: int = 10,
                     old_classes=None, new_classes=None):
    """返回 (old_task, new_task_high)。

    默认：old = CIFAR-10 类 0-4（标签 0-4，槽位 0-4）；
          new_high = 类 5-9 → 标签 0-4（复用同一组槽位，构造头冲突）。
    """
    old_classes = list(old_classes) if old_classes else [0, 1, 2, 3, 4]
    new_classes = list(new_classes) if new_classes else [5, 6, 7, 8, 9]
    key = (f"cifar_sub_seed{seed}_tr{per_class_train}_te{per_class_test}"
           f"_old{''.join(map(str, old_classes))}_new{''.join(map(str, new_classes))}")
    c = _cache_get(key)
    if c is None:
        folder = _ensure_cifar()
        rng = np.random.default_rng(seed)
        x_tr, y_tr = _load_cifar_batches(folder)
        x_te, y_te = _load_cifar_test(folder)
        ox_tr, oy_tr, ox_te, oy_te = _subset_by_class(
            x_tr, y_tr, old_classes, per_class_train, per_class_test, rng)
        nx_tr, ny_tr, nx_te, ny_te = _subset_by_class(
            x_tr, y_tr, new_classes, per_class_train, per_class_test, rng)
        remap = {cl: i for i, cl in enumerate(new_classes)}
        ny_tr = np.array([remap[v] for v in ny_tr], dtype=np.int64)
        ny_te = np.array([remap[v] for v in ny_te], dtype=np.int64)
        c = dict(ox_tr=ox_tr, oy_tr=oy_tr, ox_te=ox_te, oy_te=oy_te,
                 nx_tr=nx_tr, ny_tr=ny_tr, nx_te=nx_te, ny_te=ny_te)
        _cache_put(key, c)
    n_old = len(old_classes)
    old = TaskData("cifar10_old", c["ox_tr"], c["oy_tr"], c["ox_te"], c["oy_te"],
                   CIFAR_MEAN, CIFAR_STD, num_classes=n_old, slots=list(range(n_old)),
                   head_dim=head_dim)
    new = TaskData("cifar10_new", c["nx_tr"], c["ny_tr"], c["nx_te"], c["ny_te"],
                   CIFAR_MEAN, CIFAR_STD, num_classes=n_old, slots=list(range(n_old)),
                   head_dim=head_dim)
    return old, new


# ---------------------------------------------------------------- SVHN
def _ensure_svhn():
    """优先读 data/cache/svhn_full.npz（tools/hf_to_cache.py 转换），否则下载官方 .mat。"""
    full = DATA / "cache" / "svhn_full.npz"
    if full.exists():
        with np.load(full) as z:
            return z["x_train"], z["y_train"], z["x_test"], z["y_test"]
    from scipy.io import loadmat

    tr = _download(SVHN_TRAIN_URL, DATA / "train_32x32.mat")
    te = _download(SVHN_TEST_URL, DATA / "test_32x32.mat")
    dtr, dte = loadmat(tr), loadmat(te)
    x_tr = np.transpose(dtr["X"], (3, 2, 0, 1)).astype(np.uint8)  # -> N,3,32,32
    y_tr = dtr["y"].reshape(-1).astype(np.int64) % 10              # 10 -> 0
    x_te = np.transpose(dte["X"], (3, 2, 0, 1)).astype(np.uint8)
    y_te = dte["y"].reshape(-1).astype(np.int64) % 10
    return x_tr, y_tr, x_te, y_te


def make_svhn_task(seed: int = 0, per_class_train: int = 1000,
                   per_class_test: int = 500, head_dim: int = 10) -> TaskData:
    """低共享新任务：SVHN 10 类（占满槽位 0-9）。"""
    key = f"svhn_sub_seed{seed}_tr{per_class_train}_te{per_class_test}"
    c = _cache_get(key)
    if c is None:
        x_tr, y_tr, x_te, y_te = _ensure_svhn()
        rng = np.random.default_rng(seed + 1)

        def take(x, y, per, rng_):
            idx = []
            for cl in range(10):
                pool = np.where(y == cl)[0]
                idx.append(rng_.permutation(pool)[:per])
            idx = np.concatenate(idx)
            return x[idx], y[idx]

        xtr, ytr = take(x_tr, y_tr, per_class_train, rng)
        xte, yte = take(x_te, y_te, per_class_test, rng)
        c = dict(xtr=xtr, ytr=ytr, xte=xte, yte=yte)
        _cache_put(key, c)
    return TaskData("svhn_new", c["xtr"], c["ytr"], c["xte"], c["yte"], SVHN_MEAN, SVHN_STD,
                    num_classes=10, slots=list(range(10)), head_dim=head_dim)


def make_new_task(kind: str, seed: int = 0, head_dim: int = 10, config=None):
    """kind: 'high' (与旧任务同域的其余类) | 'low' (跨域新任务: svhn / fashion-mnist)。"""
    cfg = config or {}
    if kind == "high":
        if cfg.get("dataset", "cifar10") == "cifar100":
            _, new = make_cifar100_tasks(
                seed=seed, head_dim=head_dim,
                per_class_train=int(cfg.get("per_class_train", 500)),
                per_class_test=int(cfg.get("per_class_test", 100)),
                old_classes=cfg.get("old_classes"), new_classes=cfg.get("new_classes"))
            return new
        _, new = make_cifar_tasks(
            seed=seed, head_dim=head_dim,
            per_class_train=int(cfg.get("per_class_train", 1000)),
            old_classes=cfg.get("old_classes"), new_classes=cfg.get("new_classes"))
        return new
    if kind == "low":
        if cfg.get("low_dataset", "svhn") == "fmnist":
            return make_fmnist_task(seed=seed, head_dim=head_dim,
                                    per_class_train=int(cfg.get("per_class_train", 1000)),
                                    per_class_test=int(cfg.get("per_class_test", 500)))
        return make_svhn_task(seed=seed, head_dim=head_dim)
    raise ValueError(kind)


# ---------------------------------------------------------------- 传感器漂移（UCI Gas）
def load_gas_sensor():
    """UCI Gas Sensor Array Drift (id 224)：返回 dict batch_id -> (X, y)。

    文件为 LIBSVM 格式：label idx:val ... (128 维)，文件名 batch1.dat ... batch10.dat。
    """
    import zipfile

    zpath = DATA / "gas_sensor_array_drift.zip"
    _download(GAS_URL, zpath)
    out_dir = DATA / "gas_sensor_drift"
    if not (out_dir / "Dataset").exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(out_dir)
    return _parse_gas(out_dir)


def _parse_gas(out_dir: Path) -> dict:
    import re

    files = sorted(out_dir.rglob("batch*.dat"),
                   key=lambda p: int(re.search(r"batch(\d+)", p.name).group(1)))
    if not files:
        raise FileNotFoundError(f"no batch*.dat under {out_dir}")
    out = {}
    for p in files:
        bi = int(re.search(r"batch(\d+)", p.name).group(1))
        ys, rows = [], []
        with open(p, "r") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                ys.append(int(float(parts[0])))
                feats = np.zeros(128, dtype=np.float32)
                for tok in parts[1:]:
                    k, v = tok.split(":")
                    i = int(k) - 1
                    if 0 <= i < 128:
                        feats[i] = float(v)
                rows.append(feats)
        out[bi] = (np.stack(rows), np.asarray(ys, dtype=np.int64))
    return out


def make_sensor_tasks(seed: int = 0, head_dim: int = 6,
                      batch_old: int = 1, batch_new: int = 5,
                      test_frac: float = 0.2):
    """真实传感器漂移任务：批次 batch_old（旧）-> 批次 batch_new（漂移后新任务）。"""
    key = f"gas_b{batch_old}_to_b{batch_new}_seed{seed}_te{test_frac}"
    c = _cache_get(key)
    if c is None:
        batches = load_gas_sensor()
        if batch_old not in batches or batch_new not in batches:
            raise ValueError(f"available batches {sorted(batches)}; need {batch_old},{batch_new}")
        c = {}
        for tag, b in (("old", batch_old), ("new", batch_new)):
            X, y = batches[b]
            classes = sorted(set(y.tolist()))
            remap = {cl: i for i, cl in enumerate(classes)}
            y = np.array([remap[v] for v in y], dtype=np.int64)
            rng = np.random.default_rng(seed + (0 if tag == "old" else 1))
            tr_idx, te_idx = [], []
            for cl in range(len(classes)):
                pool = rng.permutation(np.where(y == cl)[0])
                n_te = max(1, int(len(pool) * test_frac))
                te_idx.append(pool[:n_te])
                tr_idx.append(pool[n_te:])
            tr = np.concatenate(tr_idx)
            te = np.concatenate(te_idx)
            c[f"{tag}_Xtr"], c[f"{tag}_ytr"] = X[tr], y[tr]
            c[f"{tag}_Xte"], c[f"{tag}_yte"] = X[te], y[te]
            c[f"{tag}_ncls"] = len(classes)
        _cache_put(key, c)
    old = TaskData("gas_batch%d" % batch_old, c["old_Xtr"], c["old_ytr"],
                   c["old_Xte"], c["old_yte"], None, None,
                   num_classes=int(c["old_ncls"]), slots=list(range(int(c["old_ncls"]))),
                   head_dim=head_dim)
    new = TaskData("gas_batch%d" % batch_new, c["new_Xtr"], c["new_ytr"],
                   c["new_Xte"], c["new_yte"], None, None,
                   num_classes=int(c["new_ncls"]), slots=list(range(int(c["new_ncls"]))),
                   head_dim=head_dim)
    return old, new


# ---------------------------------------------------------------- CIFAR-100（外部效度）
CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD = [0.2673, 0.2564, 0.2762]
FMNIST_MEAN = [0.2860, 0.2860, 0.2860]
FMNIST_STD = [0.3530, 0.3530, 0.3530]


def _ensure_cifar100():
    full = DATA / "cache" / "cifar100_full.npz"
    if not full.exists():
        raise FileNotFoundError(f"{full} missing; run tools/hf_to_cache.py")
    with np.load(full) as z:
        return z["x_train"], z["y_train"], z["x_test"], z["y_test"]


def make_cifar100_tasks(seed: int = 0, per_class_train: int = 500,
                        per_class_test: int = 100, head_dim: int = 25,
                        old_classes=None, new_classes=None):
    """CIFAR-100 任务对（外部效度）。默认：旧任务 = 细类 0-24，新任务 = 25-49。

    两组各 25 类，标签都重映射到 0-24 → **复用同一组头槽位**（与 CIFAR-10 主实验一致）。
    """
    old_classes = list(old_classes) if old_classes else list(range(0, 25))
    new_classes = list(new_classes) if new_classes else list(range(25, 50))
    key = (f"c100_sub_seed{seed}_tr{per_class_train}_te{per_class_test}"
           f"_old{old_classes[0]}-{old_classes[-1]}_new{new_classes[0]}-{new_classes[-1]}_off")
    c = _cache_get(key)
    if c is None:
        x_tr, y_tr, x_te, y_te = _ensure_cifar100()
        rng = np.random.default_rng(seed)

        def split(classes_):
            """官方划分：train 池取 per_class_train，官方 test 池取 per_class_test（互不重叠）。"""
            tr, te = [], []
            for cl in classes_:
                ptr = rng.permutation(np.where(y_tr == cl)[0])[:per_class_train]
                pte = rng.permutation(np.where(y_te == cl)[0])[:per_class_test]
                tr.append(ptr)
                te.append(pte)
            tr, te = np.concatenate(tr), np.concatenate(te)
            return x_tr[tr], y_tr[tr], x_te[te], y_te[te]

        ox_tr, oy_tr, ox_te, oy_te = split(old_classes)
        nx_tr, ny_tr, nx_te, ny_te = split(new_classes)
        remap = {cl: i for i, cl in enumerate(new_classes)}
        ny_tr = np.array([remap[v] for v in ny_tr], dtype=np.int64)
        ny_te = np.array([remap[v] for v in ny_te], dtype=np.int64)
        c = dict(ox_tr=ox_tr, oy_tr=oy_tr, ox_te=ox_te, oy_te=oy_te,
                 nx_tr=nx_tr, ny_tr=ny_tr, nx_te=nx_te, ny_te=ny_te)
        _cache_put(key, c)
    n_old = len(old_classes)
    old = TaskData("cifar100_old", c["ox_tr"], c["oy_tr"], c["ox_te"], c["oy_te"],
                   CIFAR100_MEAN, CIFAR100_STD, num_classes=n_old,
                   slots=list(range(n_old)), head_dim=head_dim)
    new = TaskData("cifar100_new", c["nx_tr"], c["ny_tr"], c["nx_te"], c["ny_te"],
                   CIFAR100_MEAN, CIFAR100_STD, num_classes=n_old,
                   slots=list(range(n_old)), head_dim=head_dim)
    return old, new


# ---------------------------------------------------------------- Fashion-MNIST（跨域新任务 2）
def _ensure_fmnist():
    full = DATA / "cache" / "fmnist_full.npz"
    if not full.exists():
        raise FileNotFoundError(f"{full} missing; run tools/hf_to_cache.py")
    with np.load(full) as z:
        return z["x_train"], z["y_train"], z["x_test"], z["y_test"]


def make_fmnist_task(seed: int = 0, per_class_train: int = 1000,
                     per_class_test: int = 500, head_dim: int = 10) -> TaskData:
    """跨域新任务 2：Fashion-MNIST 10 类（28x28 灰度已上采样为 3x32x32）。"""
    key = f"fmnist_sub_seed{seed}_tr{per_class_train}_te{per_class_test}"
    c = _cache_get(key)
    if c is None:
        x_tr, y_tr, x_te, y_te = _ensure_fmnist()
        rng = np.random.default_rng(seed + 7)

        def take(x, y, per, rng_):
            idx = []
            for cl in range(10):
                pool = np.where(y == cl)[0]
                idx.append(rng_.permutation(pool)[:per])
            idx = np.concatenate(idx)
            return x[idx], y[idx]

        xtr, ytr = take(x_tr, y_tr, per_class_train, rng)
        xte, yte = take(x_te, y_te, per_class_test, rng)
        c = dict(xtr=xtr, ytr=ytr, xte=xte, yte=yte)
        _cache_put(key, c)
    return TaskData("fmnist_new", c["xtr"], c["ytr"], c["xte"], c["yte"],
                    FMNIST_MEAN, FMNIST_STD, num_classes=10,
                    slots=list(range(10)), head_dim=head_dim)
