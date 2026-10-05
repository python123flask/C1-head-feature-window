"""把 hf-mirror 上的官方 CIFAR-10 / SVHN parquet 转成 data/cache/*.npz。

产物：
  data/cache/cifar_full.npz : x_train(50000,3,32,32) y_train / x_test(10000,...) y_test
  data/cache/svhn_full.npz  : 同上（SVHN cropped_digits）
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
HF = ROOT / "data" / "hf"


def resize_gray_to_rgb32(im: "Image.Image"):
    """灰度 28x28 -> RGB 32x32（Fashion-MNIST）。"""
    im = im.convert("L").resize((32, 32), Image.BILINEAR).convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def table_to_arrays(path: Path, gray_to_rgb: bool = False):
    t = pq.read_table(path)
    cols = t.column_names
    print(f"[{path.name}] columns={cols} rows={t.num_rows}", flush=True)
    # 图像列
    img_col = next(c for c in cols if c.lower() in ("img", "image"))
    lab_col = next(c for c in cols if c.lower() in ("label", "labels", "fine_label"))
    imgs = t.column(img_col).to_pylist()
    labels = t.column(lab_col).to_pylist()
    n = len(imgs)
    x = np.empty((n, 3, 32, 32), dtype=np.uint8)
    for i, im in enumerate(imgs):
        if isinstance(im, dict):
            im = im.get("bytes") or im.get("path")
        if isinstance(im, (bytes, bytearray)):
            im = Image.open(io.BytesIO(im))
        elif isinstance(im, str):
            im = Image.open(im)
        if gray_to_rgb:
            arr = resize_gray_to_rgb32(im)
        else:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            if arr.shape != (32, 32, 3):
                arr = np.asarray(Image.fromarray(arr).resize((32, 32)), dtype=np.uint8)
        x[i] = arr.transpose(2, 0, 1)
    y = np.asarray(labels, dtype=np.int64)
    print(f"  labels range {y.min()}..{y.max()} unique {len(np.unique(y))}", flush=True)
    return x, y


def save(name: str, xtr, ytr, xte, yte):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{name}.npz"
    np.savez_compressed(out, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
    print(f"[saved] {out} train={xtr.shape} test={xte.shape}", flush=True)


def main():
    if (HF / "cifar10_train.parquet").exists():
        xtr, ytr = table_to_arrays(HF / "cifar10_train.parquet")
        xte, yte = table_to_arrays(HF / "cifar10_test.parquet")
        assert xtr.shape[0] == 50000 and xte.shape[0] == 10000, (xtr.shape, xte.shape)
        save("cifar_full", xtr, ytr, xte, yte)
    if (HF / "svhn_train.parquet").exists():
        xtr, ytr = table_to_arrays(HF / "svhn_train.parquet")
        xte, yte = table_to_arrays(HF / "svhn_test.parquet")
        # SVHN 原始标签为 1..10（10 表示数字 0）；若已规范化为 0..9 则不动
        if ytr.max() == 10:
            ytr = np.where(ytr == 10, 0, ytr)
            yte = np.where(yte == 10, 0, yte)
        elif ytr.max() == 9 and ytr.min() == 0:
            pass
        else:
            raise ValueError(f"unexpected SVHN label range {ytr.min()}..{ytr.max()}")
        save("svhn_full", xtr, ytr, xte, yte)
    if (HF / "cifar100_train.parquet").exists():
        xtr, ytr = table_to_arrays(HF / "cifar100_train.parquet")
        xte, yte = table_to_arrays(HF / "cifar100_test.parquet")
        assert xtr.shape[0] == 50000 and xte.shape[0] == 10000, (xtr.shape, xte.shape)
        assert ytr.max() == 99, (ytr.min(), ytr.max())
        save("cifar100_full", xtr, ytr, xte, yte)
    if (HF / "fmnist_train.parquet").exists():
        xtr, ytr = table_to_arrays(HF / "fmnist_train.parquet", gray_to_rgb=True)
        xte, yte = table_to_arrays(HF / "fmnist_test.parquet", gray_to_rgb=True)
        assert xtr.shape[0] == 60000 and xte.shape[0] == 10000, (xtr.shape, xte.shape)
        assert ytr.max() == 9, (ytr.min(), ytr.max())
        save("fmnist_full", xtr, ytr, xte, yte)
    print("[done]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
