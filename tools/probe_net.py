"""网络测速：为 CIFAR/SVHN 寻找可用镜像。用法: python tools/probe_net.py"""
import json
import time
import urllib.request

H = {"User-Agent": "Mozilla/5.0"}


def speed(u, tmax=5.0, headers=None):
    h = dict(H)
    if headers:
        h.update(headers)
    t = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=12)
        n = 0
        while time.time() - t < tmax:
            d = r.read(1 << 20)
            if not d:
                break
            n += len(d)
        el = max(time.time() - t, 0.01)
        return f"{n/1e6:8.2f} MB {n/1e6/el:7.2f} MB/s"
    except Exception as e:
        return f"FAIL {type(e).__name__}: {e}"


def main():
    try:
        j = json.load(urllib.request.urlopen("https://pypi.org/pypi/numpy/json", timeout=15))
        big = [f for f in j["urls"] if f["size"] > 10_000_000]
        if big:
            print("pypi     ", speed(big[0]["url"]))
    except Exception as e:
        print("pypi FAIL", e)
    urls = [
        "https://ossci-datasets.s3.amazonaws.com/cifar-10-python.tar.gz",
        "https://ossci-datasets.s3.us-west-2.amazonaws.com/cifar-10-python.tar.gz",
        "http://cs231n.stanford.edu/cifar-10-python.tar.gz",
        "https://data.deepai.org/cifar10.zip",
        "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "https://www.kaggle.com/api/v1/datasets/download",
        "https://raw.githubusercontent.com/pytorch/torchvision/main/README.md",
        "https://objects.githubusercontent.com/",
        "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
        "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
        "https://huggingface.co/",
    ]
    for u in urls:
        print(f"{speed(u)}  {u}")


if __name__ == "__main__":
    main()
