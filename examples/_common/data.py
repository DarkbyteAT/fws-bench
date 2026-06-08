"""CIFAR-10 loader with ``~/.cache/cifar10`` caching.

Downloads the canonical Krizhevsky distribution from ``cs.toronto.edu``,
extracts to ``~/.cache/cifar10/cifar-10-batches-py``, and returns
per-channel-normalised float32 tensors using train-set statistics.
"""

from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np


CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_CACHE = Path.home() / ".cache" / "cifar10"
CIFAR_NAMES: tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def _download_cifar() -> Path:
    """Idempotent download + extract; returns the batches-py directory."""
    CIFAR_CACHE.mkdir(parents=True, exist_ok=True)
    tgz = CIFAR_CACHE / "cifar-10-python.tar.gz"
    extracted = CIFAR_CACHE / "cifar-10-batches-py"
    if not extracted.exists():
        if not tgz.exists():
            print(f"  downloading CIFAR-10 to {tgz} ...")
            urllib.request.urlretrieve(CIFAR_URL, tgz)
        print(f"  extracting CIFAR-10 to {extracted} ...")
        with tarfile.open(tgz) as tf:
            tf.extractall(CIFAR_CACHE)
    return extracted


def _load_pickle_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one CIFAR-10 batch pickle and return ``(images, labels)``.

    Safe pickle source: canonical CIFAR-10 distribution from
    ``cs.toronto.edu/~kriz/cifar-10-python.tar.gz`` (Krizhevsky's upstream).
    """
    with path.open("rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    data = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.asarray(d[b"labels"], dtype=np.int32)
    return data, labels


def load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(train_x, train_y, test_x, test_y)``, per-channel normalised.

    Normalisation statistics are computed from the train set only. Images
    are channel-first ``(N, 3, 32, 32)``, labels are ``int32`` in ``[0, 10)``.
    """
    root = _download_cifar()
    train_x_list: list[np.ndarray] = []
    train_y_list: list[np.ndarray] = []
    for i in range(1, 6):
        x, y = _load_pickle_batch(root / f"data_batch_{i}")
        train_x_list.append(x)
        train_y_list.append(y)
    train_x = np.concatenate(train_x_list)
    train_y = np.concatenate(train_y_list)
    test_x, test_y = _load_pickle_batch(root / "test_batch")
    mean = train_x.mean(axis=(0, 2, 3), keepdims=True)
    std = train_x.std(axis=(0, 2, 3), keepdims=True) + 1e-7
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    return train_x, train_y, test_x, test_y


def load_raw_test() -> np.ndarray:
    """Return the unnormalised test set (``N, 3, 32, 32``) for curated plots."""
    root = _download_cifar()
    return _load_pickle_batch(root / "test_batch")[0]
