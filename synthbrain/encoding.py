"""Rate-based (Poisson) encoding of images into spike trains.

A grayscale image is flattened and each pixel becomes an independent Poisson
spike generator whose firing rate is proportional to the pixel intensity:

    rate_i (Hz) = intensity_i * max_rate          intensity in [0, 1]

Over a window of `T` timesteps of width `dt` ms, pixel i spikes on each step with
probability  p_i = rate_i * dt / 1000 . This is the standard input layer used in
the Diehl & Cook (2015) MNIST-SNN (max_rate ~ 63.75 Hz, T ~ 350 ms).

`load_mnist` tries a few common backends and caches a .npz locally. Everything
downstream only needs plain NumPy arrays, so the rest of the project works with
any image data — including the synthetic fallback used by the demo when MNIST
isn't available offline.
"""

from __future__ import annotations

import os

import numpy as np


def poisson_encode(
    image: np.ndarray,
    T: int,
    dt: float = 1.0,
    max_rate: float = 63.75,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Encode a single image (any shape) into a (T, n_pixels) boolean spike train.

    `image` is normalized to [0, 1] by its own max (a zero image stays silent).
    """
    rng = rng or np.random.default_rng()
    flat = np.asarray(image, dtype=np.float64).ravel()
    peak = flat.max()
    if peak > 0:
        flat = flat / peak
    rates = flat * max_rate  # Hz, shape (n_pixels,)
    p = rates * dt / 1000.0  # spike probability per step
    draws = rng.random((T, flat.size))
    return draws < p[None, :]


def encode_batch(
    images: np.ndarray,
    T: int,
    dt: float = 1.0,
    max_rate: float = 63.75,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Encode a batch of images (N, ...) into spikes of shape (N, T, n_pixels)."""
    rng = rng or np.random.default_rng()
    return np.stack(
        [poisson_encode(img, T, dt, max_rate, rng) for img in images], axis=0
    )


def synthetic_digits(
    n_per_class: int = 20,
    classes=(0, 1),
    size: int = 28,
    noise: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tiny offline stand-in for MNIST: distinguishable stroke patterns + noise.

    Returns (images (N, size, size) in [0,1], labels (N,)). Used so the pipeline
    and demo run without a network connection. Patterns are simple bars/crosses,
    one prototype per requested class.
    """
    rng = rng or np.random.default_rng()

    def prototype(c: int) -> np.ndarray:
        img = np.zeros((size, size))
        mid = size // 2
        band = max(1, size // 8)
        if c == 0:  # vertical bar
            img[:, mid - band : mid + band] = 1.0
        elif c == 1:  # horizontal bar
            img[mid - band : mid + band, :] = 1.0
        elif c == 2:  # cross
            img[:, mid - band : mid + band] = 1.0
            img[mid - band : mid + band, :] = 1.0
        else:  # diagonal
            for k in range(size):
                lo, hi = max(0, k - band), min(size, k + band)
                img[k, lo:hi] = 1.0
        return img

    imgs, labels = [], []
    for c in classes:
        base = prototype(c)
        for _ in range(n_per_class):
            noisy = np.clip(base + rng.normal(0, noise, base.shape), 0, 1)
            imgs.append(noisy)
            labels.append(c)
    imgs = np.array(imgs)
    labels = np.array(labels)
    perm = rng.permutation(len(labels))
    return imgs[perm], labels[perm]


def load_mnist(cache_dir: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST as (images (N, 28, 28) in [0,1], labels (N,)).

    Tries, in order: a local .npz cache, torchvision, then sklearn/openml.
    Raises RuntimeError if none succeed (caller can fall back to synthetic data).
    """
    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".synthbrain")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "mnist.npz")

    if os.path.exists(cache):
        d = np.load(cache)
        return d["images"], d["labels"]

    images = labels = None

    try:  # torchvision (raw IDX, fast)
        from torchvision.datasets import MNIST  # type: ignore

        ds = MNIST(cache_dir, train=True, download=True)
        images = ds.data.numpy().astype(np.float64) / 255.0
        labels = ds.targets.numpy().astype(np.int64)
    except Exception:
        pass

    if images is None:
        try:  # sklearn / openml
            from sklearn.datasets import fetch_openml  # type: ignore

            mnist = fetch_openml("mnist_784", version=1, as_frame=False)
            images = mnist.data.reshape(-1, 28, 28).astype(np.float64) / 255.0
            labels = mnist.target.astype(np.int64)
        except Exception:
            pass

    if images is None:
        raise RuntimeError(
            "Could not load MNIST (no cache, torchvision, or sklearn/openml). "
            "Use encoding.synthetic_digits() for an offline run."
        )

    np.savez_compressed(cache, images=images, labels=labels)
    return images, labels
