"""TorchSNN forward-pass tests.

Skipped automatically when torch is not installed (e.g. the lightweight CI job),
so they never block the core test suite. Run locally with torch present to check
the GPU port's shapes and determinism on CPU.
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain.encoding import synthetic_digits
from synthbrain.torch_snn import TorchSNN


def _tiny(seed: int = 0):
    rng = np.random.default_rng(seed)
    imgs, labels = synthetic_digits(n_per_class=6, classes=(0, 1, 2), rng=rng)
    net = TorchSNN(n_input=imgs[0].size, n_exc=15, device="cpu", seed=seed)
    return net, imgs, labels


def test_counts_output_shape():
    net, imgs, _ = _tiny()
    counts = net.counts(imgs, T=30, batch_size=8)
    assert counts.shape == (len(imgs), net.n_exc)
    assert (counts >= 0).all()


def test_counts_from_spikes_shape():
    net, imgs, _ = _tiny()
    x = net._to_device(imgs[:4])
    spikes = net.encode(x, T=30)
    out = net.counts_from_spikes(spikes)
    assert tuple(out.shape) == (4, net.n_exc)


def test_predict_raises_before_assign_labels():
    net, imgs, _ = _tiny()
    with pytest.raises(RuntimeError):
        net.predict(imgs, T=30)


def test_forward_is_deterministic_with_seed():
    net1, imgs, _ = _tiny(seed=1)
    net2, _, _ = _tiny(seed=1)
    c1 = net1.counts(imgs, T=30, batch_size=8)
    c2 = net2.counts(imgs, T=30, batch_size=8)
    assert np.array_equal(c1, c2)
