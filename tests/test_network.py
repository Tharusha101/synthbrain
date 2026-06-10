"""Network-level tests: readout API, persistence, and error handling.

Fast and plot-free; NumPy only (no torch / no MNIST download), so they run in CI.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain import Network
from synthbrain.encoding import synthetic_digits

N_CLASSES = 3


def _tiny_net_and_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    imgs, labels = synthetic_digits(n_per_class=6, classes=(0, 1, 2), rng=rng)
    net = Network(n_input=imgs[0].size, n_exc=15, rng=rng)
    return net, imgs, labels


def test_present_output_shape_and_dtype():
    net, imgs, _ = _tiny_net_and_data()
    counts = net.present(imgs[0], T=40, train=False)
    assert counts.shape == (net.n_exc,)
    assert np.issubdtype(counts.dtype, np.integer)
    assert (counts >= 0).all()


def test_classify_raises_before_assign_labels():
    net, imgs, _ = _tiny_net_and_data()
    with pytest.raises(RuntimeError):
        net.classify(imgs[0], T=40)


def test_assign_labels_shape_and_range():
    net, imgs, labels = _tiny_net_and_data()
    net.train(imgs, epochs=1, T=40)
    neuron_labels = net.assign_labels(imgs, labels, T=40)
    assert neuron_labels.shape == (net.n_exc,)
    assert neuron_labels.min() >= 0
    assert neuron_labels.max() < N_CLASSES


def test_classify_returns_valid_class_after_labels():
    net, imgs, labels = _tiny_net_and_data()
    net.train(imgs, epochs=1, T=40)
    net.assign_labels(imgs, labels, T=40)
    pred = net.classify(imgs[0], T=40)
    assert isinstance(pred, int)
    assert 0 <= pred < N_CLASSES


def test_save_load_roundtrip(tmp_path):
    net, imgs, labels = _tiny_net_and_data()
    net.train(imgs, epochs=1, T=40)
    net.assign_labels(imgs, labels, T=40)
    path = str(tmp_path / "net.npz")
    net.save(path)

    loaded = Network.load(path)
    assert loaded.n_input == net.n_input
    assert loaded.n_exc == net.n_exc
    assert np.allclose(loaded.exc_syn.W, net.exc_syn.W)
    assert np.allclose(loaded.exc.theta, net.exc.theta)
    assert np.array_equal(loaded.neuron_labels, net.neuron_labels)
