"""Fast, plot-free unit checks for the SNN building blocks.

Run with:  pytest -q   (from the project root)
These mirror the deeper diagnostics in scripts/verify_*.py but stay cheap.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain.lif import LIFGroup
from synthbrain.synapses import Synapses
from synthbrain.stdp import STDP
from synthbrain import encoding


def test_lif_spikes_above_rheobase_and_silent_below():
    g = LIFGroup(1, dt=0.5)
    _, sp_hi = g.run(np.full((400, 1), 20.0))
    assert sp_hi.sum() > 0
    g.reset_state()
    _, sp_lo = g.run(np.full((400, 1), 5.0))
    assert sp_lo.sum() == 0


def test_lif_adaptive_threshold_reduces_firing():
    """With theta_plus>0, the same drive should yield fewer spikes than without."""
    base = LIFGroup(1, dt=1.0)
    _, sp_base = base.run(np.full((500, 1), 20.0))
    adapt = LIFGroup(1, dt=1.0, theta_plus=2.0, tau_theta=100.0)
    _, sp_adapt = adapt.run(np.full((500, 1), 20.0))
    assert sp_adapt.sum() < sp_base.sum()


def test_synapse_deposits_and_decays():
    syn = Synapses(1, 1, w=np.array([[3.0]]), tau_syn=5.0, dt=1.0)
    g0 = syn.step(np.array([True]))[0]
    g1 = syn.step(np.array([False]))[0]
    assert np.isclose(g0, 3.0)
    assert np.isclose(g1 / g0, np.exp(-1 / 5.0))


def test_synapse_normalize_equalizes_columns():
    syn = Synapses.all_to_all(20, 5, rng=np.random.default_rng(0))
    syn.normalize(target=10.0)
    assert np.allclose(syn.W.sum(axis=0), 10.0)


def test_stdp_sign_and_magnitude():
    def dw(delta_t):
        stdp = STDP(1, 1, tau_pre=20, tau_post=20, a_plus=0.01, a_minus=0.012)
        W = np.array([[0.5]])
        for ti in range(120):
            stdp.step(W, np.array([ti == 50]), np.array([ti == 50 + delta_t]))
        return W[0, 0] - 0.5

    assert dw(1) > 0           # pre-before-post potentiates
    assert dw(-1) < 0          # post-before-pre depresses
    assert dw(0) == 0.0
    assert dw(1) > dw(20) > 0  # exponential fall-off


def test_poisson_rate_tracks_intensity():
    rng = np.random.default_rng(3)
    sp = encoding.poisson_encode(np.array([0.0, 1.0]), T=10000, max_rate=63.75, rng=rng)
    assert sp[:, 0].sum() == 0
    rate = sp[:, 1].mean() * 1000.0
    assert abs(rate - 63.75) < 4.0
