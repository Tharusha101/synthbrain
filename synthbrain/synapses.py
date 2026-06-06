"""Synaptic connectivity and current injection.

A `Synapses` object connects a presynaptic group of `n_pre` neurons to a
postsynaptic group of `n_post` neurons through a dense weight matrix
`W` of shape (n_pre, n_post).

Each presynaptic spike deposits charge into a per-postsynaptic synaptic variable
`g` (think: a synaptic conductance / current), which then decays exponentially
with time constant `tau_syn`:

    g[t+1] = g[t] * exp(-dt / tau_syn) + sum_{j spiked} W[j, :]

`g` is exactly the current vector (shape (n_post,)) you feed into the
postsynaptic `LIFGroup.step`. Decay is applied *before* the new spikes are added
so a spike arriving on step t contributes its full weight on step t.

The weight matrix is public (`syn.W`) so a plasticity rule (see `stdp.py`) can
modify it in place during simulation.
"""

from __future__ import annotations

import numpy as np


class Synapses:
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        w: np.ndarray | None = None,
        tau_syn: float = 5.0,   # ms, synaptic current decay
        dt: float = 1.0,        # ms, must match the post group's dt
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.tau_syn = tau_syn
        self.dt = dt
        self.decay = float(np.exp(-dt / tau_syn))

        if w is None:
            w = np.zeros((n_pre, n_post), dtype=np.float64)
        else:
            w = np.asarray(w, dtype=np.float64)
            if w.shape != (n_pre, n_post):
                raise ValueError(f"w must have shape {(n_pre, n_post)}, got {w.shape}")
        self.W = w

        # State: synaptic current seen by each postsynaptic neuron.
        self.g = np.zeros(n_post, dtype=np.float64)

    # -- construction helpers ------------------------------------------------

    @classmethod
    def all_to_all(
        cls,
        n_pre: int,
        n_post: int,
        w_low: float = 0.0,
        w_high: float = 1.0,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> "Synapses":
        """Dense connectivity with weights drawn uniformly from [w_low, w_high)."""
        rng = rng or np.random.default_rng()
        w = rng.uniform(w_low, w_high, size=(n_pre, n_post))
        return cls(n_pre, n_post, w=w, **kwargs)

    @classmethod
    def random(
        cls,
        n_pre: int,
        n_post: int,
        p: float = 0.1,
        w_low: float = 0.0,
        w_high: float = 1.0,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> "Synapses":
        """Sparse connectivity: each pre->post connection exists with probability p."""
        rng = rng or np.random.default_rng()
        mask = rng.random((n_pre, n_post)) < p
        w = rng.uniform(w_low, w_high, size=(n_pre, n_post)) * mask
        return cls(n_pre, n_post, w=w, **kwargs)

    @classmethod
    def lateral_inhibition(
        cls,
        n: int,
        w_inh: float = 1.0,
        **kwargs,
    ) -> "Synapses":
        """All-to-all *inhibitory* coupling within a layer, no self-connections.

        Weights are negative (-w_inh) off the diagonal, zero on it, so a spike from
        one neuron suppresses every other neuron in the layer (winner-take-all).
        """
        w = -w_inh * (np.ones((n, n)) - np.eye(n))
        return cls(n, n, w=w, **kwargs)

    # -- dynamics ------------------------------------------------------------

    def reset_state(self):
        self.g[:] = 0.0

    def step(self, pre_spikes: np.ndarray) -> np.ndarray:
        """Decay the synaptic current, add this step's presynaptic contributions.

        pre_spikes: boolean array (shape (n_pre,)).
        Returns the current vector (shape (n_post,)) for the postsynaptic group.
        """
        self.g *= self.decay
        if pre_spikes.any():
            # Sum the weight rows of every presynaptic neuron that spiked.
            self.g += self.W[pre_spikes].sum(axis=0)
        return self.g

    # -- maintenance ---------------------------------------------------------

    def normalize(self, target: float = 78.0):
        """Rescale each postsynaptic neuron's incoming weights to a fixed L1 sum.

        Diehl & Cook (2015) keep total input weight per neuron constant to stop a
        few synapses from running away during STDP. No-op for columns summing to 0.
        """
        col_sums = self.W.sum(axis=0)
        nonzero = col_sums > 0
        self.W[:, nonzero] *= target / col_sums[nonzero]
