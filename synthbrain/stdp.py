"""Pair-based spike-timing-dependent plasticity (STDP).

Online, trace-based implementation of the classic pair STDP rule:

    * pre-before-post  -> potentiation (LTP)
    * post-before-pre  -> depression  (LTD)

Each presynaptic neuron carries a trace `x_pre`, each postsynaptic neuron a trace
`x_post`. Both decay exponentially and are incremented when their neuron spikes.
On every timestep, for the weight matrix W (shape (n_pre, n_post)):

    * a presynaptic spike depresses its row:    W[j, :] -= a_minus * x_post
    * a postsynaptic spike potentiates its col:  W[:, k] += a_plus  * x_pre

Updates use the traces *as they stand this step* (pre-increment), so a post spike
that follows a pre spike by Δt sees x_pre = exp(-Δt / tau_pre): the weight change
falls off exponentially with the spike-time difference, reproducing the
exponential STDP window. Weights are clipped to [w_min, w_max].
"""

from __future__ import annotations

import numpy as np


class STDP:
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_pre: float = 20.0,    # ms, presynaptic trace time constant
        tau_post: float = 20.0,   # ms, postsynaptic trace time constant
        a_plus: float = 0.01,     # LTP learning rate (pre-before-post)
        a_minus: float = 0.012,   # LTD learning rate (post-before-pre)
        w_min: float = 0.0,
        w_max: float = 1.0,
        dt: float = 1.0,
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_min = w_min
        self.w_max = w_max
        self.dt = dt
        self.decay_pre = float(np.exp(-dt / tau_pre))
        self.decay_post = float(np.exp(-dt / tau_post))

        self.x_pre = np.zeros(n_pre, dtype=np.float64)
        self.x_post = np.zeros(n_post, dtype=np.float64)

    def reset_state(self):
        self.x_pre[:] = 0.0
        self.x_post[:] = 0.0

    def step(self, W: np.ndarray, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> np.ndarray:
        """Apply one STDP update to W in place and return it.

        pre_spikes:  boolean (n_pre,);  post_spikes: boolean (n_post,).
        """
        # 1. Traces relax toward zero.
        self.x_pre *= self.decay_pre
        self.x_post *= self.decay_post

        # 2. Weight changes from this step's spikes, using the decayed traces
        #    (i.e. the partner's most recent activity).
        if pre_spikes.any():                       # LTD: post fired before this pre
            W[pre_spikes] -= self.a_minus * self.x_post[None, :]
        if post_spikes.any():                      # LTP: pre fired before this post
            W[:, post_spikes] += self.a_plus * self.x_pre[:, None]

        # 3. Register this step's spikes in the traces (for future pairings).
        self.x_pre[pre_spikes] += 1.0
        self.x_post[post_spikes] += 1.0

        np.clip(W, self.w_min, self.w_max, out=W)
        return W
