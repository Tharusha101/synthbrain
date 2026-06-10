"""Leaky Integrate-and-Fire (LIF) neuron model.

The membrane potential V evolves according to the standard LIF dynamics:

    tau_m * dV/dt = -(V - V_rest) + R * I(t)

Discretized with timestep dt (forward Euler):

    V[t+1] = V[t] + (dt / tau_m) * (-(V[t] - V_rest) + R * I[t])

When V crosses the (effective) threshold, the neuron emits a spike and V is reset
to V_reset. After a spike the neuron is held at V_reset for `t_refrac` ms
(refractory period).

Optional adaptive threshold (homeostasis, Diehl & Cook 2015): each spike raises a
per-neuron offset `theta` by `theta_plus`, and `theta` decays back with time
constant `tau_theta`. The effective firing threshold is `v_thresh + theta`. This is
disabled by default (`theta_plus=0.0`), in which case the dynamics are identical to
the plain LIF model.

All quantities are vectorized: a single LIFGroup instance represents `n`
independent neurons, so it works equally as one neuron or a full layer.
"""

from __future__ import annotations

import numpy as np


class LIFGroup:
    def __init__(
        self,
        n: int,
        dt: float = 1.0,  # ms, simulation timestep
        tau_m: float = 20.0,  # ms, membrane time constant
        v_rest: float = -65.0,  # mV, resting potential
        v_reset: float = -65.0,  # mV, reset potential after spike
        v_thresh: float = -52.0,  # mV, firing threshold (baseline)
        r_m: float = 1.0,  # MOhm, membrane resistance (scales input current)
        t_refrac: float = 2.0,  # ms, absolute refractory period
        theta_plus: float = 0.0,  # mV, threshold bump per spike (0 -> adaptation off)
        tau_theta: float = 1e7,  # ms, threshold-offset decay time constant
        rng: np.random.Generator | None = None,
    ):
        self.n = n
        self.dt = dt
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.r_m = r_m
        self.t_refrac = t_refrac
        self.theta_plus = theta_plus
        self.tau_theta = tau_theta
        self.theta_decay = float(np.exp(-dt / tau_theta))
        self.rng = rng or np.random.default_rng()

        # State
        self.v = np.full(n, v_rest, dtype=np.float64)
        self.refrac_until = np.zeros(
            n, dtype=np.float64
        )  # time (ms) until neuron can fire again
        self.theta = np.zeros(n, dtype=np.float64)  # adaptive threshold offset (mV)
        self.t = 0.0

    def reset_state(self):
        self.v[:] = self.v_rest
        self.refrac_until[:] = 0.0
        self.theta[:] = 0.0
        self.t = 0.0

    def step(self, i_in: np.ndarray) -> np.ndarray:
        """Advance one timestep given input current i_in (shape (n,)).

        Returns a boolean spike vector (shape (n,)).
        """
        self.t += self.dt

        # Adaptive threshold relaxes toward baseline every step.
        self.theta *= self.theta_decay

        # Neurons in refractory period are clamped and do not integrate.
        active = self.t >= self.refrac_until

        dv = (self.dt / self.tau_m) * (-(self.v - self.v_rest) + self.r_m * i_in)
        self.v = np.where(active, self.v + dv, self.v_reset)

        thresh = self.v_thresh + self.theta
        spikes = (self.v >= thresh) & active
        # Reset spiking neurons, set refractory window, and bump their threshold.
        self.v = np.where(spikes, self.v_reset, self.v)
        self.refrac_until = np.where(spikes, self.t + self.t_refrac, self.refrac_until)
        self.theta += spikes * self.theta_plus

        return spikes

    def run(self, currents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run a full simulation.

        currents: array of shape (T, n) — input current per timestep per neuron.
        Returns (v_trace, spike_trace), each of shape (T, n).
        """
        T = currents.shape[0]
        v_trace = np.empty((T, self.n), dtype=np.float64)
        spike_trace = np.empty((T, self.n), dtype=bool)
        for ti in range(T):
            sp = self.step(currents[ti])
            v_trace[ti] = self.v
            spike_trace[ti] = sp
        return v_trace, spike_trace
