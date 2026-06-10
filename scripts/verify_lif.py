"""Verify the LIF neuron behaves correctly and produce diagnostic plots."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain.lif import LIFGroup

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
)
os.makedirs(OUT, exist_ok=True)


def membrane_trace():
    """One neuron, constant supra-threshold current -> regular spiking."""
    dt = 0.5
    T = int(200 / dt)
    g = LIFGroup(1, dt=dt)
    currents = np.full((T, 1), 20.0)
    v_trace, spikes = g.run(currents)
    time = np.arange(T) * dt

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(time, v_trace[:, 0], color="#2b6cb0", lw=1.3)
    for ti in np.where(spikes[:, 0])[0]:
        ax.axvline(ti * dt, color="#e53e3e", alpha=0.35, lw=1)
    ax.axhline(g.v_thresh, ls="--", color="gray", lw=0.8, label="threshold")
    ax.set(xlabel="time (ms)", ylabel="V (mV)", title="LIF membrane potential (I = 20)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lif_membrane.png"), dpi=130)
    plt.close(fig)
    n_spikes = int(spikes.sum())
    print(f"[membrane_trace] spikes in 200ms: {n_spikes}")
    return n_spikes


def fi_curve():
    """Firing rate vs input current — should be monotincreasing with a rheobase."""
    dt = 0.5
    T = int(1000 / dt)
    currents_levels = np.linspace(0, 40, 25)
    rates = []
    for current in currents_levels:
        g = LIFGroup(1, dt=dt)
        _, spikes = g.run(np.full((T, 1), current))
        rates.append(spikes.sum() / (T * dt) * 1000.0)  # Hz
    rates = np.array(rates)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(currents_levels, rates, "o-", color="#2f855a", ms=4)
    ax.set(xlabel="input current I", ylabel="firing rate (Hz)", title="F-I curve")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lif_fi_curve.png"), dpi=130)
    plt.close(fig)

    rheobase = currents_levels[np.argmax(rates > 0)]
    monotonic = np.all(np.diff(rates) >= -1e-9)
    print(
        f"[fi_curve] rheobase ~ {rheobase:.1f}, monotonic: {monotonic}, max rate {rates.max():.0f} Hz"
    )
    return monotonic, rates


def checks():
    assert membrane_trace() > 0, "neuron failed to spike under strong current"
    # Subthreshold: weak current must NOT spike.
    g = LIFGroup(1, dt=0.5)
    _, sp = g.run(np.full((int(200 / 0.5), 1), 5.0))
    assert sp.sum() == 0, "neuron spiked under subthreshold current"
    mono, rates = fi_curve()
    assert mono, "F-I curve not monotonic"
    assert (
        rates[-1] > rates[len(rates) // 2]
    ), "firing rate did not increase with current"
    print("\nAll LIF checks passed.")


if __name__ == "__main__":
    checks()
