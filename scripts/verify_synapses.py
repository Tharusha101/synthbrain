"""Verify synaptic current injection, decay, and weight normalization."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain.synapses import Synapses

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
)
os.makedirs(OUT, exist_ok=True)


def epsp_trace():
    """A single presynaptic spike should deposit its weight, then decay by exp(-dt/tau)."""
    dt, tau = 1.0, 5.0
    w = 4.0
    syn = Synapses(1, 1, w=np.array([[w]]), tau_syn=tau, dt=dt)

    T = 60
    spike_at = 5
    g_trace = np.empty(T)
    for ti in range(T):
        pre = np.array([ti == spike_at])
        g_trace[ti] = syn.step(pre)[0]

    # On the spike step the full weight is present; one step later it has decayed once.
    peak = g_trace[spike_at]
    after = g_trace[spike_at + 1]
    ratio = after / peak
    expected = np.exp(-dt / tau)
    print(
        f"[epsp] peak={peak:.3f} (want {w}), decay ratio={ratio:.4f} (want {expected:.4f})"
    )

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(T) * dt, g_trace, color="#6b46c1", lw=1.5)
    ax.axvline(spike_at * dt, color="#e53e3e", alpha=0.4, lw=1, label="pre spike")
    ax.set(
        xlabel="time (ms)",
        ylabel="synaptic current g",
        title="EPSP: single-spike synaptic current",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "syn_epsp.png"), dpi=130)
    plt.close(fig)

    assert abs(peak - w) < 1e-9, "spike did not deposit the full weight"
    assert abs(ratio - expected) < 1e-6, "synaptic decay does not match exp(-dt/tau)"
    return True


def summation():
    """Two simultaneous presynaptic spikes should sum their weights into one post neuron."""
    syn = Synapses(3, 1, w=np.array([[1.0], [2.0], [4.0]]), tau_syn=5.0, dt=1.0)
    g = syn.step(np.array([True, False, True]))[0]  # neurons 0 and 2 fire
    print(f"[summation] g={g:.3f} (want 5.0)")
    assert abs(g - 5.0) < 1e-9, "weights from co-active inputs did not sum"
    return True


def normalization():
    """After normalize(), every postsynaptic neuron's incoming weights sum to target."""
    rng = np.random.default_rng(0)
    syn = Synapses.all_to_all(50, 8, w_low=0.0, w_high=1.0, rng=rng)
    syn.normalize(target=78.0)
    sums = syn.W.sum(axis=0)
    print(
        f"[normalize] column sums in [{sums.min():.3f}, {sums.max():.3f}] (want 78.0)"
    )
    assert np.allclose(sums, 78.0), "normalize() did not equalize incoming weight sums"
    return True


def checks():
    assert epsp_trace()
    assert summation()
    assert normalization()
    print("\nAll synapse checks passed.")


if __name__ == "__main__":
    checks()
