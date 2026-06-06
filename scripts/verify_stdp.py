"""Verify STDP reproduces the canonical asymmetric learning window."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain.stdp import STDP

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)


def pairing_dw(delta_t: int, w0: float = 0.5, tau: float = 20.0,
               a_plus: float = 0.01, a_minus: float = 0.012) -> float:
    """One pre/post spike pair separated by delta_t = t_post - t_pre; return ΔW."""
    stdp = STDP(1, 1, tau_pre=tau, tau_post=tau, a_plus=a_plus, a_minus=a_minus,
                w_min=0.0, w_max=1.0, dt=1.0)
    W = np.array([[w0]])
    base = 50
    t_pre, t_post = base, base + delta_t
    T = base + abs(delta_t) + 50
    for ti in range(T):
        pre = np.array([ti == t_pre])
        post = np.array([ti == t_post])
        stdp.step(W, pre, post)
    return W[0, 0] - w0


def window():
    deltas = np.arange(-40, 41)
    dw = np.array([pairing_dw(int(d)) for d in deltas])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="gray", lw=0.7)
    ax.axvline(0, color="gray", lw=0.7, ls="--")
    pos, neg = dw >= 0, dw < 0
    ax.scatter(deltas[pos], dw[pos], s=14, color="#2f855a", label="potentiation")
    ax.scatter(deltas[neg], dw[neg], s=14, color="#c53030", label="depression")
    ax.set(xlabel="Δt = t_post − t_pre (ms)", ylabel="ΔW", title="STDP learning window")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stdp_window.png"), dpi=130)
    plt.close(fig)
    return deltas, dw


def checks():
    deltas, dw = window()
    tau, a_plus, a_minus = 20.0, 0.01, 0.012

    pre_before_post = dw[deltas == 1][0]
    post_before_pre = dw[deltas == -1][0]
    print(f"[stdp] dt=+1 -> dW={pre_before_post:+.5f} (LTP), "
          f"dt=-1 -> dW={post_before_pre:+.5f} (LTD)")

    assert pre_before_post > 0, "pre-before-post did not potentiate"
    assert post_before_pre < 0, "post-before-pre did not depress"
    assert dw[deltas == 0][0] == 0.0, "coincident spikes should give no change"

    # Exponential fall-off: closer pairs change the weight more.
    assert dw[deltas == 1][0] > dw[deltas == 20][0] > 0, "LTP not decaying with Δt"
    assert dw[deltas == -1][0] < dw[deltas == -20][0] < 0, "LTD not decaying with Δt"

    # Quantitative match to a_plus * exp(-Δt/tau).
    assert np.isclose(pre_before_post, a_plus * np.exp(-1 / tau), atol=1e-4), "LTP magnitude off"
    assert np.isclose(post_before_pre, -a_minus * np.exp(-1 / tau), atol=1e-4), "LTD magnitude off"
    print("\nAll STDP checks passed.")


if __name__ == "__main__":
    checks()
