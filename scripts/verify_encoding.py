"""Verify Poisson encoding: rate scales with intensity, raster looks right."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain.encoding import poisson_encode

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)


def rate_scaling():
    """Empirical firing rate should track intensity * max_rate."""
    rng = np.random.default_rng(1)
    dt, T, max_rate = 1.0, 20000, 63.75
    intensities = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    spikes = poisson_encode(intensities, T, dt=dt, max_rate=max_rate, rng=rng)
    emp_rate = spikes.mean(axis=0) / dt * 1000.0          # Hz
    # Image is normalized by its own max (1.0 here), so expected = intensity * max_rate.
    expected = intensities * max_rate
    print("[rate] intensity -> empirical Hz (expected):")
    for i, e, x in zip(intensities, emp_rate, expected):
        print(f"   {i:.2f} -> {e:5.1f} ({x:5.1f})")
    assert spikes[:, 0].sum() == 0, "zero-intensity pixel produced spikes"
    assert np.all(np.diff(emp_rate) > 0), "rate not monotonic in intensity"
    assert np.allclose(emp_rate, expected, atol=3.0), "empirical rate off expected by >3 Hz"
    return emp_rate


def raster():
    """Encode a horizontal intensity gradient and show its spike raster."""
    rng = np.random.default_rng(2)
    n, T = 40, 350
    gradient = np.linspace(0, 1, n)
    spikes = poisson_encode(gradient, T, dt=1.0, max_rate=63.75, rng=rng)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ys, xs = np.where(spikes)          # (time, neuron)
    ax.scatter(ys, xs, s=2, color="#1a202c")
    ax.set(xlabel="time (ms)", ylabel="neuron (low->high intensity)",
           title="Poisson encoding of an intensity gradient")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "encoding_raster.png"), dpi=130)
    plt.close(fig)
    # Higher-intensity neurons (bottom of raster -> larger index) fire more.
    counts = spikes.sum(axis=0)
    assert counts[-1] > counts[0], "high-intensity neuron did not out-fire low-intensity one"
    print(f"[raster] low-intensity spikes={counts[0]}, high-intensity spikes={counts[-1]}")
    return True


def checks():
    rate_scaling()
    raster()
    print("\nAll encoding checks passed.")


if __name__ == "__main__":
    checks()
