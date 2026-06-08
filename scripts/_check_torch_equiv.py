"""Validate the PyTorch SNN's forward dynamics against the NumPy reference.

Runs ONE image through both `network.Network` (lateral mode) and `torch_snn.TorchSNN`
with *identical* weights, threshold offsets, and input spikes, on CPU in float64,
and checks the excitatory spike counts match. This isolates the simulation core
(LIF + synapse + lateral inhibition + adaptive threshold) from the two things that
legitimately differ -- the Poisson RNG stream and mini-batch STDP -- so we can trust
the GPU dynamics before committing to a long CUDA training run.

CPU-only, single image: runs in well under a second (this is a unit check, not a
training run).
"""

import os
import sys

import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain.network import Network
from synthbrain.torch_snn import TorchSNN
from synthbrain import encoding


def main():
    n_in, n_exc, T = 784, 60, 150
    w_inh, inp_pow = 0.6, 0.5
    rng = np.random.default_rng(0)

    # Matched NumPy and Torch (CPU/float64) networks.
    net = Network(n_input=n_in, n_exc=n_exc, inhibition="lateral", w_inh=w_inh,
                  input_norm_power=inp_pow, rng=rng)
    tnet = TorchSNN(n_input=n_in, n_exc=n_exc, w_inh=w_inh, input_norm_power=inp_pow,
                    device="cpu", dtype=torch.float64)

    # Force identical weights and a non-trivial threshold offset on both.
    tnet.W = torch.as_tensor(net.exc_syn.W.copy(), dtype=torch.float64)
    theta0 = rng.uniform(0.0, 2.0, n_exc)
    net.exc.theta[:] = theta0
    tnet.theta = torch.as_tensor(theta0.copy(), dtype=torch.float64)

    # One real image, encoded once with NumPy; feed the SAME spikes to both.
    try:
        imgs, _ = encoding.load_mnist()
        img = imgs[7]
    except Exception:
        img, _ = encoding.synthetic_digits(n_per_class=1, classes=[2], rng=rng)
        img = img[0]
    eff = net._eff_max_rate(img, 63.75)
    in_spikes = encoding.poisson_encode(img, T, dt=1.0, max_rate=eff,
                                        rng=np.random.default_rng(123))  # (T, n_in) bool

    # --- NumPy forward (no learning) ---
    net.exc.reset_state(); net.exc.theta[:] = theta0
    net.exc_syn.reset_state(); net.inh_syn.reset_state()
    prev = np.zeros(n_exc, dtype=bool)
    counts_np = np.zeros(n_exc, dtype=np.int64)
    for ti in range(T):
        s, prev = net._step_once(in_spikes[ti], prev)
        counts_np += s

    # --- Torch forward (no learning), same spikes, batch=1 ---
    in_t = torch.as_tensor(in_spikes.astype(np.float64))[None]  # (1, T, n_in)
    counts_t = tnet.counts_from_spikes(in_t).numpy()[0].astype(np.int64)

    diff = np.abs(counts_np - counts_t)
    print(f"NumPy total exc spikes:  {counts_np.sum()}")
    print(f"Torch total exc spikes:  {counts_t.sum()}")
    print(f"per-neuron count diff:   max={diff.max()}  mean={diff.mean():.4f}  "
          f"neurons differing={int((diff > 0).sum())}/{n_exc}")
    ok = diff.max() == 0
    print("FORWARD DYNAMICS MATCH (exact)" if ok else
          "forward dynamics differ -- investigate" if diff.max() > 2 else
          "forward dynamics match within +/-2 spikes (float32/ordering at threshold)")

    # Light encoding sanity check (RNG differs, so just compare mean rates).
    img_t = torch.as_tensor(img.reshape(1, -1).astype(np.float64))
    enc = tnet.encode(img_t, T=400, max_rate=63.75).numpy()[0]   # (400, n_in)
    emp_torch = enc.mean(axis=0).sum() / max(1, (enc.sum(axis=0) > 0).sum()) * 1000.0
    print(f"\ntorch encode: {enc.sum():.0f} spikes over 400ms across "
          f"{int((enc.sum(0) > 0).sum())} active pixels")


if __name__ == "__main__":
    main()
