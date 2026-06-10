"""Tiny end-to-end smoke run -- proves the repo works right after cloning.

Uses fast OFFLINE synthetic digit-like data (no MNIST download), trains a small
unsupervised SNN, prints the spike-count shape and the native readout accuracy,
and saves one receptive-field image. Runs in well under 2 minutes on CPU.

    python scripts/reproduce_tiny.py

For the real MNIST runs see scripts/scaleup_network.py (CPU) and
scripts/train_gpu.py (GPU, the winning recipe).
"""

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain import Network
from synthbrain.encoding import synthetic_digits

SEED = 0
T = 80  # timesteps per image (ms at dt=1)
EPOCHS = 2
N_EXC = 40
CLASSES = (0, 1, 2, 3)


def save_receptive_fields(net: Network, path: str, n_show: int = 16) -> None:
    """Save a small grid of learned input weights as a PNG."""
    rfs = net.receptive_fields((28, 28))
    n_show = min(n_show, net.n_exc)
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for k, ax in enumerate(axes.ravel()):
        if k < n_show:
            ax.imshow(rfs[k], cmap="hot")
            label = net.neuron_labels[k] if net.neuron_labels is not None else "?"
            ax.set_title(str(label), fontsize=8)
        ax.axis("off")
    fig.suptitle("tiny-run receptive fields (synthetic data)")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    train_imgs, train_labels = synthetic_digits(
        n_per_class=18, classes=CLASSES, rng=rng
    )
    test_imgs, test_labels = synthetic_digits(n_per_class=6, classes=CLASSES, rng=rng)
    print(
        f"data: {len(train_imgs)} train / {len(test_imgs)} test, "
        f"{len(CLASSES)} classes, image shape {train_imgs.shape[1:]}"
    )

    net = Network(n_input=train_imgs[0].size, n_exc=N_EXC, rng=rng)

    counts = net.present(train_imgs[0], T=T, train=False)
    print(f"spike counts shape: {counts.shape}  (one count per excitatory neuron)")

    print(f"training: {N_EXC} neurons, {EPOCHS} epochs, T={T} ...")
    net.train(train_imgs, epochs=EPOCHS, T=T)

    net.assign_labels(train_imgs, train_labels, T=T)
    acc = net.evaluate(test_imgs, test_labels, T=T)
    chance = 1.0 / len(CLASSES)
    print(f"native readout accuracy: {acc:.3f}  (chance = {chance:.3f})")

    os.makedirs("outputs", exist_ok=True)
    out_png = os.path.join("outputs", "reproduce_tiny_rf.png")
    save_receptive_fields(net, out_png)
    print(f"saved receptive fields -> {out_png}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
