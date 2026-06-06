"""End-to-end demo: train the unsupervised SNN, read out digit identity, visualize.

Uses MNIST if it can be loaded offline, otherwise a small synthetic stand-in so
the full pipeline still runs without a network connection. Produces:
  * outputs/net_receptive_fields.png  — learned input weights per excitatory neuron
  * outputs/net_raster.png            — excitatory spike raster for one test image
and prints unsupervised clustering accuracy (chance = 1 / n_classes).
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain.network import Network
from synthbrain import encoding

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)


def get_data(rng):
    """Return (train_imgs, train_lbls, test_imgs, test_lbls, image_shape, source)."""
    classes = [0, 1, 2, 3]
    try:
        imgs, lbls = encoding.load_mnist()
        # Small balanced subset for a quick demo.
        sel = np.concatenate([np.where(lbls == c)[0][:60] for c in classes])
        rng.shuffle(sel)
        imgs, lbls = imgs[sel], lbls[sel]
        source = "MNIST"
    except Exception as e:
        print(f"[data] MNIST unavailable ({e.__class__.__name__}); using synthetic digits.")
        imgs, lbls = encoding.synthetic_digits(n_per_class=40, classes=classes, rng=rng)
        source = "synthetic"

    n_train = int(0.75 * len(imgs))
    return (imgs[:n_train], lbls[:n_train], imgs[n_train:], lbls[n_train:],
            imgs.shape[1:], source)


def plot_receptive_fields(net, shape, n_show=16):
    rf = net.receptive_fields(shape)
    n_show = min(n_show, net.n_exc)
    cols = int(np.ceil(np.sqrt(n_show)))
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    for i, ax in enumerate(np.array(axes).ravel()):
        if i < n_show:
            ax.imshow(rf[i], cmap="hot")
            if net.neuron_labels is not None:
                ax.set_title(str(net.neuron_labels[i]), fontsize=8, pad=1)
        ax.axis("off")
    fig.suptitle("Learned receptive fields (input weights)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "net_receptive_fields.png"), dpi=130)
    plt.close(fig)


def plot_raster(net, image, T):
    trace = net.run_record(image, T=T)
    ts, ns = np.where(trace)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.scatter(ts, ns, s=6, color="#1a202c")
    ax.set(xlabel="time (ms)", ylabel="excitatory neuron",
           title="Excitatory spike raster (one test image)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "net_raster.png"), dpi=130)
    plt.close(fig)


def main():
    rng = np.random.default_rng(0)
    tr_x, tr_y, te_x, te_y, shape, source = get_data(rng)
    n_input = int(np.prod(shape))
    n_classes = len(np.unique(np.concatenate([tr_y, te_y])))
    T = 150  # ms per image — short for a quick demo

    print(f"[demo] source={source}  train={len(tr_x)}  test={len(te_x)}  "
          f"input={n_input}  classes={n_classes}")

    # Defaults (r_m=6, norm_target=90, w_inh=0.6) put the excitatory cells in a
    # healthy firing regime that learns from a cold start; see scripts/_sweep.py
    # history for how these were chosen.
    net = Network(n_input=n_input, n_exc=100, rng=rng)

    t0 = time.time()
    net.train(tr_x, epochs=5, T=T, progress=True)
    print(f"[demo] trained in {time.time() - t0:.1f}s")

    net.assign_labels(tr_x, tr_y, T=T, n_classes=n_classes)
    acc = net.evaluate(te_x, te_y, T=T)
    chance = 1.0 / n_classes
    print(f"[demo] test accuracy = {acc:.3f}  (chance = {chance:.3f})")

    plot_receptive_fields(net, shape)
    plot_raster(net, te_x[0], T)
    print(f"[demo] wrote receptive fields + raster to {OUT}")
    return acc, chance


if __name__ == "__main__":
    acc, chance = main()
    # Sanity: unsupervised readout should beat chance on this easy subset.
    assert acc > chance, "clustering accuracy did not beat chance"
    print("\nNetwork demo passed.")
