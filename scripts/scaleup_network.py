"""Days 11-12 scale-up: bigger unsupervised SNN over all 10 MNIST digits.

Same architecture as demo_network.py, scaled up to get cleaner digit templates:
  * all 10 classes (chance = 0.10)
  * 400 excitatory neurons (vs 100 in the quick demo)
  * more images per class, more epochs, longer presentation T

Produces:
  * outputs/scaleup_receptive_fields.png  — learned input weights (sample of neurons)
  * outputs/scaleup_raster.png            — excitatory spike raster for one test image
  * outputs/scaleup_net.npz               — trained weights/thresholds/labels (reusable)
and prints unsupervised clustering accuracy plus an honest per-class breakdown
(neuron coverage + per-class recall) so over/under-represented digits are visible.
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

# -- scale-up knobs (tweak here) --------------------------------------------
CLASSES = list(range(10))   # all digits
N_PER_CLASS = 150           # images per class (train+test pool)
N_EXC = 400                 # excitatory neurons
EPOCHS = 4
T = 350                     # ms per image (the Network default; > the 150 demo)
# Soften per-image input gain (1.0 = full normalization). <1 stops thin digits
# like '1' from over-firing and monopolizing the excitatory layer.
INPUT_NORM_POWER = 0.5
# Inhibition: "lateral" = direct exc->exc approximation (default; best accuracy
# here), "two_layer" = explicit inhibitory population (more biologically faithful,
# true Diehl & Cook). W_INH is the inh->exc strength: ~0.6 for lateral, ~0.35 for
# two_layer (the value that matches lateral's competition; see below).
# FINDING (two_layer experiment): the explicit inhibitory layer does NOT beat the
# lateral approximation at this data scale. Competition-matched (w_inh=0.35,
# top-10 share ~0.33) it scored 60.3% vs 66.4% for lateral, with no cleaner
# receptive fields. And it must be tuned for competition COMPARABLE to lateral,
# not the sharpest winner-take-all: w_inh=3.0 (top-10=0.95, only ~10/400 neurons
# ever win) starved STDP and cratered to 29%. See scripts/_check_inhibition.py.
INHIBITION = "lateral"
W_INH = 0.6


def get_data(rng):
    """Return (train_imgs, train_lbls, test_imgs, test_lbls, image_shape, source)."""
    try:
        imgs, lbls = encoding.load_mnist()
        sel = np.concatenate([np.where(lbls == c)[0][:N_PER_CLASS] for c in CLASSES])
        rng.shuffle(sel)
        imgs, lbls = imgs[sel], lbls[sel]
        source = "MNIST"
    except Exception as e:
        print(f"[data] MNIST unavailable ({e.__class__.__name__}); using synthetic digits.")
        imgs, lbls = encoding.synthetic_digits(n_per_class=N_PER_CLASS, classes=CLASSES, rng=rng)
        source = "synthetic"

    n_train = int(0.75 * len(imgs))
    return (imgs[:n_train], lbls[:n_train], imgs[n_train:], lbls[n_train:],
            imgs.shape[1:], source)


def plot_receptive_fields(net, shape, n_show=64):
    rf = net.receptive_fields(shape)
    n_show = min(n_show, net.n_exc)
    cols = int(np.ceil(np.sqrt(n_show)))
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.1))
    for i, ax in enumerate(np.array(axes).ravel()):
        if i < n_show:
            ax.imshow(rf[i], cmap="hot")
            if net.neuron_labels is not None:
                ax.set_title(str(net.neuron_labels[i]), fontsize=7, pad=1)
        ax.axis("off")
    fig.suptitle(f"Learned receptive fields (sample of {n_show}/{net.n_exc})", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scaleup_receptive_fields.png"), dpi=130)
    plt.close(fig)


def plot_raster(net, image, T):
    trace = net.run_record(image, T=T)
    ts, ns = np.where(trace)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.scatter(ts, ns, s=5, color="#1a202c")
    ax.set(xlabel="time (ms)", ylabel="excitatory neuron",
           title="Excitatory spike raster (one test image)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scaleup_raster.png"), dpi=130)
    plt.close(fig)


def per_class_report(net, te_x, te_y, n_classes):
    """Print neuron coverage per digit and per-class recall (honest breakdown)."""
    labels = np.asarray(te_y)
    # neuron coverage: how many excitatory neurons were assigned to each digit
    coverage = np.bincount(net.neuron_labels, minlength=n_classes)
    # per-class recall
    preds = np.array([net.classify(img, T=T) for img in te_x])
    print("[report] per-digit neuron coverage / test recall:")
    for c in range(n_classes):
        mask = labels == c
        recall = (preds[mask] == c).mean() if mask.any() else float("nan")
        print(f"  digit {c}: {coverage[c]:3d} neurons   recall={recall:.2f}  (n={int(mask.sum())})")
    return preds


def main():
    rng = np.random.default_rng(0)
    tr_x, tr_y, te_x, te_y, shape, source = get_data(rng)
    n_input = int(np.prod(shape))
    n_classes = len(np.unique(np.concatenate([tr_y, te_y])))

    print(f"[scaleup] source={source}  train={len(tr_x)}  test={len(te_x)}  "
          f"input={n_input}  classes={n_classes}  n_exc={N_EXC}  epochs={EPOCHS}  T={T}  "
          f"input_norm_power={INPUT_NORM_POWER}  inhibition={INHIBITION}  w_inh={W_INH}")

    net = Network(n_input=n_input, n_exc=N_EXC, input_norm_power=INPUT_NORM_POWER,
                  inhibition=INHIBITION, w_inh=W_INH, rng=rng)

    t0 = time.time()
    net.train(tr_x, epochs=EPOCHS, T=T, progress=True)
    print(f"[scaleup] trained in {time.time() - t0:.1f}s")

    net.assign_labels(tr_x, tr_y, T=T, n_classes=n_classes)
    acc = net.evaluate(te_x, te_y, T=T)
    chance = 1.0 / n_classes
    print(f"[scaleup] test accuracy = {acc:.3f}  (chance = {chance:.3f})")

    per_class_report(net, te_x, te_y, n_classes)

    plot_receptive_fields(net, shape)
    plot_raster(net, te_x[0], T)
    net.save(os.path.join(OUT, "scaleup_net.npz"))
    print(f"[scaleup] wrote receptive fields + raster + scaleup_net.npz to {OUT}")
    return acc, chance


if __name__ == "__main__":
    acc, chance = main()
    assert acc > chance, "clustering accuracy did not beat chance"
    print("\nScale-up run complete.")
