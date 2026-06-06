"""Scale-up: train the unsupervised SNN on all 10 MNIST digits and report honestly.

Trains a larger excitatory layer with STDP (no backprop), assigns each neuron the
digit it responds to most (labels used for read-out only), then reports:
  * overall + per-class test accuracy (chance = 0.10),
  * how many neurons specialized to each digit,
  * a confusion matrix.
Plots learned receptive fields grouped by digit, and caches the trained net.

Outputs (in outputs/):
  mnist_receptive_fields.png   mnist_confusion.png   mnist_net.npz

Run:  KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_mnist.py [--n-exc 300 ...]
"""

import argparse
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
N_CLASSES = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-exc", type=int, default=300)
    p.add_argument("--train-per-class", type=int, default=120)
    p.add_argument("--test-per-class", type=int, default=30)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="mnist")
    return p.parse_args()


def balanced_subset(imgs, lbls, per_class, rng, exclude=None):
    """Take `per_class` images of each digit; `exclude` masks already-used indices."""
    used = set() if exclude is None else set(exclude)
    picks = []
    for c in range(N_CLASSES):
        cand = [i for i in np.where(lbls == c)[0] if i not in used]
        rng.shuffle(cand)
        picks.extend(cand[:per_class])
    rng.shuffle(picks)
    return np.array(picks)


def confusion_and_acc(net, te_x, te_y, T):
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for img, lab in zip(te_x, te_y):
        cm[lab, net.classify(img, T=T)] += 1
    per_class = np.divide(np.diag(cm), cm.sum(1),
                          out=np.zeros(N_CLASSES), where=cm.sum(1) > 0)
    acc = np.trace(cm) / cm.sum()
    return cm, acc, per_class


def plot_confusion(cm, acc, path):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xlabel="predicted", ylabel="true",
           title=f"Confusion matrix (acc={acc:.3f}, chance=0.100)",
           xticks=range(N_CLASSES), yticks=range(N_CLASSES))
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_receptive_fields(net, path, cols=8):
    """One row per digit, the most selective neurons for that digit across columns."""
    rf = net.receptive_fields((28, 28))
    labels, resp = net.neuron_labels, net.neuron_response
    fig, axes = plt.subplots(N_CLASSES, cols, figsize=(cols * 1.15, N_CLASSES * 1.15))
    for c in range(N_CLASSES):
        members = np.where(labels == c)[0]
        if members.size:
            members = members[np.argsort(-resp[members, c])][:cols]
        for j in range(cols):
            ax = axes[c, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j < len(members):
                ax.imshow(rf[members[j]], cmap="hot")
            else:
                ax.imshow(np.zeros((28, 28)), cmap="hot", vmin=0, vmax=1)
            if j == 0:
                ax.set_ylabel(str(c), rotation=0, labelpad=14, fontsize=12, va="center")
    fig.suptitle("Learned receptive fields, grouped by assigned digit", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    imgs, lbls = encoding.load_mnist()

    tr_idx = balanced_subset(imgs, lbls, a.train_per_class, rng)
    te_idx = balanced_subset(imgs, lbls, a.test_per_class, rng, exclude=tr_idx)
    tr_x, tr_y = imgs[tr_idx], lbls[tr_idx]
    te_x, te_y = imgs[te_idx], lbls[te_idx]
    print(f"[train] n_exc={a.n_exc} train={len(tr_x)} test={len(te_x)} "
          f"epochs={a.epochs} T={a.T}")

    net = Network(784, a.n_exc, rng=rng)

    t0 = time.time()
    for ep in range(a.epochs):
        te0 = time.time()
        net.train(tr_x, epochs=1, T=a.T)
        print(f"[train] epoch {ep + 1}/{a.epochs} done in {time.time() - te0:.1f}s")
    train_s = time.time() - t0

    net.assign_labels(tr_x, tr_y, T=a.T, n_classes=N_CLASSES)
    dist = np.bincount(net.neuron_labels, minlength=N_CLASSES)
    cm, acc, per_class = confusion_and_acc(net, te_x, te_y, a.T)

    print(f"\n[result] trained in {train_s:.1f}s")
    print(f"[result] test accuracy = {acc:.3f}  (chance = {1/N_CLASSES:.3f})")
    print("[result] neurons per digit:", dict(enumerate(dist.tolist())))
    print("[result] per-class accuracy:")
    for c in range(N_CLASSES):
        print(f"   digit {c}: {per_class[c]:.2f}  ({dist[c]} neurons)")

    plot_confusion(cm, acc, os.path.join(OUT, f"{a.tag}_confusion.png"))
    plot_receptive_fields(net, os.path.join(OUT, f"{a.tag}_receptive_fields.png"))
    net.save(os.path.join(OUT, f"{a.tag}_net.npz"))
    print(f"[result] wrote {a.tag}_confusion.png, {a.tag}_receptive_fields.png, {a.tag}_net.npz")
    return acc


if __name__ == "__main__":
    main()
