"""GPU-accelerated scale-up training (PyTorch port of scaleup_network.py).

Same model and evaluation as scaleup_network.py, but batched through
synthbrain.torch_snn.TorchSNN so it runs on the GPU. Because the timestep
recurrence is sequential per image, batching is *across images*, which means
MINI-BATCH STDP (all images in a batch see the same start-of-batch weights;
updates are accumulated and applied once per batch). That is a deliberate,
documented change to the learning dynamics vs the strictly-online NumPy path --
so compare accuracy against scaleup_network.py's lateral baseline (68.8%) rather
than assuming parity.

Requires a CUDA build of PyTorch for GPU. The current env has CPU-only torch; to
use the RTX 4060 install e.g.:
    pip install --index-url https://download.pytorch.org/whl/cu124 torch
The script auto-detects: it runs on CUDA if available, otherwise falls back to CPU
(and says so). With more data/epochs/presentations this is where clean receptive
fields become reachable in minutes instead of hours.

Produces (same as scaleup): outputs/gpu_receptive_fields.png, outputs/gpu_net.npz,
and prints accuracy + per-digit coverage/recall.
"""

import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain.torch_snn import TorchSNN, default_device
from synthbrain import encoding

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)

# -- scale-up knobs (the GPU headroom is here: push N_PER_CLASS / EPOCHS up) --
CLASSES = list(range(10))
N_PER_CLASS = 150           # raise this (and EPOCHS) on GPU for cleaner templates
N_EXC = 400
EPOCHS = 4
T = 350
BATCH_SIZE = 128            # images in parallel. On GPU, SMALL batches starve it
                            # (kernels too tiny, launch overhead dominates) -- 32 gave
                            # only ~3x over CPU. 128-256 saturates the 4060. In
                            # "sequential" mode each image still renorms in the apply
                            # loop, so the fidelity cost of a bigger batch is only the
                            # longer frozen-W forward window. Sweep vs the 68.8%
                            # baseline; drop it if accuracy slips, raise it for speed.
INPUT_NORM_POWER = 0.5
W_INH = 0.6                 # lateral inhibition strength (the winning default)
# How per-batch weight updates are applied. "sequential" (default) applies each
# image's delta with an L1-renorm between them -> online-like, closes most of the
# mini-batch accuracy gap. "summed" is faster but over-saturates at large batch.
STDP_UPDATE = "sequential"

# Override the big knobs from the environment so you can scale up WITHOUT editing
# this file, e.g.  SB_N_PER_CLASS=600 SB_EPOCHS=10 SB_BATCH=128 SB_N_EXC=400
N_PER_CLASS = int(os.environ.get("SB_N_PER_CLASS", N_PER_CLASS))
EPOCHS = int(os.environ.get("SB_EPOCHS", EPOCHS))
BATCH_SIZE = int(os.environ.get("SB_BATCH", BATCH_SIZE))
N_EXC = int(os.environ.get("SB_N_EXC", N_EXC))


def get_data(rng):
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
    fig.suptitle(f"GPU-trained receptive fields (sample of {n_show}/{net.n_exc})", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gpu_receptive_fields.png"), dpi=130)
    plt.close(fig)


def per_class_report(net, te_x, te_y, n_classes):
    labels = np.asarray(te_y)
    coverage = np.bincount(net.neuron_labels, minlength=n_classes)
    preds = net.predict(te_x, T=T)
    print("[report] per-digit neuron coverage / test recall:")
    for c in range(n_classes):
        mask = labels == c
        recall = (preds[mask] == c).mean() if mask.any() else float("nan")
        print(f"  digit {c}: {coverage[c]:3d} neurons   recall={recall:.2f}  (n={int(mask.sum())})")


def main():
    rng = np.random.default_rng(0)
    tr_x, tr_y, te_x, te_y, shape, source = get_data(rng)
    n_input = int(np.prod(shape))
    n_classes = len(np.unique(np.concatenate([tr_y, te_y])))

    device = default_device()
    if device == "cuda":
        print(f"[gpu] CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("[gpu] WARNING: CUDA not available -- running on CPU (install a CUDA "
              "torch build for GPU). Dynamics are identical; it will just be slower.")

    print(f"[gpu] source={source}  train={len(tr_x)}  test={len(te_x)}  input={n_input}  "
          f"classes={n_classes}  n_exc={N_EXC}  epochs={EPOCHS}  T={T}  batch={BATCH_SIZE}  "
          f"input_norm_power={INPUT_NORM_POWER}  w_inh={W_INH}  stdp_update={STDP_UPDATE}")

    net = TorchSNN(n_input=n_input, n_exc=N_EXC, input_norm_power=INPUT_NORM_POWER,
                   w_inh=W_INH, stdp_update=STDP_UPDATE, device=device,
                   dtype=torch.float32, seed=0)

    t0 = time.time()
    net.train(tr_x, epochs=EPOCHS, T=T, batch_size=BATCH_SIZE, progress=True,
              rng=np.random.default_rng(0))
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"[gpu] trained in {time.time() - t0:.1f}s")

    net.assign_labels(tr_x, tr_y, T=T, n_classes=n_classes)
    acc = net.evaluate(te_x, te_y, T=T)
    chance = 1.0 / n_classes
    print(f"[gpu] test accuracy = {acc:.3f}  (chance = {chance:.3f}; "
          f"NumPy lateral baseline = 0.688)")

    per_class_report(net, te_x, te_y, n_classes)
    plot_receptive_fields(net, shape)
    net.save(os.path.join(OUT, "gpu_net.npz"))
    print(f"[gpu] wrote receptive fields + gpu_net.npz to {OUT}")
    return acc, chance


if __name__ == "__main__":
    acc, chance = main()
    assert acc > chance, "clustering accuracy did not beat chance"
    print("\nGPU scale-up run complete.")
