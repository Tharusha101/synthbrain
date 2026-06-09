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

# -- WINNING RECIPE (from scripts/sweep_receptive_fields.py; see CLAUDE.md) --
# a_plus=0.02 + theta_plus=2.0 at n_exc=800 give CLEAN, human-readable digit
# templates (tmpl_match 0.89, ZERO dead neurons) AND the best accuracy: 0.905 with
# the linear-probe readout (net.linear_probe), vs 0.615 with the native mean-count
# readout -- the native readout severely undersells these selective neurons. NOTE:
# n_exc=1600 was WORSE here (under-trained at this data budget); 800 is the sweet spot.
CLASSES = list(range(10))
N_PER_CLASS = 450
N_EXC = 800
EPOCHS = 8                  # 800 exc x 4500 imgs x 8 ep ~= 27 min on the RTX 4060
T = 350
BATCH_SIZE = 128            # 128-256 saturates the 4060; in "sequential" mode a bigger
                            # batch only lengthens the frozen-W forward window. Drop if
                            # accuracy slips, raise for speed.
INPUT_NORM_POWER = 0.5
W_INH = 0.6                 # lateral inhibition strength (won vs two_layer)
A_PLUS = 0.02               # LTP strength: THE template-cleanliness lever
THETA_PLUS = 2.0            # adaptive-threshold homeostasis: cancels the dead-neuron
                            # side-effect of strong LTP (saturates ~2.0; >2.5 over-regularizes)
STDP_UPDATE = "sequential"  # online-like mini-batch updates (vs faster "summed")

# Override knobs from the environment (no edit needed), e.g.
#   SB_N_PER_CLASS=600 SB_EPOCHS=10 SB_N_EXC=800 SB_A_PLUS=0.02 SB_THETA_PLUS=2.0
N_PER_CLASS = int(os.environ.get("SB_N_PER_CLASS", N_PER_CLASS))
EPOCHS = int(os.environ.get("SB_EPOCHS", EPOCHS))
BATCH_SIZE = int(os.environ.get("SB_BATCH", BATCH_SIZE))
N_EXC = int(os.environ.get("SB_N_EXC", N_EXC))
A_PLUS = float(os.environ.get("SB_A_PLUS", A_PLUS))
THETA_PLUS = float(os.environ.get("SB_THETA_PLUS", THETA_PLUS))


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
          f"input_norm_power={INPUT_NORM_POWER}  w_inh={W_INH}  a_plus={A_PLUS}  "
          f"theta_plus={THETA_PLUS}  stdp_update={STDP_UPDATE}")

    net = TorchSNN(n_input=n_input, n_exc=N_EXC, input_norm_power=INPUT_NORM_POWER,
                   w_inh=W_INH, a_plus=A_PLUS, theta_plus=THETA_PLUS,
                   stdp_update=STDP_UPDATE, device=device, dtype=torch.float32, seed=0)

    t0 = time.time()
    net.train(tr_x, epochs=EPOCHS, T=T, batch_size=BATCH_SIZE, progress=True,
              rng=np.random.default_rng(0))
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"[gpu] trained in {time.time() - t0:.1f}s")

    net.assign_labels(tr_x, tr_y, T=T, n_classes=n_classes)
    acc = net.evaluate(te_x, te_y, T=T)
    chance = 1.0 / n_classes
    # Native mean-count readout undersells selective neurons; the linear probe is
    # the representation's true accuracy (the headline number). See CLAUDE.md.
    probe = net.linear_probe(tr_x, tr_y, te_x, te_y, T=T)
    print(f"[gpu] native readout = {acc:.3f}   linear-probe = {probe:.3f}   "
          f"(chance = {chance:.3f})")

    per_class_report(net, te_x, te_y, n_classes)
    plot_receptive_fields(net, shape)
    net.save(os.path.join(OUT, "gpu_net.npz"))
    print(f"[gpu] wrote receptive fields + gpu_net.npz to {OUT}")
    return acc, probe, chance


if __name__ == "__main__":
    acc, probe, chance = main()
    assert probe > acc, "linear probe should beat the native readout"
    assert acc > chance, "clustering accuracy did not beat chance"
    print("\nGPU scale-up run complete.")
