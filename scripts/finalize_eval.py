"""Finalize the best model: multi-seed accuracy (mean +/- std) + confusion matrix.

The headline 0.905 (linear probe) was single-seed; accuracy wobbles a few points
across RNG streams (weight init, Poisson encoding, batch order). This retrains the
winning recipe (a_plus=0.02, theta_plus=2.0, n_exc=800) over several seeds and
reports mean +/- std for BOTH readouts, then builds a confusion matrix from the
linear probe of a representative seed (which digits actually confuse).

Writes:
  outputs/sweeps/finalize_results.json     per-seed native + probe accuracy + summary
  outputs/samples/confusion_matrix.png     linear-probe confusion matrix (seed 0)

Resumable: per-seed results flush to JSON and are skipped on a re-run.

Run (GPU):
  set KMP_DUPLICATE_LIB_OK=TRUE
  C:\\Users\\User\\anaconda3\\python.exe scripts/finalize_eval.py
Smoke: SB_SMOKE=1 ... (tiny nets/seeds, temp results file).
"""

import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from sweep_receptive_fields import load_split, draw_train, CLASSES  # noqa: E402
from synthbrain.torch_snn import TorchSNN, default_device           # noqa: E402

OUT = os.path.join(ROOT, "outputs", "sweeps")
SAMPLES = os.path.join(ROOT, "outputs", "samples")
os.makedirs(OUT, exist_ok=True)
os.makedirs(SAMPLES, exist_ok=True)
RESULTS = os.path.join(OUT, "finalize_results.json")
CONF_PNG = os.path.join(SAMPLES, "confusion_matrix.png")

SEEDS = [0, 1, 2, 3, 4]
T = 350
T_PROBE = 700
RECIPE = dict(n_exc=800, n_per_class=450, epochs=8, a_plus=0.020, a_minus=0.012,
              norm_target=90.0, w_inh=0.6, theta_plus=2.0, r_m=6.0,
              input_norm_power=0.5, batch=128)


def train_net(p, tr_x, n_input, device, seed):
    net = TorchSNN(n_input=n_input, n_exc=p["n_exc"], input_norm_power=p["input_norm_power"],
                   w_inh=p["w_inh"], a_plus=p["a_plus"], a_minus=p["a_minus"],
                   norm_target=p["norm_target"], theta_plus=p["theta_plus"], r_m=p["r_m"],
                   stdp_update="sequential", device=device, dtype=torch.float32, seed=seed)
    net.train(tr_x, epochs=p["epochs"], T=T, batch_size=p["batch"], progress=False,
              rng=np.random.default_rng(seed))
    if device == "cuda":
        torch.cuda.synchronize()
    return net


def probe_predict(net, tr_x, tr_y, te_x, T):
    """Fit logistic regression on spike counts; return test predictions."""
    Xtr = net.counts(tr_x, T=T).astype(np.float64)
    Xte = net.counts(te_x, T=T).astype(np.float64)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(Xtr), np.asarray(tr_y))
    return clf.predict(scaler.transform(Xte))


def plot_confusion(cm, acc, path):
    n = cm.shape[0]
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(n), yticks=range(n), xlabel="predicted digit",
           ylabel="true digit",
           title=f"Linear-probe confusion matrix (acc={acc:.3f})")
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "#333")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalized rate")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    global SEEDS, RESULTS
    if os.environ.get("SB_SMOKE") == "1":
        SEEDS = [0, 1]
        RECIPE.update(n_exc=60, n_per_class=20, epochs=1, batch=16)
        RESULTS = os.path.join(OUT, "finalize_results_smoke.json")

    device = default_device()
    print(f"[finalize] device={device}  seeds={SEEDS}  recipe n_exc={RECIPE['n_exc']} "
          f"a_plus={RECIPE['a_plus']} theta_plus={RECIPE['theta_plus']}")
    pool_idx, imgs, lbls, te_x, te_y, _, shape = load_split()
    n_input = int(np.prod(shape))
    n_classes = len(CLASSES)

    results = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            loaded = json.load(fh)
        # per-seed writes a list; the final write wraps it in {per_seed, summary}
        results = loaded["per_seed"] if isinstance(loaded, dict) else loaded
    done = {r["seed"] for r in results if "seed" in r}

    cm_payload = None
    for seed in SEEDS:
        if seed in done:
            continue
        tr_x, tr_y = draw_train(pool_idx, imgs, lbls, RECIPE["n_per_class"],
                                np.random.default_rng(seed))
        t0 = time.time()
        net = train_net(RECIPE, tr_x, n_input, device, seed)
        sec = time.time() - t0
        net.assign_labels(tr_x, tr_y, T=T, n_classes=n_classes)
        native = net.evaluate(te_x, te_y, T=T)
        preds = probe_predict(net, tr_x, tr_y, te_x, T_PROBE)
        probe = float((preds == np.asarray(te_y)).mean())
        results.append({"seed": seed, "native": native, "probe": probe, "sec": sec})
        print(f"[finalize] seed={seed}  native={native:.3f}  probe={probe:.3f}  ({sec:.0f}s)")
        with open(RESULTS, "w") as fh:
            json.dump(results, fh, indent=2)
        if seed == SEEDS[0]:   # confusion matrix from the first seed's probe
            cm_payload = (confusion_matrix(np.asarray(te_y), preds,
                                           labels=list(range(n_classes))), probe)

    natives = np.array([r["native"] for r in results])
    probes = np.array([r["probe"] for r in results])
    summary = {
        "native_mean": float(natives.mean()), "native_std": float(natives.std()),
        "probe_mean": float(probes.mean()), "probe_std": float(probes.std()),
        "n_seeds": len(results),
    }
    print("\n=== multi-seed summary ===")
    print(f"native readout: {summary['native_mean']:.3f} +/- {summary['native_std']:.3f}")
    print(f"linear probe  : {summary['probe_mean']:.3f} +/- {summary['probe_std']:.3f}  "
          f"(n={summary['n_seeds']} seeds, chance=0.10)")

    if cm_payload is not None:
        plot_confusion(cm_payload[0], cm_payload[1], CONF_PNG)
        print(f"[finalize] wrote {CONF_PNG}")

    with open(RESULTS, "w") as fh:
        json.dump({"per_seed": results, "summary": summary}, fh, indent=2)
    print("[finalize] DONE.")


if __name__ == "__main__":
    main()
