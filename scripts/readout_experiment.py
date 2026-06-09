"""Readout experiment: is the clean-vs-accurate gap a READOUT problem?

Wave 2 left a puzzle: the CLEAN net (a_plus=0.02, theta_plus=0.4) has gorgeous
digit templates but only 0.48 acc, while the NOISY net (n800_base) has scattered-
speck templates but 0.71 acc. Hypothesis: clean templates => highly SELECTIVE
neurons => SPARSE firing => the project's naive mean-spike-count readout is too
weak to decode them, even though the information is present. If so, a better
readout (or longer integration) should help the CLEAN net MORE than the noisy one
and close the gap. If instead a strong readout stays low on the clean net, the
clean representation genuinely lost discriminative information (a real trade-off).

This trains both nets once, then evaluates several readouts on the SAME frozen
spike-count features. The SNN stays backprop-free; the linear probe is a standard
linear-probe diagnostic on frozen unsupervised features (labels used only at
readout, consistent with the project's eval-only label rule).

Readouts compared (per net):
  native@T350   the project's mean-spike-count readout (reproduces the sweep acc)
  native@T700   same readout, but 2x integration time (does more spikes help
                sparse clean neurons read out better?)
  probe@T350    multinomial logistic regression fit on train spike counts
  probe@T700    linear probe at the longer integration window

Writes outputs/sweeps/readout_results.json.
"""

import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from sweep_receptive_fields import load_split, draw_train, CLASSES  # noqa: E402
from synthbrain.torch_snn import TorchSNN, default_device           # noqa: E402

OUT = os.path.join(ROOT, "outputs", "sweeps")
os.makedirs(OUT, exist_ok=True)
RESULTS = os.path.join(OUT, "readout_results.json")

T_TRAIN = 350
T_LONG = 700

# The two endpoints of the wave-2 trade-off, retrained here so we can probe them.
NETS = {
    "clean_a020_th40": dict(n_exc=800, n_per_class=450, epochs=8,
                            a_plus=0.020, a_minus=0.012, norm_target=90.0,
                            w_inh=0.6, theta_plus=0.4, r_m=6.0,
                            input_norm_power=0.5, batch=128),
    "noisy_n800_base": dict(n_exc=800, n_per_class=450, epochs=8,
                            a_plus=0.010, a_minus=0.012, norm_target=90.0,
                            w_inh=0.6, theta_plus=0.05, r_m=6.0,
                            input_norm_power=0.5, batch=128),
}


def build_and_train(p, n_input, device):
    net = TorchSNN(
        n_input=n_input, n_exc=p["n_exc"], input_norm_power=p["input_norm_power"],
        w_inh=p["w_inh"], a_plus=p["a_plus"], a_minus=p["a_minus"],
        norm_target=p["norm_target"], theta_plus=p["theta_plus"], r_m=p["r_m"],
        stdp_update="sequential", device=device, dtype=torch.float32, seed=0,
    )
    net.train(p["tr_x"], epochs=p["epochs"], T=T_TRAIN, batch_size=p["batch"],
              progress=False, rng=np.random.default_rng(0))
    if device == "cuda":
        torch.cuda.synchronize()
    return net


def native_acc(net, tr_x, tr_y, te_x, te_y, T, n_classes):
    """The project's mean-spike-count readout at integration window T."""
    net.assign_labels(tr_x, tr_y, T=T, n_classes=n_classes)
    return net.evaluate(te_x, te_y, T=T)


def probe_acc(net, tr_x, tr_y, te_x, te_y, T):
    """Linear probe: multinomial logistic regression on frozen spike counts."""
    Xtr = net.counts(tr_x, T=T).astype(np.float64)
    Xte = net.counts(te_x, T=T).astype(np.float64)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(Xtr), tr_y)
    return float(clf.score(scaler.transform(Xte), te_y))


def main():
    global T_TRAIN, T_LONG
    if os.environ.get("SB_SMOKE") == "1":
        T_TRAIN, T_LONG = 80, 120
        for p in NETS.values():
            p.update(n_exc=60, n_per_class=20, epochs=1, batch=16)
    device = default_device()
    print(f"[readout] device={device}")
    pool_idx, imgs, lbls, te_x, te_y, _, shape = load_split()
    n_input = int(np.prod(shape))
    n_classes = len(CLASSES)

    results = []
    for name, p in NETS.items():
        tr_x, tr_y = draw_train(pool_idx, imgs, lbls, p["n_per_class"],
                                np.random.default_rng(0))
        p["tr_x"] = tr_x
        t0 = time.time()
        net = build_and_train(p, n_input, device)
        sec = time.time() - t0
        print(f"[{name}] trained in {sec:.0f}s; evaluating readouts...")

        row = {"net": name, "train_sec": sec}
        row["native_T350"] = native_acc(net, tr_x, tr_y, te_x, te_y, T_TRAIN, n_classes)
        row["native_T700"] = native_acc(net, tr_x, tr_y, te_x, te_y, T_LONG, n_classes)
        row["probe_T350"] = probe_acc(net, tr_x, tr_y, te_x, te_y, T_TRAIN)
        row["probe_T700"] = probe_acc(net, tr_x, tr_y, te_x, te_y, T_LONG)
        results.append(row)
        print(f"[{name}] native@T350={row['native_T350']:.3f}  "
              f"native@T700={row['native_T700']:.3f}  "
              f"probe@T350={row['probe_T350']:.3f}  probe@T700={row['probe_T700']:.3f}")
        with open(RESULTS, "w") as fh:
            json.dump(results, fh, indent=2)

    print("\n=== readout comparison ===")
    print(f"{'net':18s} {'nat@350':>8s} {'nat@700':>8s} {'prb@350':>8s} {'prb@700':>8s}")
    for r in results:
        print(f"{r['net']:18s} {r['native_T350']:8.3f} {r['native_T700']:8.3f} "
              f"{r['probe_T350']:8.3f} {r['probe_T700']:8.3f}")
    if len(results) == 2:
        clean, noisy = results[0], results[1]
        gap_native = noisy["native_T350"] - clean["native_T350"]
        gap_probe = noisy["probe_T700"] - clean["probe_T700"]
        print(f"\nclean-vs-noisy gap: native@T350={gap_native:+.3f}  "
              f"best-probe@T700={gap_probe:+.3f}")
        print("If the probe gap << native gap, the clean net HAD the info and the "
              "naive readout was the bottleneck. If it persists, the trade-off is real.")
    print("[readout] DONE.")


if __name__ == "__main__":
    main()
