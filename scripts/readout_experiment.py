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
from sweep_receptive_fields import (  # noqa: E402
    load_split, draw_train, CLASSES, tmpl_match, dead_frac, save_rf_png)
from synthbrain.torch_snn import TorchSNN, default_device           # noqa: E402

OUT = os.path.join(ROOT, "outputs", "sweeps")
os.makedirs(OUT, exist_ok=True)
RESULTS = os.path.join(OUT, "readout_results.json")

T_TRAIN = 350
T_LONG = 700

# Nets to train + probe. The first two are the wave-2 trade-off endpoints (already
# in readout_results.json -> skipped on resume). The last two are the wave-4 winner
# (a_plus=0.02, theta_plus=2.0) at n_exc=800 and 1600: the probe gives the true
# headline accuracy for the CLEANEST templates, and n1600 tests the last scaling lever.
NETS = {
    "clean_a020_th40": dict(n_exc=800, n_per_class=450, epochs=8,
                            a_plus=0.020, a_minus=0.012, norm_target=90.0,
                            w_inh=0.6, theta_plus=0.4, r_m=6.0,
                            input_norm_power=0.5, batch=128),
    "noisy_n800_base": dict(n_exc=800, n_per_class=450, epochs=8,
                            a_plus=0.010, a_minus=0.012, norm_target=90.0,
                            w_inh=0.6, theta_plus=0.05, r_m=6.0,
                            input_norm_power=0.5, batch=128),
    "best_th200_n800": dict(n_exc=800, n_per_class=450, epochs=8,
                            a_plus=0.020, a_minus=0.012, norm_target=90.0,
                            w_inh=0.6, theta_plus=2.0, r_m=6.0,
                            input_norm_power=0.5, batch=128),
    "best_th200_n1600": dict(n_exc=1600, n_per_class=600, epochs=8,
                             a_plus=0.020, a_minus=0.012, norm_target=90.0,
                             w_inh=0.6, theta_plus=2.0, r_m=6.0,
                             input_norm_power=0.5, batch=96),
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
    global T_TRAIN, T_LONG, RESULTS
    if os.environ.get("SB_SMOKE") == "1":
        T_TRAIN, T_LONG = 80, 120
        for p in NETS.values():
            p.update(n_exc=60, n_per_class=20, epochs=1, batch=16)
        RESULTS = os.path.join(OUT, "readout_results_smoke.json")  # don't touch real results
    device = default_device()
    print(f"[readout] device={device}")
    pool_idx, imgs, lbls, te_x, te_y, class_means, shape = load_split()
    n_input = int(np.prod(shape))
    n_classes = len(CLASSES)

    # Resume: keep nets already in readout_results.json, train only the new ones.
    results = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            results = json.load(fh)
    done = {r["net"] for r in results}
    if done:
        print(f"[readout] resuming; skipping already-done: {sorted(done)}")

    for name, p in NETS.items():
        if name in done:
            continue
        tr_x, tr_y = draw_train(pool_idx, imgs, lbls, p["n_per_class"],
                                np.random.default_rng(0))
        p["tr_x"] = tr_x
        t0 = time.time()
        net = build_and_train(p, n_input, device)
        sec = time.time() - t0
        print(f"[{name}] trained in {sec:.0f}s; evaluating readouts...")

        row = {"net": name, "n_exc": p["n_exc"], "train_sec": sec}
        # native@T350 also assigns neuron_labels@T350 -> use those for tmpl/RF
        row["native_T350"] = native_acc(net, tr_x, tr_y, te_x, te_y, T_TRAIN, n_classes)
        row["tmpl_match"] = tmpl_match(net, shape, class_means)
        row["dead"] = dead_frac(net)
        save_rf_png(net, shape, name)
        row["native_T700"] = native_acc(net, tr_x, tr_y, te_x, te_y, T_LONG, n_classes)
        row["probe_T350"] = probe_acc(net, tr_x, tr_y, te_x, te_y, T_TRAIN)
        row["probe_T700"] = probe_acc(net, tr_x, tr_y, te_x, te_y, T_LONG)
        results.append(row)
        print(f"[{name}] native@T350={row['native_T350']:.3f}  "
              f"native@T700={row['native_T700']:.3f}  "
              f"probe@T350={row['probe_T350']:.3f}  probe@T700={row['probe_T700']:.3f}  "
              f"tmpl={row['tmpl_match']:.3f}  dead={row['dead']:.3f}")
        with open(RESULTS, "w") as fh:
            json.dump(results, fh, indent=2)

    print("\n=== readout comparison ===")
    print(f"{'net':20s} {'n_exc':>5s} {'nat@350':>8s} {'nat@700':>8s} "
          f"{'prb@350':>8s} {'prb@700':>8s} {'tmpl':>6s} {'dead':>6s}")
    for r in results:
        print(f"{r['net']:20s} {r.get('n_exc', 0):5d} {r['native_T350']:8.3f} "
              f"{r['native_T700']:8.3f} {r['probe_T350']:8.3f} {r['probe_T700']:8.3f} "
              f"{r.get('tmpl_match', float('nan')):6.3f} {r.get('dead', float('nan')):6.3f}")
    print("\nnative = project's mean-spike-count readout; prb = linear probe on counts.")
    print("[readout] DONE.")


if __name__ == "__main__":
    main()
