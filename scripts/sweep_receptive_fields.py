"""Receptive-field hyperparameter sweep (GPU, TorchSNN).

Why this exists
---------------
The big-data hypothesis was FALSIFIED (CLAUDE.md): 10x more presentations did NOT
clean up the noisy scattered-pixel receptive fields, and accuracy stayed flat
(~66%). So the bottleneck is the LEARNING DYNAMICS, not data volume. This script
sweeps the dynamics levers that *should* concentrate weight mass into clean digit
strokes, and MEASURES cleanliness (not just accuracy, which has plateaued).

Levers swept (all TorchSNN constructor / train args):
  norm_target  weight L1 budget per neuron -- lower => sparser => (hopefully) cleaner
  a_minus      LTD magnitude -- stronger pruning of off-stroke pixels
  a_plus       LTP magnitude -- stronger potentiation of on-stroke pixels
  w_inh        lateral competition -- sharper specialization
  T            integration window
  n_exc        layer size -- finer specialization (needs more data)

Metrics (printed + saved per config to outputs/sweeps/results.json):
  acc         test accuracy on a FIXED test set (comparable across configs)
  tmpl_match  mean Pearson corr(receptive field, class-mean image of its label).
              The direct "does this neuron look like its digit" score; HIGHER=cleaner.
  tv          mean normalized total-variation of the receptive fields.
              Scattered specks => high TV; smooth strokes => low. LOWER=cleaner.
  dead        fraction of neurons that never respond (wasted capacity). LOWER better.
  cov_spread  max/min per-class neuron coverage (class balance). LOWER better.

Resumable: every finished config is appended to results.json and SKIPPED on a
re-run, so an OOM/crash mid-sweep loses nothing and unfinished configs can be
edited and the script re-launched.

Run (GPU):
  set KMP_DUPLICATE_LIB_OK=TRUE
  C:\\Users\\User\\anaconda3\\python.exe scripts/sweep_receptive_fields.py
Smoke test (tiny, ~1 min) to validate end to end before the long run:
  SB_SMOKE=1 ... python scripts/sweep_receptive_fields.py
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain.torch_snn import TorchSNN, default_device
from synthbrain import encoding

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "sweeps")
os.makedirs(OUT, exist_ok=True)
CLASSES = list(range(10))
TEST_PER_CLASS = 150           # fixed held-out test set, same for every config
SMOKE = os.environ.get("SB_SMOKE") == "1"
WAVE = os.environ.get("SB_WAVE", "1")   # "1" = OFAT screen (done), "2" = a_plus x theta_plus

# Wave 1 shared defaults; each config overrides a subset.
BASE_W1 = dict(
    n_exc=400, n_per_class=300, epochs=6, T=350,
    a_plus=0.01, a_minus=0.012, norm_target=90.0, w_inh=0.6,
    theta_plus=0.05, r_m=6.0, input_norm_power=0.5, batch=128, max_rate=63.75,
)

# Ordered most-informative-first so an early stop still yields signal.
# Phase 1: one-factor-at-a-time around the baseline (fast, n_exc=400, ~18k pres).
# Phase 2: combine the cleanliness levers (sparsity + competition + pruning).
# Phase 3: scale neurons with proportionally more data.
CONFIGS_W1 = {
    # -- Phase 1: OFAT screening --
    "baseline":   {},
    "norm60":     dict(norm_target=60),
    "norm45":     dict(norm_target=45),
    "norm30":     dict(norm_target=30),
    "aminus018":  dict(a_minus=0.018),
    "aminus024":  dict(a_minus=0.024),
    "aminus036":  dict(a_minus=0.036),
    "aplus015":   dict(a_plus=0.015),
    "aplus020":   dict(a_plus=0.020),
    "T500":       dict(T=500),
    "T700":       dict(T=700),
    "winh10":     dict(w_inh=1.0),
    "winh15":     dict(w_inh=1.5),
    # -- Phase 2: combine the promising levers at n_exc=400 --
    "combo_A":    dict(norm_target=45, w_inh=1.0, a_minus=0.018),
    "combo_B":    dict(norm_target=45, w_inh=1.0, a_minus=0.024, a_plus=0.012),
    # -- Phase 3: scale the layer (more data for bigger nets) --
    "n800_base":      dict(n_exc=800, n_per_class=450, epochs=8),
    "n800_combo":     dict(n_exc=800, n_per_class=450, epochs=8,
                           norm_target=45, w_inh=1.0, a_minus=0.018),
    "n1600_combo":    dict(n_exc=1600, n_per_class=600, epochs=8, batch=96,
                           norm_target=45, w_inh=1.0, a_minus=0.018),
    "n1600_combo_lng": dict(n_exc=1600, n_per_class=600, epochs=12, batch=96,
                            norm_target=45, w_inh=1.0, a_minus=0.018),
}

# Wave 2: a_plus x theta_plus at n_exc=800.
# Wave 1 finding: a_plus is THE template-cleanliness lever (tmpl_match -0.13 ->
# +0.49 at a_plus=0.02, visibly clean digit strokes) but strong LTP kills ~43% of
# neurons -> accuracy collapses (0.65 -> 0.40). The Diehl & Cook fix is stronger
# adaptive-threshold homeostasis (theta_plus): firing raises a neuron's OWN
# threshold so quiet neurons get a turn, preventing the winners from starving the
# rest. This grid pairs the cleanliness lever (a_plus) with stronger homeostasis
# (theta_plus, baseline 0.05 is too weak) at the accuracy-friendly n_exc=800,
# hunting a config that is BOTH clean (tmpl>0) AND accurate (acc>=0.70, dead~0).
BASE_W2 = dict(
    n_exc=800, n_per_class=450, epochs=8, T=350,
    a_plus=0.016, a_minus=0.012, norm_target=90.0, w_inh=0.6,
    theta_plus=0.2, r_m=6.0, input_norm_power=0.5, batch=128, max_rate=63.75,
)
CONFIGS_W2 = {
    "a016_th20": dict(a_plus=0.016, theta_plus=0.2),   # best a-priori guess
    "a020_th20": dict(a_plus=0.020, theta_plus=0.2),
    "a016_th10": dict(a_plus=0.016, theta_plus=0.1),
    "a020_th40": dict(a_plus=0.020, theta_plus=0.4),
    "a013_th10": dict(a_plus=0.013, theta_plus=0.1),
    "a020_th10": dict(a_plus=0.020, theta_plus=0.1),
    "a016_th40": dict(a_plus=0.016, theta_plus=0.4),
    "a013_th20": dict(a_plus=0.013, theta_plus=0.2),
    "a013_th40": dict(a_plus=0.013, theta_plus=0.4),
    "a020_th05": dict(a_plus=0.020, theta_plus=0.05),  # control: clean LTP, NO homeostasis fix
}

# Wave 3: push theta_plus past the wave-2 grid edge. Wave 2's best corner was the
# HIGHEST theta tested (a020_th40: acc .48 / tmpl .43 / dead .21) and dead was
# still falling -- so the optimum is beyond 0.4, not at it. This extends the
# a_plus=0.02 line (and a check at 0.016) to theta {0.6,0.8,1.2} to find where
# homeostasis saturates and how much more accuracy it buys back.
BASE_W3 = dict(BASE_W2)
CONFIGS_W3 = {
    "a020_th60":  dict(a_plus=0.020, theta_plus=0.6),
    "a020_th80":  dict(a_plus=0.020, theta_plus=0.8),
    "a020_th120": dict(a_plus=0.020, theta_plus=1.2),
    "a016_th60":  dict(a_plus=0.016, theta_plus=0.6),
    "a016_th80":  dict(a_plus=0.016, theta_plus=0.8),
}

if WAVE == "3":
    BASE, CONFIGS = BASE_W3, CONFIGS_W3
    RESULTS = os.path.join(OUT, "results_wave3.json")
elif WAVE == "2":
    BASE, CONFIGS = BASE_W2, CONFIGS_W2
    RESULTS = os.path.join(OUT, "results_wave2.json")
else:
    BASE, CONFIGS = BASE_W1, CONFIGS_W1
    RESULTS = os.path.join(OUT, "results.json")

if SMOKE:
    # tiny, fast end-to-end validation of the harness
    CONFIGS = {
        "smoke_base": dict(n_per_class=20, epochs=1, T=80, n_exc=60),
        "smoke_sparse": dict(n_per_class=20, epochs=1, T=80, n_exc=60, norm_target=45),
    }
    TEST_PER_CLASS = 20


def load_split():
    """Load MNIST once; return (pool_x, pool_y, test_x, test_y, class_means, shape).

    A fixed test set (first TEST_PER_CLASS per class) is held out so accuracy is
    comparable across configs. class_means are the raw per-class mean images used
    by the tmpl_match metric.
    """
    imgs, lbls = encoding.load_mnist()
    pool_idx, test_idx = [], []
    for c in CLASSES:
        idx = np.where(lbls == c)[0]
        test_idx.append(idx[:TEST_PER_CLASS])
        pool_idx.append(idx[TEST_PER_CLASS:])
    test_idx = np.concatenate(test_idx)
    test_x, test_y = imgs[test_idx], lbls[test_idx]
    shape = imgs.shape[1:]
    n_classes = len(CLASSES)
    class_means = np.zeros((n_classes, int(np.prod(shape))))
    for c in CLASSES:
        class_means[c] = imgs[lbls == c].mean(axis=0).ravel()
    return pool_idx, imgs, lbls, test_x, test_y, class_means, shape


def draw_train(pool_idx, imgs, lbls, n_per_class, rng):
    sel = np.concatenate([pool_idx[c][:n_per_class] for c in range(len(CLASSES))])
    rng.shuffle(sel)
    return imgs[sel], lbls[sel]


def tmpl_match(net, shape, class_means):
    """Mean Pearson corr between each receptive field and its label's class mean."""
    rf = net.receptive_fields(shape).reshape(net.n_exc, -1)
    labels = net.neuron_labels
    corrs = []
    for i in range(net.n_exc):
        a = rf[i] - rf[i].mean()
        b = class_means[labels[i]] - class_means[labels[i]].mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 0:
            corrs.append(float(a @ b / denom))
    return float(np.mean(corrs)) if corrs else 0.0


def mean_tv(net, shape):
    """Mean total-variation of mass-normalized receptive fields (lower=smoother)."""
    rf = net.receptive_fields(shape)
    tvs = []
    for r in rf:
        s = r.sum()
        if s <= 0:
            continue
        rn = r / s
        tvs.append(float(np.abs(np.diff(rn, axis=0)).sum()
                         + np.abs(np.diff(rn, axis=1)).sum()))
    return float(np.mean(tvs)) if tvs else float("nan")


def dead_frac(net):
    resp = net.neuron_response  # (n_exc, n_classes) avg counts; from assign_labels
    if resp is None:
        return float("nan")
    return float((resp.sum(axis=1) <= 1e-9).mean())


def cov_spread(net, n_classes):
    cov = np.bincount(net.neuron_labels, minlength=n_classes).astype(float)
    return float(cov.max() / max(cov.min(), 1.0)), int((cov == 0).sum())


def save_rf_png(net, shape, name, n_show=64):
    rf = net.receptive_fields(shape)
    n_show = min(n_show, net.n_exc)
    cols = int(np.ceil(np.sqrt(n_show)))
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.05, rows * 1.05))
    for i, ax in enumerate(np.array(axes).ravel()):
        if i < n_show:
            ax.imshow(rf[i], cmap="hot")
            if net.neuron_labels is not None:
                ax.set_title(str(net.neuron_labels[i]), fontsize=6, pad=1)
        ax.axis("off")
    fig.suptitle(f"{name}  (sample {n_show}/{net.n_exc})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"rf_{name}.png"), dpi=120)
    plt.close(fig)


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            return json.load(fh)
    return []


def print_leaderboard(results):
    if not results:
        return
    rows = sorted(results, key=lambda r: r["acc"], reverse=True)
    print("\n=== leaderboard (by acc) ===")
    print(f"{'config':18s} {'acc':>6s} {'tmpl':>6s} {'tv':>7s} {'dead':>5s} "
          f"{'covsp':>6s} {'sec':>6s}")
    for r in rows:
        print(f"{r['name']:18s} {r['acc']:6.3f} {r['tmpl_match']:6.3f} "
              f"{r['tv']:7.4f} {r['dead']:5.2f} {r['cov_spread']:6.1f} {r['sec']:6.0f}")
    print("baseline ref: NumPy lateral = 0.688 acc; templates were noisy (tmpl low).\n")


def run_one(name, cfg, data):
    pool_idx, imgs, lbls, test_x, test_y, class_means, shape = data
    p = dict(BASE, **cfg)
    n_input = int(np.prod(shape))
    n_classes = len(CLASSES)
    device = default_device()

    rng = np.random.default_rng(0)
    tr_x, tr_y = draw_train(pool_idx, imgs, lbls, p["n_per_class"], rng)

    net = TorchSNN(
        n_input=n_input, n_exc=p["n_exc"], input_norm_power=p["input_norm_power"],
        w_inh=p["w_inh"], a_plus=p["a_plus"], a_minus=p["a_minus"],
        norm_target=p["norm_target"], theta_plus=p["theta_plus"], r_m=p["r_m"],
        stdp_update="sequential", device=device, dtype=torch.float32, seed=0,
    )

    t0 = time.time()
    net.train(tr_x, epochs=p["epochs"], T=p["T"], batch_size=p["batch"],
              progress=False, rng=np.random.default_rng(0))
    if device == "cuda":
        torch.cuda.synchronize()
    sec = time.time() - t0

    net.assign_labels(tr_x, tr_y, T=p["T"], n_classes=n_classes)
    acc = net.evaluate(test_x, test_y, T=p["T"])
    tm = tmpl_match(net, shape, class_means)
    tv = mean_tv(net, shape)
    dead = dead_frac(net)
    spread, n_zero = cov_spread(net, n_classes)
    save_rf_png(net, shape, name)

    rec = dict(name=name, params=p, acc=acc, tmpl_match=tm, tv=tv, dead=dead,
               cov_spread=spread, n_zero_classes=n_zero, sec=sec,
               n_train=int(len(tr_x)), presentations=int(len(tr_x) * p["epochs"]))
    print(f"[{name}] acc={acc:.3f} tmpl={tm:.3f} tv={tv:.4f} dead={dead:.2f} "
          f"covsp={spread:.1f} zero={n_zero} in {sec:.0f}s "
          f"(pres={rec['presentations']}, n_exc={p['n_exc']})")
    return rec


def main():
    device = default_device()
    if device == "cuda":
        print(f"[sweep] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[sweep] WARNING: no CUDA -- running on CPU, will be slow.")
    print(f"[sweep] {len(CONFIGS)} configs; results -> {RESULTS}")

    data = load_split()
    results = load_results()
    done = {r["name"] for r in results}
    if done:
        print(f"[sweep] resuming; {len(done)} already done: {sorted(done)}")

    for name, cfg in CONFIGS.items():
        if name in done:
            continue
        try:
            rec = run_one(name, cfg, data)
        except RuntimeError as e:
            # e.g. CUDA OOM -- record and continue so the rest of the sweep runs
            print(f"[{name}] FAILED: {e}")
            if device == "cuda":
                torch.cuda.empty_cache()
            rec = dict(name=name, params=dict(BASE, **cfg), acc=float("nan"),
                       tmpl_match=float("nan"), tv=float("nan"), dead=float("nan"),
                       cov_spread=float("nan"), n_zero_classes=-1, sec=0.0,
                       error=str(e))
        results.append(rec)
        with open(RESULTS, "w") as fh:
            json.dump(results, fh, indent=2)
        print_leaderboard([r for r in results if not np.isnan(r["acc"])])

    print("[sweep] DONE.")


if __name__ == "__main__":
    main()
