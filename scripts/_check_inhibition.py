"""No-training diagnostic for the two-layer inhibitory wiring.

Builds an UNTRAINED network in each inhibition mode and presents a few MNIST
images WITHOUT learning, reporting:
  * exc activity   — total excitatory spikes + how many exc neurons were active
  * inh activity   — total inhibitory spikes (two_layer only); should track exc,
                     confirming exc_i -> inh_i actually drives the inh partner
  * WTA / top-10   — share of exc spikes from the 10 busiest neurons; higher =
                     more competitive (inh_i -> exc_{j!=i} is suppressing the rest)

This verifies the explicit inhibitory population fires and competes BEFORE
committing to a long training run; tune w_exc_inh / w_inh here, not in training.

CAUTION (learned the hard way): do NOT pick w_inh for the *sharpest* WTA. An
over-competitive layer (w_inh=3.0 -> top10=0.95, ~10/400 neurons ever win) starves
STDP and cratered scale-up accuracy to 29% (vs 66% lateral baseline). Tune for
competition *comparable to lateral* instead: match its active-count (~56/100) and
top-10 share (~0.36). That lands around two_layer w_inh~0.3-0.4.
"""

import os
import sys

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain.network import Network
from synthbrain import encoding

T = 350
N_EXC = 100
N_IMG = 5


def summarize(imgs, **net_kwargs):
    rng = np.random.default_rng(0)
    net = Network(
        n_input=imgs.shape[1] * imgs.shape[2], n_exc=N_EXC, rng=rng, **net_kwargs
    )
    exc_tot, exc_active, inh_tot, top_share = [], [], [], []
    for img in imgs:
        out = net.simulate(img, T=T)
        counts = out["exc_spikes"].sum(axis=0)
        exc_tot.append(int(counts.sum()))
        exc_active.append(int((counts > 0).sum()))
        if counts.sum() > 0:
            top_share.append(np.sort(counts)[::-1][:10].sum() / counts.sum())
        if "inh_spikes" in out:
            inh_tot.append(int(out["inh_spikes"].sum()))
    return {
        "exc_spikes/img": np.mean(exc_tot),
        "exc_active/img": np.mean(exc_active),
        "inh_spikes/img": np.mean(inh_tot) if inh_tot else float("nan"),
        "top10_share": np.mean(top_share) if top_share else float("nan"),
    }


def main():
    imgs, lbls = encoding.load_mnist()
    sel = np.concatenate([np.where(lbls == c)[0][:1] for c in range(N_IMG)])
    imgs = imgs[sel]

    print(f"Untrained networks, {N_IMG} images, T={T}, n_exc={N_EXC}\n")
    configs = [
        ("lateral w_inh=0.6 (baseline)", dict(inhibition="lateral", w_inh=0.6)),
        (
            "two_layer w_ei=40 w_inh=0.3",
            dict(inhibition="two_layer", w_exc_inh=40, w_inh=0.3),
        ),
        (
            "two_layer w_ei=40 w_inh=0.4",
            dict(inhibition="two_layer", w_exc_inh=40, w_inh=0.4),
        ),
        (
            "two_layer w_ei=40 w_inh=0.6",
            dict(inhibition="two_layer", w_exc_inh=40, w_inh=0.6),
        ),
        (
            "two_layer w_ei=40 w_inh=3.0",
            dict(inhibition="two_layer", w_exc_inh=40, w_inh=3.0),
        ),
    ]
    for name, kw in configs:
        s = summarize(imgs, **kw)
        print(
            f"{name:32s}  exc={s['exc_spikes/img']:7.1f}  active={s['exc_active/img']:5.1f}/100"
            f"  inh={s['inh_spikes/img']:7.1f}  top10={s['top10_share']:.2f}"
        )


if __name__ == "__main__":
    main()
