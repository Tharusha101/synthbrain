"""No-training diagnostic: how input_norm_power equalizes per-digit input drive.

Reproduces Network._eff_max_rate over real MNIST and reports, per digit, the
per-pixel effective firing rate and the (approx) total input spike budget, for a
few power settings. Lower max/min spread => fairer drive across digits, so no
single digit (notably the thin '1') monopolizes the excitatory layer.
"""

import os
import sys


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain import encoding

MAX_RATE = 63.75  # Network.present default
INPUT_NORM = 100.0  # Network default
N_PER_CLASS = 100


def stats(imgs, lbls, power):
    rows = []
    for c in range(10):
        sel = imgs[lbls == c][:N_PER_CLASS]
        s = sel.sum(axis=(1, 2))
        nf = s / (sel.max(axis=(1, 2)) + 1e-12)
        eff = MAX_RATE * (INPUT_NORM / (nf + 1e-12)) ** power  # per-pixel Hz
        total = eff * s  # ~ total input spikes (a.u.)
        rows.append((c, eff.mean(), total.mean()))
    return rows


def main():
    imgs, lbls = encoding.load_mnist()
    for power in (1.0, 0.5, 0.0):
        rows = stats(imgs, lbls, power)
        effs = [r[1] for r in rows]
        tots = [r[2] for r in rows]
        print(
            f"\npower={power}:  eff-rate spread max/min={max(effs)/min(effs):.2f}   "
            f"total-drive spread max/min={max(tots)/min(tots):.2f}"
        )
        for c, e, t in rows:
            print(f"  digit {c}: eff_rate={e:6.1f} Hz   total_drive={t:9.0f}")


if __name__ == "__main__":
    main()
