"""Real-time spike-raster animation of the trained SNN over a sequence of digits.

Several MNIST digits are presented one after another. Live panels:
  * the current input image,
  * a fading "retina" of its Poisson input spikes (28x28),
  * the excitatory spike raster across the whole sequence, each neuron coloured by
    the digit it learned, with per-digit bands, a moving time cursor and a running
    unsupervised prediction.

You watch a different colour-cluster light up for each digit -> the learned
selectivity, on display. Trains a small network on first run and caches it to
outputs/trained_net.npz so re-runs are fast. Writes:
  * outputs/spike_animation.gif
  * outputs/spike_animation_final.png   (last frame, for a quick look)

Run:  KMP_DUPLICATE_LIB_OK=TRUE python scripts/animate_network.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthbrain.network import Network
from synthbrain import encoding

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "trained_net.npz")

CLASSES = [0, 1, 2, 3]
N_EXC = 100
T = 160             # ms per digit
SEQ_LEN = 6         # how many digits to show in sequence
TARGET_FRAMES = 200 # cap total frames (GIF size / render time)
FADE = 0.55         # retina afterglow per frame
NET_KW = dict(r_m=6.0, w_inh=0.6, norm_target=90.0)


def get_net_and_data(rng):
    """Return (net, test_images, test_labels, image_shape)."""
    try:
        imgs, lbls = encoding.load_mnist()
        sel = np.concatenate([np.where(lbls == c)[0][:60] for c in CLASSES])
        rng.shuffle(sel)
        imgs, lbls = imgs[sel], lbls[sel]
    except Exception as e:
        print(f"[anim] MNIST unavailable ({e.__class__.__name__}); using synthetic digits.")
        imgs, lbls = encoding.synthetic_digits(n_per_class=40, classes=CLASSES, rng=rng)

    n_train = int(0.75 * len(imgs))
    tr_x, tr_y = imgs[:n_train], lbls[:n_train]
    te_x, te_y = imgs[n_train:], lbls[n_train:]
    shape = imgs.shape[1:]

    if os.path.exists(CACHE):
        print("[anim] loading cached network ->", CACHE)
        net = Network.load(CACHE, rng=rng, **NET_KW)
        if net.neuron_labels is None:
            net.assign_labels(tr_x, tr_y, T=150, n_classes=len(CLASSES))
    else:
        print("[anim] training network (first run; will be cached)...")
        net = Network(int(np.prod(shape)), N_EXC, rng=rng, **NET_KW)
        net.train(tr_x, epochs=5, T=150, progress=True)
        net.assign_labels(tr_x, tr_y, T=150, n_classes=len(CLASSES))
        net.save(CACHE)
        print("[anim] cached ->", CACHE)
    return net, te_x, te_y, shape


def build_sequence(net, te_x, te_y, seq_len):
    """Pick a varied, lively sequence: round-robin over classes, liveliest first."""
    spikes = np.array([net.run_record(img, T=T).sum() for img in te_x])
    by_class = {c: [i for i in np.argsort(-spikes) if te_y[i] == c] for c in CLASSES}
    seq, rank = [], 0
    while len(seq) < seq_len and any(len(v) > rank for v in by_class.values()):
        for c in CLASSES:
            if len(seq) >= seq_len:
                break
            if len(by_class[c]) > rank:
                seq.append(by_class[c][rank])
        rank += 1
    return seq


def main():
    rng = np.random.default_rng(0)
    net, te_x, te_y, shape = get_net_and_data(rng)
    cmap = plt.cm.tab10
    labels = net.neuron_labels

    # Pin the encoding RNG so the same sequence renders whether we trained or loaded.
    net.rng = np.random.default_rng(7)
    seq = build_sequence(net, te_x, te_y, SEQ_LEN)
    N = len(seq)

    # Simulate each digit; stitch events onto a global timeline.
    segments, ev_t, ev_n = [], [], []
    for d, idx in enumerate(seq):
        image, label = te_x[idx], int(te_y[idx])
        rec = net.simulate(image, T=T)
        inp, exc = rec["input_spikes"], rec["exc_spikes"]
        ts, ns = np.where(exc)
        ev_t.append(ts + d * T)
        ev_n.append(ns)
        cum = np.cumsum(exc, axis=0)
        pred = np.full(T, -1)
        for t in range(T):
            if cum[t].sum() > 0:
                scores = [cum[t][labels == c].mean() if (labels == c).any() else -1 for c in CLASSES]
                pred[t] = CLASSES[int(np.argmax(scores))]
        segments.append({"image": image, "label": label, "inp": inp, "pred": pred})
        print(f"[anim] digit {d + 1}/{N}: idx={idx} true={label} exc_spikes={int(exc.sum())}")

    ts_g = np.concatenate(ev_t) if ev_t else np.array([], int)
    ns_g = np.concatenate(ev_n) if ev_n else np.array([], int)
    ev_colors = cmap(np.array(labels)[ns_g] % 10) if (labels is not None and ns_g.size) else None
    total_T = N * T
    stride = max(1, round(total_T / TARGET_FRAMES))

    # ---- figure ----
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 3.0], height_ratios=[1, 1],
                          hspace=0.35, wspace=0.22)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_ret = fig.add_subplot(gs[1, 0])
    ax_ras = fig.add_subplot(gs[:, 1])

    im_img = ax_img.imshow(segments[0]["image"], cmap="gray")
    ax_img.set_title(f"input digit (true = {segments[0]['label']})", fontsize=10)
    ax_img.axis("off")

    H, W = shape
    retina = np.zeros((H, W))
    im_ret = ax_ret.imshow(retina, cmap="inferno", vmin=0, vmax=1)
    ax_ret.set_title("input spikes (retina)", fontsize=10)
    ax_ret.axis("off")

    scat = ax_ras.scatter([], [], s=9)
    cursor = ax_ras.axvline(0, color="white", lw=1.2, alpha=0.85)
    ax_ras.set_xlim(0, total_T)
    ax_ras.set_ylim(-1, net.n_exc)
    ax_ras.set_xlabel("time (ms)")
    ax_ras.set_ylabel("excitatory neuron")
    ax_ras.set_facecolor("#0b1021")

    # Per-digit bands + true-label markers.
    for d in range(N):
        if d % 2 == 1:
            ax_ras.axvspan(d * T, (d + 1) * T, color="white", alpha=0.04)
        if d > 0:
            ax_ras.axvline(d * T, color="white", lw=0.8, alpha=0.35)
        ax_ras.text((d + 0.5) * T, net.n_exc - 2, str(segments[d]["label"]),
                    color="white", ha="center", va="top", fontsize=9, alpha=0.65)
    ras_title = ax_ras.set_title("excitatory raster", fontsize=11)

    if labels is not None:
        present = sorted(set(int(labels[n]) for n in ns_g)) if ns_g.size else CLASSES
        handles = [Line2D([0], [0], marker="o", ls="", mfc=cmap(c % 10), mec="none",
                          label=f"digit {c}") for c in present]
        ax_ras.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.3)

    state = {"seg": -1}

    def update(t_global):
        nonlocal retina
        seg = min(t_global // T, N - 1)
        local = t_global - seg * T
        if seg != state["seg"]:           # new digit: switch image, clear retina
            state["seg"] = seg
            retina[:] = 0.0
            im_img.set_data(segments[seg]["image"])
            ax_img.set_title(f"input digit (true = {segments[seg]['label']})", fontsize=10)

        retina *= FADE
        lo = max(0, local - stride)
        fired = segments[seg]["inp"][lo:local + 1].any(axis=0).reshape(H, W)
        retina[fired] = 1.0
        im_ret.set_data(retina)

        mask = ts_g <= t_global
        scat.set_offsets(np.c_[ts_g[mask], ns_g[mask]] if mask.any() else np.empty((0, 2)))
        if ev_colors is not None and mask.any():
            scat.set_color(ev_colors[mask])
        cursor.set_xdata([t_global, t_global])

        p = segments[seg]["pred"][local]
        verdict = "?" if p < 0 else f"{p} " + ("(correct)" if p == segments[seg]["label"] else "(wrong)")
        ras_title.set_text(
            f"excitatory raster   t={t_global:4d} ms   digit {seg + 1}/{N}   prediction: {verdict}")
        return im_img, im_ret, scat, cursor, ras_title

    frames = list(range(0, total_T, stride))
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

    gif_path = os.path.join(OUT, "spike_animation.gif")
    anim.save(gif_path, writer=PillowWriter(fps=20))
    update(total_T - 1)
    fig.savefig(os.path.join(OUT, "spike_animation_final.png"), dpi=110)
    plt.close(fig)
    print(f"[anim] wrote {gif_path} ({N} digits) and spike_animation_final.png")


if __name__ == "__main__":
    main()
