"""GPU-accelerated, batched spiking network (PyTorch port of `network.Network`).

This is a *from-scratch* reimplementation of the same model -- LIF neurons, plastic
input->exc synapses, lateral inhibition, adaptive-threshold homeostasis, pair-based
STDP -- written in PyTorch tensors so it runs batched on a CUDA GPU. We do NOT use
snnTorch/Brian2; the equations are the same ones in `lif.py` / `synapses.py` /
`stdp.py`, just vectorized over a batch axis and runnable on `cuda`.

Why this exists: the NumPy core trains one image at a time through a Python
timestep loop, so scaling to the 10-50x more presentations needed for clean
receptive fields is hours on CPU. The one axis left to parallelize is *across
images* (the per-image timestep recurrence is sequential). Batching that way means
**mini-batch STDP**: every image in a batch sees the same start-of-batch weights,
and their weight/threshold updates are accumulated and applied once per batch
instead of strictly image-by-image. That is a small, deliberate, well-documented
change to the learning dynamics (it is how Diehl & Cook-style nets are scaled on
GPUs) -- so accuracy must be re-validated against the NumPy baseline, not assumed.

`batch=1` reduces to the online rule *except* that weights are still frozen within
the single image (the NumPy path updates W every step); the no-train forward pass,
however, is identical to NumPy up to float precision -- that is what
`scripts/_check_torch_equiv.py` checks on CPU before any GPU training run.

Only the "lateral" inhibition mode is ported (it won the head-to-head vs the
explicit two_layer population; see CLAUDE.md). Readout (assign_labels / classify /
evaluate) mirrors `network.Network`.
"""

from __future__ import annotations

import numpy as np
import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class TorchSNN:
    def __init__(
        self,
        n_input: int,
        n_exc: int,
        dt: float = 1.0,
        # input gain (matches network._eff_max_rate)
        input_norm: float = 100.0,
        input_norm_power: float = 0.5,
        # input -> exc plastic synapses
        w_init_scale: float = 0.3,
        w_max: float = 1.0,
        tau_syn: float = 5.0,
        norm_target: float = 90.0,
        # lateral inhibition
        w_inh: float = 0.6,
        tau_inh: float = 8.0,
        # excitatory neuron excitability + homeostasis
        r_m: float = 6.0,
        theta_plus: float = 0.05,
        tau_theta: float = 1e4,
        # LIF constants
        tau_m: float = 20.0,
        v_rest: float = -65.0,
        v_reset: float = -65.0,
        v_thresh: float = -52.0,
        t_refrac: float = 2.0,
        # STDP
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        w_min: float = 0.0,
        # how per-batch weight updates are applied:
        #   "sequential" = apply each image's delta in turn with an L1-renorm
        #     between them (online-like; most faithful, default)
        #   "summed"     = sum all images' deltas, apply + renorm once (fastest,
        #     but large batches over-saturate before the single renorm)
        stdp_update: str = "sequential",
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
    ):
        self.n_input = n_input
        self.n_exc = n_exc
        self.dt = float(dt)
        self.input_norm = input_norm
        self.input_norm_power = input_norm_power
        self.w_max = w_max
        self.w_min = w_min
        self.norm_target = norm_target
        self.r_m = r_m
        self.theta_plus = theta_plus
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.t_refrac = t_refrac
        self.a_plus = a_plus
        self.a_minus = a_minus
        if stdp_update not in ("sequential", "summed"):
            raise ValueError(
                f"stdp_update must be 'sequential' or 'summed', got {stdp_update!r}"
            )
        self.stdp_update = stdp_update
        self.device = torch.device(device or default_device())
        self.dtype = dtype

        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)

        # decay factors (scalars)
        self.exc_decay = float(np.exp(-dt / tau_syn))
        self.inh_decay = float(np.exp(-dt / tau_inh))
        self.theta_decay = float(np.exp(-dt / tau_theta))
        self.decay_pre = float(np.exp(-dt / tau_pre))
        self.decay_post = float(np.exp(-dt / tau_post))

        f = dict(device=self.device, dtype=dtype)
        # Plastic feedforward weights, init uniform[0, w_init_scale) then L1-normalized.
        W = torch.rand((n_input, n_exc), generator=self.gen, **f) * w_init_scale
        self.W = W
        self._normalize()

        # Fixed lateral inhibition matrix: -w_inh off-diagonal, 0 on diagonal.
        eye = torch.eye(n_exc, **f)
        self.W_inh = -w_inh * (torch.ones((n_exc, n_exc), **f) - eye)

        # Long-term per-neuron threshold offset (homeostasis); persists across batches.
        self.theta = torch.zeros(n_exc, **f)

        self.neuron_labels: np.ndarray | None = None
        self.neuron_response: np.ndarray | None = None

    # -- weight maintenance --------------------------------------------------

    def _normalize(self):
        """L1-normalize each exc neuron's incoming weights to `norm_target`.

        Maskless on purpose: the boolean-mask form (W[:, nz] *= ...) forces a
        device->host SYNC every call to size the mask, which is murder in the
        per-image sequential loop (~thousands of calls/epoch, each stalling the GPU).
        clamp_min keeps it a single fused kernel; a zero column scales 0 * (target/eps)
        = 0, so it stays zero -- identical result, no sync.
        """
        self.W *= self.norm_target / self.W.sum(dim=0).clamp_min(1e-12)

    # -- input encoding (matches encoding.poisson_encode + network._eff_max_rate) --

    def encode(
        self, images: torch.Tensor, T: int, max_rate: float = 63.75
    ) -> torch.Tensor:
        """Poisson-encode a batch of images into spikes (B, T, n_input) as `dtype`.

        images: (B, n_input) float on `device`, in [0, 1] (raw pixel intensities).
        Per image we normalize by its own peak, then equalize total drive with the
        same gain law as the NumPy path (`input_norm_power`).
        """
        B = images.shape[0]
        peak = images.amax(dim=1, keepdim=True).clamp_min(1e-12)  # (B,1)
        flat = images / peak  # peak-normalized
        if self.input_norm > 0:
            nf = flat.sum(dim=1, keepdim=True)  # effective #on-pixels
            gain = (self.input_norm / (nf + 1e-12)) ** self.input_norm_power
        else:
            gain = torch.ones_like(peak)
        p = flat * (max_rate * gain) * self.dt / 1000.0  # (B, n_input) spike prob
        draws = torch.rand(
            (B, T, self.n_input),
            generator=self.gen,
            device=self.device,
            dtype=self.dtype,
        )
        return (draws < p.unsqueeze(1)).to(self.dtype)

    # -- core batched simulation ---------------------------------------------

    def _simulate(self, in_spikes: torch.Tensor, train: bool):
        """Run B independent images for T steps through the shared (frozen) weights.

        in_spikes: (B, T, n_input) float spike train.
        Returns exc spike `counts` (B, n_exc). If train, also returns the
        accumulated weight delta `dW` and per-image final `theta` (B, n_exc) for
        the caller to fold into the shared state. `dW` is (n_input, n_exc) summed
        over the batch in "summed" mode, or per-image (B, n_input, n_exc) in
        "sequential" mode (so each image's delta can be applied + renormed in turn).
        """
        B, T, _ = in_spikes.shape
        seq = train and self.stdp_update == "sequential"
        f = dict(device=self.device, dtype=self.dtype)
        v = torch.full((B, self.n_exc), self.v_rest, **f)
        refrac_until = torch.zeros((B, self.n_exc), **f)
        theta = self.theta.unsqueeze(0).expand(B, -1).clone()  # per-image local copy
        g_exc = torch.zeros((B, self.n_exc), **f)
        g_inh = torch.zeros((B, self.n_exc), **f)
        prev_exc = torch.zeros((B, self.n_exc), **f)
        counts = torch.zeros((B, self.n_exc), **f)

        if train:
            x_pre = torch.zeros((B, self.n_input), **f)
            x_post = torch.zeros((B, self.n_exc), **f)
            dW = (
                torch.zeros((B, self.n_input, self.n_exc), **f)
                if seq
                else torch.zeros((self.n_input, self.n_exc), **f)
            )

        t = 0.0
        for ti in range(T):
            s_in = in_spikes[:, ti, :]  # (B, n_input)
            # synaptic currents (decay then deposit)
            g_exc = g_exc * self.exc_decay + s_in @ self.W
            g_inh = (
                g_inh * self.inh_decay + prev_exc @ self.W_inh
            )  # 1-step delayed inhibition
            i_in = g_exc + g_inh

            # LIF update
            t += self.dt
            theta = theta * self.theta_decay
            active = t >= refrac_until
            dv = (self.dt / self.tau_m) * (-(v - self.v_rest) + self.r_m * i_in)
            v = torch.where(active, v + dv, torch.full_like(v, self.v_reset))
            spikes = (v >= (self.v_thresh + theta)) & active
            v = torch.where(spikes, torch.full_like(v, self.v_reset), v)
            refrac_until = torch.where(
                spikes, torch.full_like(refrac_until, t + self.t_refrac), refrac_until
            )
            sp = spikes.to(self.dtype)
            theta = theta + sp * self.theta_plus

            if train:
                # traces decay (pre-increment), then accumulate this step's STDP delta
                x_pre = x_pre * self.decay_pre
                x_post = x_post * self.decay_post
                # LTP: pre-before-post (x_pre x post);  LTD: post-before-pre (pre x x_post)
                if seq:
                    # keep deltas per image (b i j) so each can be applied separately
                    dW += self.a_plus * torch.einsum(
                        "bi,bj->bij", x_pre, sp
                    ) - self.a_minus * torch.einsum("bi,bj->bij", s_in, x_post)
                else:
                    # reduce over the batch immediately (cheaper, no per-image buffer)
                    dW += self.a_plus * (x_pre.transpose(0, 1) @ sp) - self.a_minus * (
                        s_in.transpose(0, 1) @ x_post
                    )
                x_pre = x_pre + s_in
                x_post = x_post + sp

            counts += sp
            prev_exc = sp

        if train:
            return counts, dW, theta
        return counts

    # -- training ------------------------------------------------------------

    def train(
        self,
        images: np.ndarray,
        epochs: int = 1,
        T: int = 350,
        batch_size: int = 32,
        max_rate: float = 63.75,
        progress: bool = False,
        rng: np.random.Generator | None = None,
    ):
        """Mini-batch unsupervised training. `images` is (N, H, W) or (N, n_input)."""
        rng = rng or np.random.default_rng(0)
        X = self._to_device(images)
        N = X.shape[0]
        for ep in range(epochs):
            order = rng.permutation(N)
            for bi in range(0, N, batch_size):
                idx = order[bi : bi + batch_size]
                batch = X[torch.as_tensor(idx, device=self.device)]
                in_spikes = self.encode(batch, T, max_rate=max_rate)
                counts, dW, theta = self._simulate(in_spikes, train=True)
                B = batch.shape[0]
                if self.stdp_update == "sequential":
                    # Apply each image's delta in turn, renormalizing between them.
                    # The batched forward froze W within the batch (the only residual
                    # approximation), but the WEIGHT UPDATES are now online-like: one
                    # image-delta, clip, L1-renorm, repeat -- so no intra-batch
                    # saturation. dW is (B, n_input, n_exc).
                    for b in range(B):
                        self.W += dW[b]
                        self.W.clamp_(self.w_min, self.w_max)
                        self._normalize()
                else:
                    # Summed: one big update per batch (fastest; large batches
                    # over-saturate before the single renorm -> drifts from online).
                    self.W += dW
                    self.W.clamp_(self.w_min, self.w_max)
                    self._normalize()
                # Homeostasis: shared threshold = MEAN of the batch's per-image
                # theta. This is stable at ANY batch size -- it applies the slow
                # decay once and adds the mean bump (a bounded AR(1)). Do NOT sum
                # the per-image increments: that multiplies the per-image decay
                # drift by B, and for B * (1-decay^T) > 1 (here B > ~29) theta
                # sign-flips and explodes to +/-inf, silencing neurons (the batch=128
                # collapse to below-chance). Mean keeps theta in the NumPy range.
                self.theta = theta.mean(dim=0)
                if progress:
                    print(f"  epoch {ep + 1}/{epochs}  image {min(bi + B, N)}/{N}")

    # -- forward (no learning) -----------------------------------------------

    def counts(
        self,
        images: np.ndarray,
        T: int = 350,
        batch_size: int = 256,
        max_rate: float = 63.75,
    ) -> np.ndarray:
        """Excitatory spike counts per image (N, n_exc) with plasticity off."""
        X = self._to_device(images)
        out = []
        for bi in range(0, X.shape[0], batch_size):
            batch = X[bi : bi + batch_size]
            in_spikes = self.encode(batch, T, max_rate=max_rate)
            out.append(self._simulate(in_spikes, train=False).cpu().numpy())
        return np.concatenate(out, axis=0)

    def counts_from_spikes(self, in_spikes: torch.Tensor) -> torch.Tensor:
        """Forward pass from a precomputed spike tensor (B, T, n_input). For tests."""
        return self._simulate(in_spikes, train=False)

    # -- readout (labels used only here, never during learning) --------------

    def assign_labels(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        T: int = 350,
        n_classes: int | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        labels = np.asarray(labels)
        n_classes = n_classes or int(labels.max()) + 1
        counts = self.counts(images, T=T, batch_size=batch_size)  # (N, n_exc)
        rate_sum = np.zeros((self.n_exc, n_classes))
        per_class = np.zeros(n_classes, dtype=np.int64)
        for c in range(n_classes):
            m = labels == c
            if m.any():
                rate_sum[:, c] = counts[m].sum(axis=0)
                per_class[c] = int(m.sum())
        per_class[per_class == 0] = 1
        avg = rate_sum / per_class[None, :]
        self.neuron_labels = avg.argmax(axis=1)
        self.neuron_response = avg
        return self.neuron_labels

    def predict(
        self, images: np.ndarray, T: int = 350, batch_size: int = 256
    ) -> np.ndarray:
        """Predicted class per image: the class whose neurons respond most strongly."""
        if self.neuron_labels is None:
            raise RuntimeError("call assign_labels() before predict()")
        counts = self.counts(images, T=T, batch_size=batch_size)  # (N, n_exc)
        n_classes = int(self.neuron_labels.max()) + 1
        scores = np.zeros((len(counts), n_classes))
        for c in range(n_classes):
            mask = self.neuron_labels == c
            if mask.any():
                scores[:, c] = counts[:, mask].mean(axis=1)
        return scores.argmax(axis=1)

    def evaluate(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        T: int = 350,
        batch_size: int = 256,
    ) -> float:
        labels = np.asarray(labels)
        preds = self.predict(images, T=T, batch_size=batch_size)
        return float((preds == labels).mean())

    def linear_probe(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        test_images: np.ndarray,
        test_labels: np.ndarray,
        T: int = 350,
        batch_size: int = 256,
        C: float = 1.0,
    ) -> float:
        """Linear-probe readout: logistic regression on frozen spike counts.

        The native readout (assign_labels/predict) labels whole neuron GROUPS by
        mean spike count, which badly undersells SELECTIVE neurons -- the clean,
        sparse-firing receptive fields that strong a_plus + theta_plus produce. A
        linear probe on the per-neuron count vector recovers the representation's
        true linearly-decodable accuracy (e.g. 0.48 native -> 0.90 probe on the
        a_plus=0.02 / theta_plus=2.0 net). The SNN stays backprop-free; only this
        readout is supervised (labels at eval only) -- the standard linear-probe
        diagnostic. Requires scikit-learn.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        Xtr = self.counts(train_images, T=T, batch_size=batch_size).astype(np.float64)
        Xte = self.counts(test_images, T=T, batch_size=batch_size).astype(np.float64)
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=C)
        clf.fit(scaler.transform(Xtr), np.asarray(train_labels))
        return float(clf.score(scaler.transform(Xte), np.asarray(test_labels)))

    def receptive_fields(self, image_shape: tuple[int, int]) -> np.ndarray:
        return self.W.transpose(0, 1).reshape(self.n_exc, *image_shape).cpu().numpy()

    # -- persistence ---------------------------------------------------------

    def save(self, path: str):
        labels = self.neuron_labels if self.neuron_labels is not None else np.array([])
        np.savez(
            path,
            n_input=self.n_input,
            n_exc=self.n_exc,
            W=self.W.cpu().numpy(),
            theta=self.theta.cpu().numpy(),
            neuron_labels=labels,
        )

    # -- helpers -------------------------------------------------------------

    def _to_device(self, images: np.ndarray) -> torch.Tensor:
        arr = np.asarray(images, dtype=np.float32)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return torch.as_tensor(arr, device=self.device, dtype=self.dtype)
