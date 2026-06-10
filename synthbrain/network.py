"""Unsupervised spiking network (Diehl & Cook 2015 style).

Architecture:

    input (Poisson)  --plastic STDP-->  excitatory LIF layer
                                          |  ^
                                          |  | lateral inhibition
                                          +--+ (winner-take-all)

  * Input neurons are Poisson generators produced by `encoding.poisson_encode`.
  * Each excitatory neuron integrates input through an all-to-all plastic synapse
    matrix learned with STDP.
  * Excitatory neurons inhibit one another (lateral inhibition), so that for a
    given input only a few neurons win and get to learn it -> the layer self-
    organizes into digit-selective cells.
  * Each excitatory neuron has an adaptive firing threshold (homeostasis) so no
    single neuron dominates every input.

Learning is fully unsupervised. Labels are used ONLY after training to read out
which digit each neuron came to represent (`assign_labels` / `classify`).
"""

from __future__ import annotations

import numpy as np

from .lif import LIFGroup
from .synapses import Synapses
from .stdp import STDP
from .encoding import poisson_encode


class Network:
    def __init__(
        self,
        n_input: int,
        n_exc: int,
        dt: float = 1.0,
        # input gain control: equalize each image's total drive (0 disables)
        input_norm: float = 100.0,
        # how hard to equalize: 1.0 = full per-image normalization, 0.0 = off.
        # <1 softens the boost so thin digits (e.g. '1') don't over-fire and
        # monopolize the excitatory layer.
        input_norm_power: float = 1.0,
        # input -> exc plastic synapses
        w_init_scale: float = 0.3,
        w_max: float = 1.0,
        tau_syn: float = 5.0,
        norm_target: float = 90.0,
        # inhibition: "lateral" = direct exc->exc negative coupling (default,
        # an approximation); "two_layer" = explicit inhibitory population wired
        # exc_i->inh_i (1:1) and inh_i->exc_{j!=i}, i.e. true Diehl & Cook 2015.
        inhibition: str = "lateral",
        w_inh: float = 0.6,  # inh->exc magnitude (both modes)
        tau_inh: float = 8.0,
        w_exc_inh: float = 40.0,  # exc_i->inh_i drive (two_layer); strong enough
        # that one exc spike makes its inh partner fire
        r_m_inh: float = 6.0,  # inhibitory-neuron excitability (two_layer)
        # excitatory neuron excitability + homeostasis
        r_m: float = 6.0,
        theta_plus: float = 0.05,
        tau_theta: float = 1e4,
        # STDP
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        rng: np.random.Generator | None = None,
    ):
        self.n_input = n_input
        self.n_exc = n_exc
        self.input_norm = input_norm
        self.input_norm_power = input_norm_power
        self.dt = dt
        self.norm_target = norm_target
        self.inhibition = inhibition
        self.rng = rng or np.random.default_rng()

        # Plastic feedforward synapses (input -> excitatory).
        self.exc_syn = Synapses.all_to_all(
            n_input,
            n_exc,
            w_low=0.0,
            w_high=w_init_scale,
            rng=self.rng,
            tau_syn=tau_syn,
            dt=dt,
        )
        self.exc_syn.normalize(norm_target)

        # Inhibition within the excitatory layer.
        if inhibition == "lateral":
            # Direct exc->exc negative coupling (winner-take-all approximation).
            self.inh_syn = Synapses.lateral_inhibition(
                n_exc, w_inh=w_inh, tau_syn=tau_inh, dt=dt
            )
        elif inhibition == "two_layer":
            # Explicit inhibitory population (one inh neuron per exc neuron).
            # exc_i -> inh_i, one-to-one fixed excitatory (diagonal weights).
            self.exc_to_inh_syn = Synapses(
                n_exc, n_exc, w=w_exc_inh * np.eye(n_exc), tau_syn=tau_syn, dt=dt
            )
            # inh_i -> exc_{j!=i}, fixed inhibitory (off-diagonal negative).
            self.inh_to_exc_syn = Synapses.lateral_inhibition(
                n_exc, w_inh=w_inh, tau_syn=tau_inh, dt=dt
            )
            # Inhibitory neurons: plain LIF, no threshold adaptation, fast relay.
            self.inh = LIFGroup(n_exc, dt=dt, r_m=r_m_inh, theta_plus=0.0, rng=self.rng)
        else:
            raise ValueError(
                f"inhibition must be 'lateral' or 'two_layer', got {inhibition!r}"
            )

        # Excitatory neurons with an adaptive threshold. r_m sets excitability:
        # the feedforward weight budget is kept modest (for STDP contrast), so a
        # higher membrane resistance is what lifts the drive above rheobase and
        # lets the layer fire from a cold (uniform-weight) start.
        self.exc = LIFGroup(
            n_exc,
            dt=dt,
            r_m=r_m,
            theta_plus=theta_plus,
            tau_theta=tau_theta,
            rng=self.rng,
        )

        # Learning rule on the feedforward weights.
        self.stdp = STDP(
            n_input,
            n_exc,
            a_plus=a_plus,
            a_minus=a_minus,
            w_min=0.0,
            w_max=w_max,
            dt=dt,
        )

        self.neuron_labels: np.ndarray | None = None
        self.neuron_response: np.ndarray | None = None  # set by assign_labels()

    # -- single-image dynamics ----------------------------------------------

    def _reset_between_images(self):
        """Clear fast state between presentations but keep the learned threshold.

        Membrane potential, refractory timers, synaptic currents and STDP traces
        are transient; the adaptive threshold `theta` is long-term homeostasis and
        must persist across images.
        """
        theta = self.exc.theta.copy()
        self.exc.reset_state()
        self.exc.theta[:] = theta
        self.exc_syn.reset_state()
        if self.inhibition == "two_layer":
            self.inh.reset_state()
            self.exc_to_inh_syn.reset_state()
            self.inh_to_exc_syn.reset_state()
        else:
            self.inh_syn.reset_state()
        self.stdp.reset_state()

    def _step_once(
        self, in_spikes_t: np.ndarray, prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance the network one timestep.

        `prev` is the previous step's inhibitory drive applied to the exc layer:
        prev_exc spikes for 'lateral', prev_inh spikes for 'two_layer'. Both modes
        therefore apply inhibition with a one-step delay. Returns
        (exc_spikes, new_prev) where new_prev feeds the next step.
        """
        in_current = self.exc_syn.step(in_spikes_t)
        if self.inhibition == "two_layer":
            inh_current = self.inh_to_exc_syn.step(prev)  # prev = prev_inh
            exc_spikes = self.exc.step(in_current + inh_current)
            inh_spikes = self.inh.step(self.exc_to_inh_syn.step(exc_spikes))
            return exc_spikes, inh_spikes
        inh_current = self.inh_syn.step(prev)  # prev = prev_exc
        exc_spikes = self.exc.step(in_current + inh_current)
        return exc_spikes, exc_spikes

    def _eff_max_rate(self, image: np.ndarray, max_rate: float) -> float:
        """Per-image input gain: scale the firing rate so every digit delivers a
        comparable *total* input drive, regardless of how many pixels are inked.

        Without this, dense digits (8, 0) drive far more current than thin ones (1)
        and monopolize the excitatory layer, leaving sparse digits with no neurons.
        `nf = sum(intensity)/max(intensity)` is the effective number of fully-on
        pixels; we rescale so `nf` matches the `input_norm` reference.

        `input_norm_power` controls how aggressively we equalize: 1.0 is full
        normalization, 0.0 disables it. Values <1 compress the gain spread, which
        stops very thin digits (e.g. '1') from being boosted so hard that they
        over-fire and capture a disproportionate share of the excitatory layer.
        """
        if self.input_norm <= 0:
            return max_rate
        nf = image.sum() / (image.max() + 1e-12)
        gain = (self.input_norm / (nf + 1e-12)) ** self.input_norm_power
        return max_rate * gain

    def present(
        self,
        image: np.ndarray,
        T: int = 350,
        max_rate: float = 63.75,
        train: bool = True,
    ) -> np.ndarray:
        """Run one image through the SNN for T timesteps.

        Args:
            image: Image (H, W) or flattened (n_input,), values in [0, 1]
                (re-normalised internally by its own max).
            T: Number of simulation timesteps (ms at dt=1).
            max_rate: Peak Poisson input rate in Hz (before per-image gain).
            train: If True, apply STDP and L1-renormalise the feedforward
                weights (mutates ``self.exc_syn.W``); if False, weights are
                left unchanged.

        Returns:
            Excitatory spike counts, shape (n_exc,), dtype int64.
        """
        max_rate = self._eff_max_rate(image, max_rate)
        input_spikes = poisson_encode(
            image, T, dt=self.dt, max_rate=max_rate, rng=self.rng
        )

        counts = np.zeros(self.n_exc, dtype=np.int64)
        prev = np.zeros(self.n_exc, dtype=bool)
        for ti in range(T):
            exc_spikes, prev = self._step_once(input_spikes[ti], prev)

            if train:
                self.stdp.step(self.exc_syn.W, input_spikes[ti], exc_spikes)

            counts += exc_spikes

        if train:
            self.exc_syn.normalize(self.norm_target)
        self._reset_between_images()
        return counts

    def simulate(self, image: np.ndarray, T: int = 350, max_rate: float = 63.75):
        """Present one image without learning, recording everything for visualization.

        Returns a dict with:
            input_spikes (T, n_input) bool   — Poisson input layer
            exc_spikes   (T, n_exc)   bool   — excitatory layer spikes
            v_trace      (T, n_exc)   float  — excitatory membrane potentials
            inh_spikes   (T, n_exc)   bool   — inhibitory layer (two_layer mode only)
        """
        max_rate = self._eff_max_rate(image, max_rate)
        input_spikes = poisson_encode(
            image, T, dt=self.dt, max_rate=max_rate, rng=self.rng
        )
        exc_spikes = np.zeros((T, self.n_exc), dtype=bool)
        v_trace = np.zeros((T, self.n_exc), dtype=np.float64)
        two_layer = self.inhibition == "two_layer"
        inh_spikes = np.zeros((T, self.n_exc), dtype=bool) if two_layer else None
        prev = np.zeros(self.n_exc, dtype=bool)
        for ti in range(T):
            s, prev = self._step_once(input_spikes[ti], prev)
            exc_spikes[ti] = s
            v_trace[ti] = self.exc.v
            if two_layer:
                inh_spikes[ti] = prev  # in two_layer, _step_once returns inh spikes
        self._reset_between_images()
        out = {
            "input_spikes": input_spikes,
            "exc_spikes": exc_spikes,
            "v_trace": v_trace,
        }
        if two_layer:
            out["inh_spikes"] = inh_spikes
        return out

    def run_record(
        self, image: np.ndarray, T: int = 350, max_rate: float = 63.75
    ) -> np.ndarray:
        """Present one image without learning; return just the (T, n_exc) spike trace."""
        return self.simulate(image, T=T, max_rate=max_rate)["exc_spikes"]

    def train(
        self,
        images: np.ndarray,
        epochs: int = 1,
        T: int = 350,
        max_rate: float = 63.75,
        progress: bool = False,
    ):
        """Present every image `epochs` times with plasticity ON (mutates W).

        Args:
            images: (N, H, W) or (N, n_input), pixel values in [0, 1].
            epochs: How many times to sweep the whole set (shuffled each time).
        """
        for ep in range(epochs):
            order = self.rng.permutation(len(images))
            for i, idx in enumerate(order):
                self.present(images[idx], T=T, max_rate=max_rate, train=True)
                if progress and (i + 1) % max(1, len(order) // 10) == 0:
                    print(f"  epoch {ep + 1}/{epochs}  image {i + 1}/{len(order)}")

    # -- readout (uses labels only here, never during learning) --------------

    def assign_labels(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        T: int = 350,
        n_classes: int | None = None,
    ):
        """Label each neuron with the class it fires most for (eval only).

        Runs every image with plasticity OFF (weights unchanged). Labels are
        used here purely to read out which digit each neuron came to represent;
        STDP training itself never sees them.

        Args:
            images: (N, H, W) or (N, n_input), pixel values in [0, 1].
            labels: (N,) integer class ids.

        Returns:
            Per-neuron class assignment, shape (n_exc,).
        """
        labels = np.asarray(labels)
        n_classes = n_classes or int(labels.max()) + 1
        rate_sum = np.zeros((self.n_exc, n_classes), dtype=np.float64)
        per_class = np.zeros(n_classes, dtype=np.int64)
        for img, lab in zip(images, labels):
            rate_sum[:, lab] += self.present(img, T=T, train=False)
            per_class[lab] += 1
        per_class[per_class == 0] = 1
        avg_rate = rate_sum / per_class[None, :]
        self.neuron_labels = avg_rate.argmax(axis=1)
        self.neuron_response = avg_rate  # (n_exc, n_classes) mean spikes/image
        return self.neuron_labels

    def classify(self, image: np.ndarray, T: int = 350) -> int:
        """Predict a label by which class's neurons respond most strongly.

        Does not mutate weights. Requires ``assign_labels`` to have been called
        first; raises RuntimeError otherwise. Returns the predicted class id.
        """
        if self.neuron_labels is None:
            raise RuntimeError("call assign_labels() before classify()")
        counts = self.present(image, T=T, train=False)
        n_classes = int(self.neuron_labels.max()) + 1
        scores = np.zeros(n_classes)
        for c in range(n_classes):
            mask = self.neuron_labels == c
            if mask.any():
                scores[c] = counts[mask].mean()
        return int(scores.argmax())

    def evaluate(self, images: np.ndarray, labels: np.ndarray, T: int = 350) -> float:
        """Classification accuracy over a labelled set."""
        labels = np.asarray(labels)
        correct = sum(
            self.classify(img, T=T) == lab for img, lab in zip(images, labels)
        )
        return correct / len(labels)

    def receptive_fields(self, image_shape: tuple[int, int]) -> np.ndarray:
        """Return learned input weights reshaped to (n_exc, H, W) for visualization."""
        return self.exc_syn.W.T.reshape(self.n_exc, *image_shape)

    # -- persistence ---------------------------------------------------------

    def save(self, path: str):
        """Save the learned state (weights, thresholds, neuron labels) to .npz."""
        labels = self.neuron_labels if self.neuron_labels is not None else np.array([])
        np.savez(
            path,
            n_input=self.n_input,
            n_exc=self.n_exc,
            W=self.exc_syn.W,
            theta=self.exc.theta,
            neuron_labels=labels,
        )

    @classmethod
    def load(cls, path: str, **kwargs) -> "Network":
        """Reconstruct a network and restore its learned state from a .npz file.

        Hyperparameters not stored in the file (r_m, w_inh, ...) come from kwargs,
        so pass the same ones used at training time for faithful dynamics.
        """
        d = np.load(path)
        net = cls(int(d["n_input"]), int(d["n_exc"]), **kwargs)
        net.exc_syn.W[:] = d["W"]
        net.exc.theta[:] = d["theta"]
        labels = d["neuron_labels"]
        net.neuron_labels = labels if labels.size else None
        return net
