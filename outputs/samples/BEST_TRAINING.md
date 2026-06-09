# Best training — synthbrain SNN (final)

Unsupervised, **no-backprop** spiking neural network that learns clean, human-readable
MNIST digit templates via STDP, and classifies at **~91%** with a linear-probe readout.

## Saved canonical model (`outputs/samples/gpu_best_net.npz`)
Produced by `scripts/train_gpu.py` with the baked-in defaults (seed 0, MNIST split
3375 train / 1125 test, 8 epochs, ~20.6 min on the RTX 4060):
- **linear-probe accuracy = 0.911**  (native mean-count readout = 0.595; chance = 0.10)
- receptive fields: `outputs/samples/gpu_best_receptive_fields.png` (nearly every
  one of 800 neurons is a clean digit; ~zero dead).
- The native per-digit recalls in the train log look weak for 4/5/8/9 -- that is the
  native readout underselling selective neurons, NOT the representation; the linear
  probe (and the confusion matrix) show the true per-digit picture.

## Multi-seed robustness (`scripts/finalize_eval.py`, 5 seeds)
The 0.91 is not a lucky seed -- retraining the recipe over 5 RNG seeds (varying
weight init, Poisson encoding, batch order; fixed 150/class test set):

| readout | mean +/- std |
|---------|--------------|
| **linear probe** | **0.900 +/- 0.007** |
| native mean-count | 0.602 +/- 0.013 |

Confusion matrix (`confusion_matrix.png`, linear probe): strong diagonal, 0/1/6
near-perfect; the residual errors are the classic shape confusions -- 9->4, 9->7,
3->8, 3->5, 5->3, 8->3. The probe recovers the digits the native readout drops
(e.g. digit 4: native recall ~0.38 -> probe ~0.93).

## Winning configuration (`best_th200_n800`)
Baked in as the defaults of `scripts/train_gpu.py`.

| knob | value | role |
|------|-------|------|
| n_input | 784 | 28×28 Poisson-encoded pixels |
| n_exc | **800** | excitatory LIF layer (sweet spot; 1600 was under-trained) |
| a_plus | **0.02** | LTP strength — THE template-cleanliness lever |
| a_minus | 0.012 | LTD strength |
| theta_plus | **2.0** | adaptive-threshold homeostasis — kills the dead-neuron side-effect (saturates here) |
| norm_target | 90 | L1 weight budget per neuron |
| w_inh | 0.6 | lateral inhibition (won vs the two_layer population) |
| r_m | 6.0 | membrane excitability |
| input_norm_power | 0.5 | per-image input-gain equalization |
| T | 350 ms | presentation window (dt = 1 ms) |
| inhibition | lateral | winner-take-all approximation |
| stdp_update | sequential | online-like mini-batch updates |
| tau_m / v_rest / v_thresh / v_reset / t_refrac | 20 / -65 / -52 / -65 / 2 | standard LIF |

**Training:** 450 imgs/class × 10 = 4500 train images, 8 epochs = **36,000 presentations**,
batch 128, ~27 min on an RTX 4060 Laptop (8 GB). Fully unsupervised — labels are used
ONLY at readout.

## Results (from `readout_results.json`, fixed 150/class test set)

| net (n_exc=800) | native readout | **linear probe** | tmpl_match | dead |
|-----------------|---------------|------------------|-----------|------|
| noisy baseline (a_plus=0.01, θ=0.05) | 0.709 | 0.875 | −0.12 | 0.00 |
| clean θ=0.4 | 0.481 | 0.846 | 0.43 | 0.21 |
| **★ clean θ=2.0 (this model)** | 0.615 | **0.905** | **0.894** | **0.000** |
| clean θ=2.0, n_exc=1600 | 0.613 | 0.872 | 0.55 | 0.13 |

- **tmpl_match** = mean Pearson corr of each receptive field with its digit's class-mean
  image (baseline was −0.13 → 0.89 = genuinely digit-like).
- **dead** = fraction of neurons that never fire (0.000 = full capacity used).
- Chance = 0.10. The cleanest net is also the most accurate under a fair readout.

## Key finding
The clean-vs-accurate "trade-off" was an artifact of the readout: the native
mean-spike-count readout undersells selective neurons. A linear probe (logistic
regression on frozen spike counts; the SNN itself stays backprop-free) recovers the
representation's true accuracy — **0.905**, the best of every configuration tested.

## Saved artifacts
- `outputs/gpu_net.npz` — trained weights / thresholds / labels (regenerate via `train_gpu.py`)
- `outputs/gpu_receptive_fields.png` / `outputs/samples/sweep_best_a020_th200_clean.png` — the receptive-field grid
- `outputs/samples/readout_results.json`, `sweep_wave{1..4}_results.json` — all sweep + probe metrics
- `outputs/samples/sweep_n1600_undertrained.png` — the under-trained 1600-neuron contrast

## Reproduce
```
set KMP_DUPLICATE_LIB_OK=TRUE
C:\Users\User\anaconda3\python.exe scripts/train_gpu.py
```
Prints native + linear-probe accuracy and a per-digit coverage/recall report; writes
`gpu_net.npz` + `gpu_receptive_fields.png`.
