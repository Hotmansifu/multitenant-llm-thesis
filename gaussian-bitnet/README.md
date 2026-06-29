# gaussian-bitnet/ — RQ4

Gaussian-kernel attention on a BitNet (ternary-weight) model, its quality
evaluation, and the partial-homomorphic-encryption (PHE) inference experiments.

Motivation: ternary weights {-1, 0, +1} turn the linear layers into additions
and subtractions only, which is what an addition-only PHE scheme (Paillier)
supports. Gaussian-kernel attention replaces softmax because its exponent is
bounded, which is friendlier to encrypted / approximate evaluation. Attention
stays on the trusted client in plaintext, so the softmax exponential never runs
under encryption.

## What each script does

**Models**
- `model.py` — GPT with a Gaussian-kernel attention option
  (`use_gaussian_kernel`, `gaussian_sigma`); the Gaussian path uses eager
  attention instead of SDPA.
- `model_bitnet.py` — the ternary BitNet variant used by the PHE scripts.

**Quality evaluation**
- `experiment_gaussian_kernel.py` — the Gaussian-vs-softmax quality experiment:
  trains a softmax baseline and four Gaussian variants (sigma = sqrt(d), 1, 5,
  learnable) on the same data and compares validation loss / perplexity. This is
  the script behind the ~0.06% Gaussian-vs-softmax result. Writes
  `gaussian_experiment_results.json`, a convergence plot, and a text report.
- `bitnet_compound_extended.py` — extended compound test: BitNet-only vs
  BitNet+Gaussian, sweeping sigma = 4, 5, 6, at 2400 iterations.
- `compound_test_124m.py` — the compound-error experiment at 124M scale on the
  Chicago dataset, four configurations (softmax, Gaussian, BitNet,
  BitNet+Gaussian). Produces the reported ~0% interaction effect.

**Partial-PHE inference**
- `paillier_s3_demo.py` — **LoRA-only Paillier path.** Only the low-rank LoRA
  route is encrypted; the public base weights stay in plaintext. Measured
  overhead ~16,383x, far below encrypting the full hidden state.
- `addition_only_phe_demo.py` — addition-only PHE using the
  W = W1 - Wm1 decomposition of the ternary weights, with a trusted setup that
  is independent of the server. Online cost ~1.0x.
- `paillier_additive_setup_ckpt_v2.py` — **the additive-masking setup-cost
  measurement** on the real checkpoint. Measures the one-time Paillier setup
  (~160 min on the 30M model), reports the per-term cost spread before
  extrapolating (over 30 real neurons across 2 layers), and verifies
  correctness (error ~1.4e-14). Online inference is separate (~1.0x).

**Training**
- `train_otw_gaussian_big.py` — config for training the ~124M Gaussian-kernel
  model on the OpenWebText toy slice.

## Headline numbers

- Gaussian vs softmax: ~0.06% validation-loss difference.
- BitNet x Gaussian interaction effect: ~0%.
- LoRA-only Paillier path: ~16,383x (measured).
- Additive masking: ~160 min one-time setup, ~1.0x online, correctness
  ~1.4e-14.

## External dependencies

The production-scale inference benchmark uses Microsoft's
**BitNet b1.58-2B-4T** model run through **bitnet.cpp** (the I2_S 2-bit packed
format); neither is included here — see the root README for links. The smaller
demos use a locally trained checkpoint. `model_bitnet.py` can also pull GPT-2
weights via `transformers` (`from_pretrained`).

## Running

```
# additive-masking setup-cost measurement (optionally pass a checkpoint path)
python paillier_additive_setup_ckpt_v2.py
# LoRA-only Paillier overhead
python paillier_s3_demo.py
# quality / interaction
python compound_test_124m.py
```

Scripts insert their own directory on `sys.path` and import `model` /
`model_bitnet` from here, so run them from inside this folder. The PHE scripts
require `phe` (python-paillier); they expect a trained checkpoint.
