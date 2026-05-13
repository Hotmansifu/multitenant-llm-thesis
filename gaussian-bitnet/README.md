# Gaussian Kernel & BitNet Experiments (RQ4)

Corresponds to **§3.3, §4.4** of the thesis.

## Research Question

- **RQ4:** Can inference of a one-bit model be secured using homomorphic encryption,
  and what is the performance penalty?

## Overview

Two model modifications are validated as prerequisites for PHE-compatible inference:

1. **Gaussian kernel attention** — replaces softmax to avoid polynomial approximation of exp() under encryption
2. **BitNet ternary weights** {-1, 0, 1} — eliminates floating-point multiplications, making Paillier encryption compatible

## Files

| File | Description |
|---|---|
| `model.py` | GPT model with Gaussian kernel attention support (`use_gaussian_kernel=True`) |
| `experiment_gaussian_kernel.py` | Gaussian vs softmax baseline comparison (RQ4 quality validation) |
| `bitnet_compound_extended.py` | Sigma sweep (σ=4, 5, 6) on 30M model — confirmed σ=4 as optimal |
| `compound_test_124m.py` | Four-combination test at 124M parameters on Chicago dataset |
| `train_otw_gaussian_big.py` | Training config for 124M Gaussian model on OpenWebText |
| `paillier_s3_demo.py` | LoRA adapter path Paillier encryption benchmark (16,383× overhead) |

## Results

| Experiment | Result |
|---|---|
| Gaussian vs softmax (124M, OpenWebText) | Val loss 4.806 vs 4.803 — 0.06% difference |
| BitNet+Gaussian compound (30M, 2400 iters) | 0% interaction effect at σ=4 |
| BitNet+Gaussian compound (124M, 4800 iters) | 0% interaction effect |
| LoRA adapter encryption (64 elements, 2048-bit) | 16,383× overhead vs monolithic baseline |

## Running the LoRA Encryption Benchmark

```bash
cd /home/coder/project/nanoGPT
PYTHONNOUSERSITE=1 PYTHONPATH=. python paillier_s3_demo.py
```

## Notes

- All training used σ=4 (`gaussian_sigma2=16.0`) confirmed optimal by `bitnet_compound_extended.py`
- Timing results vary with machine load
- LoRA rank r=8 used throughout (T=8, r=8 → 64 elements encrypted per layer)