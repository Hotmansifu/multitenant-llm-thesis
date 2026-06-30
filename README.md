# Scalable and Secure Multi-Tenant LLM Deployment

Code and data for the bachelor thesis *Scalable and Secure Multi-Tenant LLM
Deployment: Memory Efficiency, Encryption, and Quantization Approaches*
(L. Andghuladze, Constructor University, 2026).

The thesis investigates three approaches to hosting many fine-tuned LLM tenants
on shared, untrusted infrastructure: memory-efficient fine-tuning, a
split-process trusted/untrusted architecture, and partial homomorphic
encryption (PHE) for inference protection.

## Repository layout

```
memory-efficiency/   RQ1 / RQ2 — parameter-efficient fine-tuning and memory cost
gaussian-bitnet/     RQ4 — Gaussian-kernel attention, BitNet ternary weights, PHE
split-process/       RQ3 — split trusted/untrusted execution (D1/D2/D3)
data_submission/     datasets + measured result files (see data_submission/manifest.md)
```

Each code folder has its own README describing every script and the headline
numbers it produces.

## Research questions

- **RQ1 / RQ2 — `memory-efficiency/`** — How do LoRA, layer freezing, and full
  fine-tuning compare in per-tenant and total deployment memory, and does the
  benefit hold as model size grows? LoRA stores a 0.78 MB adapter (109x smaller
  per tenant than layer freezing, 242x than full fine-tuning).
- **RQ3 — `split-process/`** — What is the overhead of splitting computation
  across a trusted client and an untrusted server? Measured at ~45.8% (shared
  memory) and ~491.1% (TCP), and ~56,000x when the full hidden state is sent
  under Paillier encryption.
- **RQ4 — `gaussian-bitnet/`** — Can ternary (BitNet) weights plus Gaussian
  attention make additive PHE viable? Quality impact is ~0.06% (Gaussian vs
  softmax) with ~0% compound interaction; encrypting only the LoRA path costs
  ~16,383x, and a PHE-assisted additive-masking variant moves Paillier work into
  a ~160 min setup with ~1.0x online overhead.

## Requirements

Python 3.10+ with `torch`, `numpy`, `tiktoken`, `transformers`, and `phe`
(python-paillier, for the PHE scripts). Each script inserts its own folder on
`sys.path` and imports `model` / `model_bitnet` locally, so run scripts from
inside their folder.

## Reproducing

1. **Data** — the tokenized datasets are in `data_submission/data/`
   (Chicago Q/A and an OpenWebText toy slice). The full OpenWebText corpus and
   the trained checkpoints are not included due to size; see
   `data_submission/manifest.md` for what is included and what is referenced.
2. **Train a checkpoint** — see `gaussian-bitnet/README.md` for the Gaussian
   (`train.py train_otw_gaussian_big.py`) and BitNet-Gaussian
   (`train_bitnet_gaussian_demo.py`) training commands.
3. **Run an experiment** — each folder's README lists the exact commands; the
   result files in `data_submission/` show the expected output formats.

## Attribution

This work builds on the master thesis of L. Hoxha (*In-Memory Fine-Tuning of
LLMs via Encrypted Parameter Deltas*, Constructor University, 2025); files that
originate there retain their original headers. The Gaussian-kernel attention,
the BitNet / ternary-weight experiments, the split-process architecture, and the
PHE / additive-masking work are contributions of this thesis.

`gaussian-bitnet/train.py` and `gaussian-bitnet/configurator.py` are from
nanoGPT (A. Karpathy, MIT License, https://github.com/karpathy/nanoGPT), with a
Gaussian-kernel attention option added for this thesis.

The production-scale inference benchmark uses Microsoft's **BitNet b1.58-2B-4T**
(https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) run through **bitnet.cpp**
(https://github.com/microsoft/BitNet); neither is redistributed here.
