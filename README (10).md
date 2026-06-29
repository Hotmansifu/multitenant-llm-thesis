# Scalable and Secure Multi-Tenant LLM Deployment

Code accompanying the bachelor thesis *Scalable and Secure Multi-Tenant LLM
Deployment* (L. Andghuladze, Constructor University, 2026, supervised by
Prof. A. Tormasov).

This repository contains the experiments behind the thesis's four research
questions: memory-efficient multi-tenant fine-tuning, secure split-process
inference, and privacy-preserving inference using partial homomorphic
encryption (PHE) on a ternary (BitNet) model with Gaussian-kernel attention.

## Relationship to prior work

This thesis **continues the master thesis of L. Hoxha**, *In-Memory
Fine-Tuning of LLMs via Encrypted Parameter Deltas* (Constructor University,
2025). The memory-efficiency code and the encrypted-delta infrastructure in
`memory-efficiency/` build directly on her codebase; files that originate from
her project retain her authorship headers.

The contributions of *this* thesis are:

- the multi-tenant LoRA memory evaluation (per-tenant and total-deployment
  scaling across 30M–774M models),
- the split-process secure-inference architecture and its three transport
  variants (`split-process/`),
- the Gaussian-kernel + BitNet (ternary) model and its quality evaluation
  (`gaussian-bitnet/`),
- the partial-PHE inference experiments, including the LoRA-only Paillier path
  and the additive-masking variant (`gaussian-bitnet/`).

## Repository layout

| Folder | Thesis RQ | Contents |
|---|---|---|
| `memory-efficiency/` | RQ1, RQ2 | Full / layer-freezing / LoRA fine-tuning, delta-size and memory measurements, catastrophic-forgetting check. Builds on Hoxha (2025). |
| `gaussian-bitnet/` | RQ4 | Gaussian-kernel attention, BitNet ternary model, quality eval, and the partial-PHE inference experiments (LoRA-only Paillier, additive masking). |
| `split-process/` | RQ3 | Two-process split inference with three transports: shared memory (D1), TCP sockets (D2), and Paillier PHE (D3). |

Each folder has its own README describing the individual scripts and how to run
them.

## Key results (as reported in the thesis)

These are the headline numbers each folder produces. "Measured" means recorded
directly from a run; "extrapolated" means computed from measured per-operation
rates and real model statistics (the thesis states this explicitly for those
figures).

**Memory efficiency (RQ1 / RQ2)**
- LoRA adapter size: 0.78 MB per tenant at 30M, 16.25 MB at 774M.
- Total-deployment (100 tenants): LoRA is ~44.7× smaller than layer freezing
  and ~98.5× smaller than full fine-tuning at 30M; at 774M the advantage over
  layer freezing narrows to ~17× (illustrating the expected narrowing in RQ2).
- Per-tenant (30M): LoRA uses ~109× less storage than layer freezing and ~242×
  less than full fine-tuning.
- Full-fine-tuning delta at 774M: ~338.1 GB for 100 tenants
  (architecture-determined).
- Catastrophic-forgetting check: the LoRA model did not degrade on the
  general-knowledge probe relative to base (no forgetting observed).

**Gaussian / BitNet quality (RQ4)**
- Gaussian-kernel attention vs softmax: ~0.06% validation-loss difference.
- No measurable interaction between BitNet ternary weights and the Gaussian
  kernel (≈0%).

**Secure split-process inference (RQ3)**
- D1 (shared memory): ~45.8% ± 2.6% overhead vs plaintext baseline.
- D2 (TCP sockets): ~491.1% ± 11.5% overhead.
- D3 (Paillier PHE on all transfers): ~56,000× overhead (extrapolated via the
  thesis's per-element rate model).
- LoRA-only Paillier path (encrypt only the low-rank route): ~16,383×
  (measured) — far cheaper than encrypting the full hidden state.

**Additive-masking partial-PHE (RQ4)**
- One-time setup: ~160 minutes on the 30M BitNet model.
- Online inference overhead: ~1.0× (plaintext additions only).
- Correctness verified to ~1.4×10⁻¹⁴.

See each folder's README for which script prints which number.

## External dependencies (not included in this repository)

These are large, third-party, and obtainable from their original sources. They
are **not** redistributed here.

- **BitNet b1.58-2B-4T model** — Microsoft.
  https://huggingface.co/microsoft/BitNet-b1.58-2B-4T
- **bitnet.cpp** (inference engine for the model above) — Microsoft.
  https://github.com/microsoft/BitNet
- **nanoGPT** (training scaffold the `gaussian-bitnet/` model is based on) —
  A. Karpathy. https://github.com/karpathy/nanoGPT
- **Datasets**: the Chicago employee-salary Q/A set and the filtered
  OpenWebText slice are produced by the data-preparation scripts under
  `memory-efficiency/`; the raw sources are public.
- **Checkpoints** (`.pt` / `ckpt.pt`) are produced by the training scripts and
  are not stored here due to size.

## Requirements

Python 3.11 is recommended (3.8 also works for the `memory-efficiency` scripts).
Core packages:

```
torch
numpy
tiktoken
phe            # python-paillier, for the PHE experiments
transformers   # optional, for GPT-2 from_pretrained in the model files
peft           # for LoRA
psutil         # for memory measurements
matplotlib     # for plots
```

Install with `pip install torch numpy tiktoken phe peft psutil matplotlib`.
The split-process and BitNet experiments additionally require a trained
checkpoint and (for the production-scale benchmark) the external BitNet model
and bitnet.cpp listed above.

## Reproducing

1. Prepare a checkpoint with the relevant training/config script in the target
   folder (or obtain the external BitNet model for the production benchmark).
2. Run the experiment script; each prints its result to stdout and, where
   applicable, writes a `*_results.json`.
3. Folder READMEs list the exact commands and expected outputs.
