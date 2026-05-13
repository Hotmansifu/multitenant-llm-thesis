# Memory Efficiency Experiments (RQ1 & RQ2)

Corresponds to **§3.1, §4.1, §4.2** of the thesis.

## Research Questions

- **RQ1:** Which fine-tuning approach consumes the least memory for multi-tenant inference at 30M parameters?
- **RQ2:** Is the memory efficiency advantage preserved at 774M parameters?

## Environment

- Ubuntu 22.04 (Linux 5.15.0)
- 32-core Intel Haswell processor
- 314 GB RAM
- Python 3.8, PyTorch 2.4.1
- Virtualized cloud instance

## Files

| File | Description |
|---|---|
| `finetune_encrypt_memmap.py` | Full fine-tuning and layer freezing (75%) training script |
| `finetune_lora.py` | LoRA fine-tuning script (r=4, r=8, r=16) |
| `finetune_freeze.py` | Layer freezing helper |
| `model.py` | GPT-2 model (nanoGPT-based) |
| `test_lora_memory_774m.py` | Memory measurement at 774M parameters (RQ2) |
| `lora_memory_results.json` | Measured adapter sizes (LoRA r=8: 0.78 MB, base model: 89 MB) |
| `forgetting_test_results.json` | Catastrophic forgetting sanity check results |

## Results

| Method | Adapter size | 100 tenants total | Val loss | vs baseline |
|---|---|---|---|---|
| Full fine-tuning | 189 MB | 18.99 GB | 9.36 | 69% worse |
| Layer freezing (75%) | 85 MB | 8.59 GB | 7.43 | 34% worse |
| LoRA r=8 | 0.78 MB | 167 MB | 6.43 | 16% worse |
| Base model (shared) | 89 MB | — | 5.54 | baseline |

At 774M parameters, LoRA remains 17× more memory-efficient than layer freezing.

## Dataset

- Training: Chicago Q&A dataset
- Validation: OpenWebText
- Catastrophic forgetting test: 10 general knowledge questions (2/10 correct before and after LoRA)