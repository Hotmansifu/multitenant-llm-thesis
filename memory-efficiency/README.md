# memory-efficiency/ — RQ1 & RQ2

Memory-efficient multi-tenant fine-tuning: comparing full fine-tuning, layer
freezing, and LoRA, and measuring how the storage cost per tenant scales with
model size.

L. Hoxha's master thesis *In-Memory Fine-Tuning of LLMs via Encrypted Parameter
Deltas* (Constructor University, 2025) was used as a starting point for this
folder (see the root README). The multi-tenant LoRA evaluation and the
size/scaling measurements are this thesis's contribution.

## What each script does

**Fine-tuning methods**
- `finetune_lora.py` — LoRA fine-tuning; trains only low-rank adapters, base
  weights frozen. Produces the small per-tenant adapter measured in RQ2.
- `finetune_freeze.py` — selective fine-tuning that freezes the bottom layers
  (uses `layer_freezing.py`); has its own batch/eval helpers.
- `layer_freezing.py` — helper that freezes the bottom N% of transformer layers
  (`freeze_bottom_layers`, `print_freeze_summary`).

**Size / memory measurements**
- `full_finetune_774m_size.py` — full-fine-tuning delta size for the 774M model
  (architecture-determined; the basis for the ~338.1 GB / 100-tenant figure).
- `test_lora_memory_774m.py` — LoRA adapter memory at 774M (the per-tenant
  scaling point); defines its own minimal nanoGPT-style model.
- `validate_measurement_v2.py` — cross-checks the measurement method against the
  known 30M full-delta reference (188.4 MB expected vs 189.0 MB measured, ~0.3%)
  before extrapolating to 774M.

**Quality / sanity checks**
- `check_catastrophic_forgetting.py` — checks whether a Chicago-trained LoRA
  degrades general-knowledge answers (it did not, in the reported run). Writes
  `forgetting_test_results.json`.

**Shared helpers**
- `model.py` — the GPT model definition.
- `data_loader.py` — `get_batch` (random batch from memory-mapped tokens).
- `validation.py` — `estimate_loss` (mean loss over eval batches).
- `text_generation.py` — `generate_text` (used by the forgetting check).

**Result files**
- `lora_memory_results.json`, `forgetting_test_results.json` — recorded outputs
  backing the numbers below.

## Headline numbers

- LoRA adapter: 0.78 MB at 30M, 16.25 MB at 774M.
- Total-deployment (100 tenants): LoRA ~44.7x smaller than layer freezing and
  ~98.5x smaller than full fine-tuning at 30M; ~17x vs layer freezing at 774M.
- Per-tenant (30M): ~109x less than layer freezing, ~242x less than full
  fine-tuning.
- Full-FT delta at 774M: ~338.1 GB for 100 tenants.
- Measurement-method check: 188.4 MB expected vs 189.0 MB measured (~0.3%).

## Running

```
# validate the measurement method, then measure LoRA at 774M
python validate_measurement_v2.py
python test_lora_memory_774m.py
```

The fine-tuning scripts (`finetune_lora.py`, `finetune_freeze.py`) and the
forgetting check expect a base checkpoint and a tokenized dataset; paths may
need adjusting to your environment. `finetune_lora.py` requires `peft`.

> Note: the base model, datasets, and checkpoints are not stored here. The
> data-preparation scripts and the trained checkpoints live outside the
> repository (see the root README).
