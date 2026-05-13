"""
LoRA Memory Measurement — 774M model (config originally labeled "600M")
Run on nanoGPT server:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/coder/project/nanoGPT \
  ~/.pyenv/versions/3.11.9/bin/python3 test_lora_memory_774m.py
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import os, tempfile

# ── Minimal nanoGPT-style model (exact same architecture) ──────────────────

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return torch.nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, bias):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd,     bias=bias)
    def forward(self, x): return x

class MLP(nn.Module):
    def __init__(self, n_embd, bias):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
    def forward(self, x): return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, bias):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp  = MLP(n_embd, bias)

class GPT(nn.Module):
    def __init__(self, n_layer, n_head, n_embd, block_size, vocab_size, bias):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(vocab_size, n_embd),
            wpe  = nn.Embedding(block_size, n_embd),
            h    = nn.ModuleList([Block(n_embd, n_head, bias) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_embd, bias=bias),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying (nanoGPT)
    def forward(self, x): return x

# ── Step 1: Instantiate & count params ────────────────────────────────────

print("Building model (n_layer=36, n_head=20, n_embd=1280)...")
model = GPT(n_layer=36, n_head=20, n_embd=1280,
            block_size=1024, vocab_size=50304, bias=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
print(f"  NOTE: config gives 774M, not 600M — flagged for thesis.")

# ── Step 2: Apply PEFT LoRA r=8, alpha=16 on c_attn + c_proj ─────────────

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.0,
    bias="none",
)

print("\nApplying PEFT LoRA (r=8, alpha=16, c_attn + c_proj)...")
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# ── Step 3: Save adapter & measure actual file size ───────────────────────

with tempfile.TemporaryDirectory() as tmpdir:
    peft_model.save_pretrained(tmpdir)
    adapter_bytes = os.path.getsize(os.path.join(tmpdir, "adapter_model.safetensors"))
    adapter_mb    = adapter_bytes / 1024 / 1024
    print(f"\nAdapter file size (measured): {adapter_mb:.4f} MB")

# ── Step 4: Report all numbers ────────────────────────────────────────────

lora_100_gb          = (adapter_mb * 100) / 1024
layer_freeze_per_mb  = (total_params * 0.25 * 4) / (1024 ** 2)
layer_freeze_100_gb  = (layer_freeze_per_mb * 100) / 1024
advantage            = layer_freeze_100_gb / lora_100_gb

print()
print("=" * 55)
print("RESULTS")
print("=" * 55)
print(f"Model parameters:              {total_params/1e6:.1f}M  (spec said 600M)")
print(f"LoRA adapter per tenant:       {adapter_mb:.2f} MB  [MEASURED]")
print(f"LoRA 100 tenants:              {lora_100_gb:.3f} GB")
print(f"Layer freeze per tenant:       {layer_freeze_per_mb:.1f} MB  (25% of params x 4 bytes)")
print(f"Layer freeze 100 tenants:      {layer_freeze_100_gb:.2f} GB")
print(f"Advantage (LoRA vs freeze):    {advantage:.1f}x")
print("=" * 55)