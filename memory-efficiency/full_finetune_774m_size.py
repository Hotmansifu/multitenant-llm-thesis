"""
Full fine-tuning delta size measurement — 774M model
No training needed — file size is determined by architecture alone.
"""
import torch
import torch.nn as nn
import os, tempfile

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, bias):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd,     bias=bias)

class MLP(nn.Module):
    def __init__(self, n_embd, bias):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)

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
        self.transformer.wte.weight = self.lm_head.weight
    def forward(self, x): return x

print("Building 774M model...")
model = GPT(n_layer=36, n_head=20, n_embd=1280,
            block_size=1024, vocab_size=50304, bias=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

# Save full state dict — same as what full fine-tuning produces
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "delta_full_774m.pt")
    torch.save(model.state_dict(), path)
    size_bytes = os.path.getsize(path)
    size_mb    = size_bytes / 1024 / 1024
    size_gb    = size_mb / 1024

print()
print("=" * 50)
print("RESULTS")
print("=" * 50)
print(f"Full fine-tuning per tenant:  {size_mb:.1f} MB  [MEASURED]")
print(f"Full fine-tuning 100 tenants: {size_gb * 100:.2f} GB")
print("=" * 50)
