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
            wpe  = nn.Embedding(block_size, n_embd),  # block_size=256
            h    = nn.ModuleList([Block(n_embd, n_head, bias) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_embd, bias=bias),
        ))
        # NO weight tying — lm_head stored separately, matching original
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
    def forward(self, x): return x

model = GPT(n_layer=6, n_head=6, n_embd=384,
            block_size=256, vocab_size=50304, bias=True)  # block_size=256

total_params = sum(p.numel() for p in model.parameters())
print(f"30M model params: {total_params:,} ({total_params/1e6:.1f}M)")

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "test_full.pt")
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / 1024 / 1024

print(f"Our method gives:     {size_mb:.1f} MB")
print(f"Actual delta_full.pt: 189.0 MB")
print(f"Match: {'✅ YES' if abs(size_mb - 189) < 5 else '❌ NO'}")
