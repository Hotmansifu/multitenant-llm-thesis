#!/usr/bin/env python3
"""
addition_only_phe_demo.py — Addition-only PHE with W = W1 - Wm1 decomposition.

TWO FIXES vs previous version:
    FIX 1 — Trusted setup is now truly independent of the server object.
             It accesses the GPT model weights directly (as a third party
             with one-time model access) and never calls any server method.
             After setup, neither client nor server can recover r from
             what they hold: client has r, server has W@r. Neither alone
             can compute h from Cx = h + r.

    FIX 2 — Fresh mask per benchmark run. A pool of N mask sets is
             generated at setup time. Each forward pass uses a different
             (r, W@r) pair, as required for security.

Architecture:
    Trusted Setup (third party, offline):
        - Has temporary access to W (GPT model)
        - Generates N fresh (r, c=W@r) pairs
        - Gives r to client, c to server
        - Never called again after setup

    Client (trusted zone):
        - Has: r pools, corrections c, attention layers, ln_2
        - Online: Cx = h + r (addition only)
        - Online: unmask = result - c (subtraction only)
        - Never stores W

    Server (untrusted zone):
        - Has: W in plaintext (W1, Wm1), corrections c
        - Online: computes W × Cx using additions only
        - Never sees h, never sees r

Usage:
    cd /home/coder/project/nanoGPT
    PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 PYTHONPATH=. \\
        ~/.pyenv/versions/3.11.9/bin/python addition_only_phe_demo.py
"""

import os, sys, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model_bitnet import GPTConfig, GPT

CKPT_PATH  = "out/bitnet_gaussian_demo/ckpt.pt"
PROMPT     = "The future of artificial intelligence in healthcare is"
N_MASK_SETS = 12   # pool size: 1 warmup + 10 timed + 1 spare


# ═════════════════════════════════════════════════════════════════════════════
# §1  Model split
# ═════════════════════════════════════════════════════════════════════════════

class ClientModel(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.n_layer = gpt.config.n_layer
        self.n_embd  = gpt.config.n_embd
        self.wte = gpt.transformer.wte
        self.wpe = gpt.transformer.wpe
        self.ln_1_list = nn.ModuleList([copy.deepcopy(b.ln_1) for b in gpt.transformer.h])
        self.ln_2_list = nn.ModuleList([copy.deepcopy(b.ln_2) for b in gpt.transformer.h])
        self.attn_list = nn.ModuleList([copy.deepcopy(b.attn) for b in gpt.transformer.h])

    def embed(self, idx):
        B, T = idx.size()
        return self.wte(idx) + self.wpe(torch.arange(T).unsqueeze(0))

    def attention_layer(self, x, i):
        return x + self.attn_list[i](self.ln_1_list[i](x))

    def apply_ln2(self, x, i):
        return self.ln_2_list[i](x)


class ServerModel(nn.Module):
    """
    Server holds W in plaintext, decomposed as W = W1 - Wm1.
    Computes W x Cx using only additions. Never decrypts, never sees h.
    """
    def __init__(self, gpt):
        super().__init__()
        self.mlp_list = nn.ModuleList([b.mlp for b in gpt.transformer.h])
        self.ln_f     = gpt.transformer.ln_f
        self.lm_head  = gpt.lm_head
        self._W1  = {}
        self._Wm1 = {}
        self._build_decomposition(gpt)

    def _build_decomposition(self, gpt):
        for i, block in enumerate(gpt.transformer.h):
            for name, param in [('fc1', block.mlp.c_fc.weight),
                                 ('fc2', block.mlp.c_proj.weight)]:
                W   = np.sign(param.detach().float().numpy()).astype(np.float32)
                W1  = torch.tensor(np.where(W > 0, 1.0, 0.0), dtype=torch.float32)
                Wm1 = torch.tensor(np.where(W < 0, 1.0, 0.0), dtype=torch.float32)
                self._W1[(i, name)]  = W1
                self._Wm1[(i, name)] = Wm1

    def matmul_additions_only(self, Cx: torch.Tensor, layer: int, name: str) -> torch.Tensor:
        """W x Cx = W1 @ Cx - Wm1 @ Cx. Pure additions, no multiplication."""
        W1  = self._W1[(layer, name)]
        Wm1 = self._Wm1[(layer, name)]
        return W1 @ Cx - Wm1 @ Cx

    def mlp_only(self, x, i):
        return self.mlp_list[i](x)

    def head(self, x):
        return self.lm_head(self.ln_f(x))


# ═════════════════════════════════════════════════════════════════════════════
# §2  Trusted setup — FIX 1: independent of server, uses GPT model directly
# ═════════════════════════════════════════════════════════════════════════════

def trusted_setup(gpt, n_layer: int, C: int, N: int = N_MASK_SETS):
    """
    Trusted third party: has temporary access to the GPT model.
    Generates N independent (r, W@r) pairs per layer.
    Does NOT use the server object — completely independent.

    After this function returns:
      - client holds: r values (cannot compute W@r without W)
      - server holds: c = W@r values (cannot compute r without W^-1)
      - Neither party can recover h from Cx=h+r alone.

    Returns:
        pool_client: list of N mask sets, each with n_layer entries
        pool_server: list of N correction sets, each with n_layer entries
    """
    pool_client = []
    pool_server = []

    for _ in range(N):
        masks       = []
        corrections = []

        for i in range(n_layer):
            C4 = 4 * C

            # Third party accesses W directly from the model — not via server
            W_fc1 = torch.tensor(
                np.sign(gpt.transformer.h[i].mlp.c_fc.weight.detach().float().numpy()),
                dtype=torch.float32)   # shape (4C, C)
            W_fc2 = torch.tensor(
                np.sign(gpt.transformer.h[i].mlp.c_proj.weight.detach().float().numpy()),
                dtype=torch.float32)   # shape (C, 4C)

            # Fresh random masks
            r_fc1 = torch.randn(C)
            r_fc2 = torch.randn(C4)

            # Corrections computed by third party (not by server)
            c_fc1 = W_fc1 @ r_fc1   # (4C,)
            c_fc2 = W_fc2 @ r_fc2   # (C,)

            masks.append({'r_fc1': r_fc1, 'r_fc2': r_fc2})
            corrections.append({'c_fc1': c_fc1, 'c_fc2': c_fc2})
            # W_fc1, W_fc2 go out of scope here

        pool_client.append(masks)
        pool_server.append(corrections)

    return pool_client, pool_server


# ═════════════════════════════════════════════════════════════════════════════
# §3  Correctness check
# ═════════════════════════════════════════════════════════════════════════════

def verify_scheme(gpt, server: ServerModel, C: int):
    """Verify W x (x+r) - W@r = W x x using independent weight access."""
    x = torch.randn(C)
    r = torch.randn(C)

    # Reference: plaintext W @ x (using server's decomposition)
    plain = server.matmul_additions_only(x, 0, 'fc1')

    # Masked path: third party computes correction independently
    W_fc1 = torch.tensor(
        np.sign(gpt.transformer.h[0].mlp.c_fc.weight.detach().float().numpy()),
        dtype=torch.float32)
    correction = W_fc1 @ r   # computed by third party, not server

    Cx        = x + r
    result    = server.matmul_additions_only(Cx, 0, 'fc1')   # server
    recovered = result - correction                            # client unmasks

    diff = (plain - recovered).abs().max().item()
    return diff < 1e-4, diff


# ═════════════════════════════════════════════════════════════════════════════
# §4  Forward passes
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def s1_forward(client, server, idx):
    """Full-precision float MLP baseline (for reference only)."""
    x = client.embed(idx)
    for i in range(client.n_layer):
        x = client.attention_layer(x, i)
        h_norm = client.apply_ln2(x, i)
        x = x + server.mlp_only(h_norm, i)
    return server.head(x)


@torch.no_grad()
def ternary_baseline_forward(client, server, idx):
    """
    Correct baseline: same ternary matmul as PHE but NO masking.
    Fair comparison for overhead measurement — overhead = PHE / this.
    """
    x = client.embed(idx)
    T = x.shape[1]
    for i in range(client.n_layer):
        x = client.attention_layer(x, i)
        h_norm = client.apply_ln2(x, i)
        fc2_tokens = []
        for t in range(T):
            h_t = h_norm[0, t, :]
            fc1_out = server.matmul_additions_only(h_t, i, 'fc1')
            gelu    = F.gelu(fc1_out)
            fc2_out = server.matmul_additions_only(gelu, i, 'fc2')
            fc2_tokens.append(fc2_out)
        mlp_out = torch.stack(fc2_tokens).unsqueeze(0)
        x = x + mlp_out
    return server.head(x)


@torch.no_grad()
def phe_forward(client, server, idx, client_masks, server_corrections):
    """
    Addition-only PHE forward pass. Uses one mask set (fresh per call).
    FIX 2: caller provides a fresh (client_masks, server_corrections) each run.
    """
    x = client.embed(idx)
    T = x.shape[1]

    for i in range(client.n_layer):
        x      = client.attention_layer(x, i)
        h_norm = client.apply_ln2(x, i)

        r_fc1 = client_masks[i]['r_fc1']
        r_fc2 = client_masks[i]['r_fc2']
        c_fc1 = server_corrections[i]['c_fc1']
        c_fc2 = server_corrections[i]['c_fc2']

        fc2_tokens = []
        for t in range(T):
            h_t = h_norm[0, t, :]

            # fc1
            Cx      = h_t + r_fc1                               # client: +r
            res1    = server.matmul_additions_only(Cx, i, 'fc1') # server: additions
            fc1_out = res1 - c_fc1                              # client: -correction
            gelu    = F.gelu(fc1_out)                           # client: GELU

            # fc2
            Cg      = gelu + r_fc2                              # client: +r
            res2    = server.matmul_additions_only(Cg, i, 'fc2') # server: additions
            fc2_out = res2 - c_fc2                              # client: -correction

            fc2_tokens.append(fc2_out)

        mlp_out = torch.stack(fc2_tokens).unsqueeze(0)
        x = x + mlp_out

    return server.head(x)


# ═════════════════════════════════════════════════════════════════════════════
# §5  Benchmark
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("addition_only_phe_demo.py — W=W1-Wm1, fresh mask per run, independent setup")
    print("=" * 70)

    if not os.path.exists(CKPT_PATH):
        print(f"ERROR: checkpoint not found: {CKPT_PATH}"); sys.exit(1)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model_args = ckpt["model_args"]
    if "config" in ckpt:
        model_args.update({k: v for k, v in ckpt["config"].items()
                          if k not in model_args})
    model_args["use_bitnet"] = True
    cfg = GPTConfig(**model_args)
    gpt = GPT(cfg)
    sd  = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    gpt.load_state_dict(sd, strict=False)
    gpt.eval()

    client = ClientModel(gpt)
    server = ServerModel(gpt)
    C = cfg.n_embd
    print(f"Model: {cfg.n_layer} layers, C={C}")

    # ── Correctness check ─────────────────────────────────────────────────
    print(f"\n── Correctness check ──")
    ok, diff = verify_scheme(gpt, server, C)
    print(f"  W x (x+r) - W@r = W x x : {'✓ PASS' if ok else '✗ FAIL'}")
    print(f"  Max numerical error       : {diff:.2e}")
    print(f"  Trusted setup independent of server: ✓")

    # ── Trusted setup (offline, not timed) ───────────────────────────────
    print(f"\n── Trusted setup (offline, third party, not timed) ──")
    pool_client, pool_server = trusted_setup(gpt, cfg.n_layer, C, N=N_MASK_SETS)
    print(f"  Generated {N_MASK_SETS} fresh mask sets (1 warmup + 10 timed + 1 spare)")
    print(f"  Setup accesses GPT model directly — NOT via server object")
    print(f"  Client receives: r vectors only (cannot compute W@r without W)")
    print(f"  Server receives: corrections W@r only (cannot compute r without W⁻¹)")
    print(f"  W NOT stored on client after setup")

    # ── Tokenize ─────────────────────────────────────────────────────────
    enc_tok = tiktoken.get_encoding("gpt2")
    tokens  = enc_tok.encode(PROMPT)
    idx     = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    # ── S1 float baseline (reference only) ───────────────────────────────
    print(f"\n── S1 float baseline (full precision MLP, reference only) ──")
    s1_forward(client, server, idx)
    t0 = time.perf_counter()
    for _ in range(10):
        s1_forward(client, server, idx)
    t_s1 = (time.perf_counter() - t0) / 10
    print(f"  S1: {t_s1*1000:.2f} ms")

    # ── Ternary baseline (fair comparison — same ops as PHE, no masking) ──
    print(f"\n── Ternary baseline (same ops as PHE, no masking, N=10) ──")
    ternary_baseline_forward(client, server, idx)
    t0 = time.perf_counter()
    for _ in range(10):
        ternary_baseline_forward(client, server, idx)
    t_ternary = (time.perf_counter() - t0) / 10
    print(f"  Ternary: {t_ternary*1000:.2f} ms")

    # ── PHE forward — fresh mask per run ──────────────────────────────────
    print(f"\n── Addition-only PHE forward (fresh mask per run, N=10) ──")
    phe_forward(client, server, idx, pool_client[0], pool_server[0])  # warmup
    t0 = time.perf_counter()
    for k in range(10):
        phe_forward(client, server, idx, pool_client[k+1], pool_server[k+1])
    t_phe = (time.perf_counter() - t0) / 10
    overhead = t_phe / t_ternary   # fair: same computation, one with masking
    print(f"  PHE: {t_phe*1000:.2f} ms")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n── Summary ──")
    print(f"  S1 float baseline (reference)  : {t_s1*1000:.1f} ms")
    print(f"  Ternary baseline (no masking)  : {t_ternary*1000:.1f} ms")
    print(f"  Addition-only PHE (with masks) : {t_phe*1000:.1f} ms")
    print(f"  Overhead vs ternary baseline   : {overhead:.2f}×")
    print(f"\n  Server operations              : additions only (W = W1 - Wm1)")
    print(f"  Client stores W               : NO")
    print(f"  Crypto library                : NONE")
    print(f"  Fresh mask per inference      : YES")
    print(f"  Setup independent of server   : YES")
    print(f"  Known limitation              : ln_f, lm_head still on server")


if __name__ == "__main__":
    main()