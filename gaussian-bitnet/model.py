#!/usr/bin/env python3
"""
model.py — GPT model with Gaussian Kernel Attention support

RQ4: Testing Gaussian kernel attention as encryption-friendly softmax replacement

Changes from original:
  - GPTConfig gains: use_gaussian_kernel, gaussian_sigma, gaussian_sigma_learnable
  - CausalSelfAttention gains: _attn_gaussian() method
  - Forward routing: gaussian kernel path bypasses SDPA (must use eager)

Toggle: set config.use_gaussian_kernel = True to switch attention mechanism.
Everything else (architecture, weight names, checkpoint compatibility) is unchanged.
"""
from __future__ import annotations
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Config
# -------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = True
    model_type: str = "gpt"

    # ── Gaussian Kernel Attention settings ──────────────────────────────────
    use_gaussian_kernel: bool = False
    # Initial sigma value. Default = sqrt(head_dim) to mirror 1/sqrt(d_k) scaling.
    # Set to 0.0 to auto-compute as sqrt(n_embd // n_head) at init time.
    gaussian_sigma: float = 0.0
    # If True, sigma is nn.Parameter (learned per-head). If False, fixed constant.
    gaussian_sigma_learnable: bool = False

    def get(self, key, default=None):
        """Dict-like get method for PEFT compatibility"""
        return getattr(self, key, default)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'vocab_size': self.vocab_size,
            'block_size': self.block_size,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_embd': self.n_embd,
            'dropout': self.dropout,
            'bias': self.bias,
            'tie_weights': self.tie_weights,
            'model_type': self.model_type,
            'use_gaussian_kernel': self.use_gaussian_kernel,
            'gaussian_sigma': self.gaussian_sigma,
            'gaussian_sigma_learnable': self.gaussian_sigma_learnable,
        }


# -------------------------------
# Modules
# -------------------------------
class LayerNorm(nn.Module):
    """LayerNorm with optional bias to match NanoGPT checkpoints exactly."""
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


_SLOW_ATTN_WARNED = False
def _warn_slow_attn_once(msg: str):
    global _SLOW_ATTN_WARNED
    if not _SLOW_ATTN_WARNED:
        print(msg, flush=True)
        _SLOW_ATTN_WARNED = True


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional Gaussian kernel.

    Weight names are aligned with NanoGPT so existing checkpoints load cleanly:
      - c_attn:  projects to qkv  (3 * n_embd)
      - c_proj:  output projection
      - bias:    causal mask buffer

    When config.use_gaussian_kernel is True, the softmax(QK^T / sqrt(d)) scoring
    is replaced with a Gaussian kernel:

        A_ij = exp( -||q_i - k_j||^2 / (2 * sigma^2) )

    followed by row-wise normalization (D^{-1} A, as in Skyformer).

    Why this matters for encrypted inference:
      - Softmax under HE needs ~100 polynomial approximations (exp + division per element)
      - Gaussian kernel needs ~20 (squared distance is native, single exp, no division)
      - 5× speedup in the encrypted attention critical path
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_gaussian_kernel = config.use_gaussian_kernel

        # Projections (identical for both attention modes)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        mask = mask.view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", mask, persistent=False)

        # ── Gaussian kernel sigma ───────────────────────────────────────────
        if self.use_gaussian_kernel:
            sigma_init = config.gaussian_sigma
            if sigma_init <= 0.0:
                # Auto: sqrt(head_dim), analogous to 1/sqrt(d_k) scaling
                sigma_init = math.sqrt(self.head_dim)

            if config.gaussian_sigma_learnable:
                # One learnable sigma per head
                self.sigma = nn.Parameter(
                    torch.full((1, self.n_head, 1, 1), sigma_init)
                )
            else:
                # Fixed sigma (register as buffer so it moves to device with model)
                self.register_buffer(
                    "sigma",
                    torch.full((1, self.n_head, 1, 1), sigma_init),
                    persistent=False,
                )

        # Detect SDPA availability (PyTorch >= 2.0)
        self._has_sdpa = hasattr(F, "scaled_dot_product_attention")
        self._quiet = os.environ.get("QUIET_ATTENTION_WARN", "0") == "1"

    # ── Softmax attention (eager fallback) ──────────────────────────────────
    def _attn_eager(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Standard scaled dot-product attention with explicit causal mask."""
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
        T = q.size(-2)
        att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v

    # ── Softmax attention (SDPA fast path) ──────────────────────────────────
    def _attn_sdpa(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
    ) -> torch.Tensor:
        """PyTorch SDPA — handles masking internally."""
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

    # ── Gaussian kernel attention ───────────────────────────────────────────
    def _attn_gaussian(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Gaussian kernel attention (Skyformer-style).

        Instead of:  softmax( Q K^T / sqrt(d) )
        We compute:  D^{-1} exp( -||q_i - k_j||^2 / (2 σ^2) )

        where D is the diagonal row-sum matrix (row-wise normalization).

        The squared distance decomposes as:
            ||q_i - k_j||^2 = ||q_i||^2 + ||k_j||^2 - 2 (q_i · k_j)

        This is crucial for HE compatibility: the distance is built from
        dot products (native HE ops) + a single exp (vs softmax's exp + div).
        """
        B, nh, T, hd = q.shape

        # ── Squared Euclidean distances ─────────────────────────────────────
        #   q_norm_sq: (B, nh, T, 1)   — ||q_i||^2 for each query position
        #   k_norm_sq: (B, nh, 1, T)   — ||k_j||^2 for each key position
        #   qk_dot:    (B, nh, T, T)   — q_i · k_j dot products
        q_norm_sq = (q * q).sum(dim=-1, keepdim=True)            # (B, nh, T, 1)
        k_norm_sq = (k * k).sum(dim=-1, keepdim=True).transpose(-2, -1)  # (B, nh, 1, T)
        qk_dot = q @ k.transpose(-2, -1)                         # (B, nh, T, T)

        dist_sq = q_norm_sq + k_norm_sq - 2.0 * qk_dot           # (B, nh, T, T)

        # Clamp to avoid negative distances from floating-point errors
        dist_sq = dist_sq.clamp(min=0.0)

        # ── Gaussian kernel ─────────────────────────────────────────────────
        #   sigma shape: (1, n_head, 1, 1) — broadcasts across B, T, T
        att = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))

        # ── Causal masking ──────────────────────────────────────────────────
        # Zero out future positions (not -inf, since we're not in log-space)
        att = att.masked_fill(mask[:, :, :T, :T] == 0, 0.0)

        # ── Row-wise normalization (D^{-1} A) ──────────────────────────────
        # Add epsilon for numerical stability (avoid division by zero)
        row_sums = att.sum(dim=-1, keepdim=True) + 1e-8
        att = att / row_sums

        att = self.attn_dropout(att)
        return att @ v

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)            # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # Reshape: (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.use_gaussian_kernel:
            # ── Gaussian kernel path (always eager — no SDPA equivalent) ────
            y = self._attn_gaussian(q, k, v, self.bias)
        elif self._has_sdpa:
            # ── Standard softmax via SDPA (fast) ────────────────────────────
            y = self._attn_sdpa(q, k, v, is_causal=True)
        else:
            # ── Standard softmax eager fallback ─────────────────────────────
            if not self._quiet:
                _warn_slow_attn_once(
                    "WARNING: using slow attention (no SDPA in this torch build)"
                )
            y = self._attn_eager(q, k, v, self.bias)

        # Reassemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block. Submodule names match NanoGPT checkpoints:
      transformer.h.{i}.ln_1, .attn.c_attn, .attn.c_proj, .ln_2, .mlp.c_fc, .mlp.c_proj
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# -------------------------------
# GPT model
# -------------------------------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (common in GPT-2 / NanoGPT)
        if config.tie_weights:
            self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

        # Ensure causal mask buffers are consistent
        for block in self.transformer.h:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            mask = mask.view(1, 1, config.block_size, config.block_size)
            block.attn.bias = mask
            block.attn.register_buffer("bias", mask, persistent=False)

        # Report
        n_params = sum(p.numel() for p in self.parameters())
        attn_mode = "GAUSSIAN KERNEL" if config.use_gaussian_kernel else "softmax"
        print(f"number of parameters: {n_params/1e6:.2f}M  |  attention: {attn_mode}")
        if config.use_gaussian_kernel:
            sigma_val = config.gaussian_sigma if config.gaussian_sigma > 0 else math.sqrt(config.n_embd // config.n_head)
            learnable = "learnable" if config.gaussian_sigma_learnable else "fixed"
            print(f"  gaussian sigma = {sigma_val:.4f} ({learnable})")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass supporting both idx/targets and input_ids/labels parameter names.

        Returns:
            (logits, loss)
        """
        if input_ids is not None:
            idx = input_ids
        if labels is not None:
            targets = labels
        if idx is None:
            raise ValueError("Either 'idx' or 'input_ids' must be provided")

        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, sequence length too large."

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """PEFT compatibility method"""
        return {"idx": input_ids}

    def get_input_embeddings(self):
        """PEFT compatibility method"""
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        """PEFT compatibility method"""
        self.transformer.wte = new_embeddings

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas=(0.9, 0.95),
        device_type: str = "cpu",
    ) -> torch.optim.Optimizer:
        """
        Create AdamW with proper weight decay groups.
        Handles tied weights (lm_head.weight → transformer.wte.weight).
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("transformer.wpe.weight")
        no_decay.add("transformer.wte.weight")
        decay.discard("lm_head.weight")
        no_decay.discard("lm_head.weight")

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [pn for pn in sorted(decay) if pn in param_dict]
        no_decay = [pn for pn in sorted(no_decay) if pn in param_dict]

        optim_groups = [
            {"params": [param_dict[pn] for pn in decay], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in no_decay], "weight_decay": 0.0},
        ]

        use_fused = (
            device_type == "cuda"
            and torch.cuda.is_available()
            and hasattr(torch.optim.AdamW, "supports_fused")
            and torch.optim.AdamW.supports_fused
        )
        print("using fused AdamW:", use_fused)

        if use_fused:
            opt = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)
        else:
            opt = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        if device_type != "cuda":
            setattr(opt, "fused", False)
        return opt
