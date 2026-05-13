#!/usr/bin/env python3
"""
bitnet_compound_extended.py
============================
Extended compound test — runs 3 & 4 only with 2400 iterations.
Also tests σ=4 and σ=6 alongside σ=5 for BitNet+Gaussian.

Runs:
  1. BitNet only (softmax + ternary) — 2400 iters
  2. BitNet + Gaussian σ=5           — 2400 iters
  3. BitNet + Gaussian σ=4           — 2400 iters
  4. BitNet + Gaussian σ=6           — 2400 iters

Uses baseline ppl=1.54 and gaussian_only ppl=1.56 from previous run.
"""

import os
import sys
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from model import GPTConfig, GPT


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

DATA_DIR = Path('/home/coder/project/data/chicago/bin')
FALLBACK_DATA_DIRS = [
    PROJECT_ROOT / 'data' / 'chicago' / 'bin',
    PROJECT_ROOT / 'data' / 'chicago',
]
OUTPUT_DIR = PROJECT_ROOT / 'out' / 'compound_test_extended'

MODEL_CONFIG = dict(
    n_layer=6, n_head=8, n_embd=768,
    block_size=256, dropout=0.10, bias=True, tie_weights=True,
)

TRAIN_CONFIG = dict(
    batch_size=8, max_iters=2400, eval_interval=200, eval_iters=50,
    learning_rate=1e-4, weight_decay=0.10, warmup_iters=100, min_lr=6e-5,
    grad_clip=1.0,
)

# From previous run (1200 iters) — no need to re-run these
PREVIOUS_RESULTS = {
    'baseline_1200': {'val_loss': 0.4345, 'perplexity': 1.54, 'time': 247.8, 'iters': 1200},
    'gaussian_only_1200': {'val_loss': 0.4430, 'perplexity': 1.56, 'time': 268.4, 'iters': 1200},
}


# ═══════════════════════════════════════════════════════════════════════════
# BITNET (same as before)
# ═══════════════════════════════════════════════════════════════════════════

class STESign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        scale = weight.abs().mean() + 1e-8
        w_normalized = weight / scale
        w_ternary = w_normalized.round().clamp(-1, 1)
        ctx.save_for_backward(weight)
        return w_ternary * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ternary_quantize(weight):
    return STESign.apply(weight)


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        w_q = ternary_quantize(self.linear.weight)
        return F.linear(x, w_q, self.linear.bias)


def apply_bitnet_to_model(model):
    replaced = 0
    for block in model.transformer.h:
        for name in ['c_attn', 'c_proj']:
            old = getattr(block.attn, name)
            new = BitLinear(old.in_features, old.out_features, bias=old.bias is not None)
            new.linear.weight = old.weight
            new.linear.bias = old.bias
            setattr(block.attn, name, new)
            replaced += 1
        for name in ['c_fc', 'c_proj']:
            old = getattr(block.mlp, name)
            new = BitLinear(old.in_features, old.out_features, bias=old.bias is not None)
            new.linear.weight = old.weight
            new.linear.bias = old.bias
            setattr(block.mlp, name, new)
            replaced += 1
    print(f"  BitNet: replaced {replaced} Linear layers with BitLinear")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# DATA & TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def find_data_dir():
    if DATA_DIR.exists() and (DATA_DIR / 'train.bin').exists():
        return DATA_DIR
    for d in FALLBACK_DATA_DIRS:
        if d.exists():
            if (d / 'train.bin').exists():
                return d
            if (d / 'bin' / 'train.bin').exists():
                return d / 'bin'
    print("ERROR: Cannot find training data.")
    sys.exit(1)


def load_data(data_dir):
    train_data = np.memmap(str(data_dir / 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(str(data_dir / 'val.bin'), dtype=np.uint16, mode='r')
    print(f"Loaded {len(train_data):,} train / {len(val_data):,} val tokens")
    return train_data, val_data


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device):
    model.eval()
    results = {}
    for name, data in [('train', train_data), ('val', val_data)]:
        losses = []
        for _ in range(cfg['eval_iters']):
            x, y = get_batch(data, cfg['batch_size'], model.config.block_size, device)
            _, loss = model(x, targets=y)
            losses.append(loss.item())
        results[name] = np.mean(losses)
    model.train()
    return results


def get_lr(it, cfg):
    if it < cfg['warmup_iters']:
        return cfg['learning_rate'] * it / cfg['warmup_iters']
    if it > cfg['max_iters']:
        return cfg['min_lr']
    decay_ratio = (it - cfg['warmup_iters']) / (cfg['max_iters'] - cfg['warmup_iters'])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg['min_lr'] + coeff * (cfg['learning_rate'] - cfg['min_lr'])


def train_model(model, train_data, val_data, device, run_name):
    tc = TRAIN_CONFIG
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'ln_' in name or 'wte' in name or 'wpe' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optim_groups = [
        {"params": decay_params, "weight_decay": tc['weight_decay']},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=tc['learning_rate'], betas=(0.9, 0.95))

    history = []
    t0 = time.time()
    model.train()

    for it in range(tc['max_iters']):
        lr = get_lr(it, tc)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if it % tc['eval_interval'] == 0 or it == tc['max_iters'] - 1:
            losses = estimate_loss(model, train_data, val_data, tc, device)
            ppl = math.exp(losses['val']) if losses['val'] < 20 else float('inf')
            elapsed = time.time() - t0
            print(f"  [{run_name}] iter {it:5d} | val {losses['val']:.4f} | ppl {ppl:.2f} | {elapsed:.0f}s")
            history.append({'iter': it, 'train_loss': losses['train'], 'val_loss': losses['val'], 'ppl': ppl})

        x, y = get_batch(train_data, tc['batch_size'], model.config.block_size, device)
        _, loss = model(x, targets=y)
        loss.backward()
        if tc['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc['grad_clip'])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    final = estimate_loss(model, train_data, val_data, {**tc, 'eval_iters': tc['eval_iters'] * 2}, device)
    final_ppl = math.exp(final['val']) if final['val'] < 20 else float('inf')
    total_time = time.time() - t0
    print(f"  ✓ {run_name} done in {total_time:.1f}s — val_loss={final['val']:.4f}, ppl={final_ppl:.2f}")

    metrics = {'val_loss': final['val'], 'train_loss': final['train'], 'perplexity': final_ppl, 'time': total_time}
    return model, metrics, history


# ═══════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(all_results, output_path):
    baseline_ppl = PREVIOUS_RESULTS['baseline_1200']['perplexity']

    lines = []
    lines.append("=" * 90)
    lines.append("EXTENDED COMPOUND ERROR TEST: BitNet + Gaussian Kernel (2400 iters)")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"Model: {MODEL_CONFIG['n_layer']}L / {MODEL_CONFIG['n_head']}H / {MODEL_CONFIG['n_embd']}d")
    lines.append(f"Training: {TRAIN_CONFIG['max_iters']} iterations (extended from 1200)")
    lines.append(f"Baseline perplexity (1200 iters): {baseline_ppl}")
    lines.append("")

    # Previous results for reference
    lines.append("PREVIOUS RESULTS (1200 iters):")
    lines.append("-" * 70)
    lines.append(f"  {'Baseline':<30} | ppl {PREVIOUS_RESULTS['baseline_1200']['perplexity']:.2f} | base")
    lines.append(f"  {'Gaussian σ=5 only':<30} | ppl {PREVIOUS_RESULTS['gaussian_only_1200']['perplexity']:.2f} | +0.9%")
    lines.append("")

    # New results
    lines.append("NEW RESULTS (2400 iters):")
    lines.append("-" * 90)
    lines.append(f"{'Run':<30} | {'Val Loss':>10} | {'Perplexity':>12} | {'Time (s)':>10} | {'Δ vs baseline':>14}")
    lines.append("-" * 90)

    for name, r in all_results.items():
        delta = (r['perplexity'] - baseline_ppl) / baseline_ppl * 100
        lines.append(f"{name:<30} | {r['val_loss']:>10.4f} | {r['perplexity']:>12.2f} | {r['time']:>10.1f} | {delta:>+13.1f}%")

    lines.append("-" * 90)
    lines.append("")

    # Compound analysis
    bitnet_ppl = all_results['bitnet_2400']['perplexity']
    bitnet_delta = (bitnet_ppl - baseline_ppl) / baseline_ppl * 100

    lines.append("COMPOUND ERROR ANALYSIS (2400 iters):")
    lines.append("-" * 50)
    lines.append(f"  Gaussian only (1200i):       +0.9%")
    lines.append(f"  BitNet only (2400i):         +{bitnet_delta:.1f}%")

    # Find best sigma
    sigma_results = {}
    for name, r in all_results.items():
        if 'gaussian' in name.lower():
            sigma_results[name] = (r['perplexity'] - baseline_ppl) / baseline_ppl * 100

    if sigma_results:
        best_name = min(sigma_results, key=sigma_results.get)
        best_delta = sigma_results[best_name]
        lines.append(f"")
        lines.append(f"  Sigma comparison (BitNet + Gaussian, 2400i):")
        for name, delta in sorted(sigma_results.items()):
            marker = " ← BEST" if name == best_name else ""
            lines.append(f"    {name:<28} +{delta:.1f}%{marker}")
        lines.append(f"")
        lines.append(f"  Best compound:               +{best_delta:.1f}%")
        additive = 0.9 + bitnet_delta
        lines.append(f"  Expected (additive):         +{additive:.1f}%")
        lines.append(f"  Interaction effect:          {best_delta - additive:+.1f}%")

    lines.append("")

    # Verdict
    best_compound = min(sigma_results.values()) if sigma_results else 999
    if best_compound < 5.0:
        lines.append("✅ COMPOUND DEGRADATION < 5% — ACCEPTABLE")
        lines.append("   Ready to proceed to encryption prototype.")
    elif best_compound < 7.0:
        lines.append("⚠️  COMPOUND DEGRADATION 5-7% — MARGINAL but workable")
        lines.append("   Consider more training iterations or sigma tuning.")
    else:
        lines.append("❌ COMPOUND DEGRADATION > 7% — needs investigation")

    # Comparison: 1200 vs 2400
    if 'bitnet_gaussian_s5_2400' in all_results:
        old_compound = 5.3  # from previous run
        new_compound = (all_results['bitnet_gaussian_s5_2400']['perplexity'] - baseline_ppl) / baseline_ppl * 100
        improvement = old_compound - new_compound
        lines.append(f"")
        lines.append(f"  Improvement from extended training (σ=5):")
        lines.append(f"    1200 iters: +5.3%")
        lines.append(f"    2400 iters: +{new_compound:.1f}%")
        lines.append(f"    Gained:     {improvement:+.1f}%")

    lines.append("")
    lines.append("=" * 90)

    report = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\n{report}")
    return report


def generate_plot(all_histories, output_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        colors = ['#FF9800', '#F44336', '#9C27B0', '#4CAF50']
        for i, (name, history) in enumerate(all_histories.items()):
            iters = [h['iter'] for h in history]
            val_losses = [h['val_loss'] for h in history]
            ppls = [h['ppl'] for h in history]
            c = colors[i % len(colors)]
            ax1.plot(iters, val_losses, color=c, label=name, linewidth=2)
            ax2.plot(iters, ppls, color=c, label=name, linewidth=2)

        # Add baseline reference line
        ax1.axhline(y=0.4345, color='#2196F3', linestyle='--', alpha=0.7, label='Baseline (1200i)')
        ax2.axhline(y=1.54, color='#2196F3', linestyle='--', alpha=0.7, label='Baseline (1200i)')

        ax1.set_xlabel('Iteration'); ax1.set_ylabel('Val Loss'); ax1.set_title('Convergence — Val Loss')
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        ax2.set_xlabel('Iteration'); ax2.set_ylabel('Perplexity'); ax2.set_title('Convergence — Perplexity')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        plt.suptitle('Extended Compound Test (2400 iters) + Sigma Tuning', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot saved: {output_path}")
    except ImportError:
        print("⚠ matplotlib not available, skipping plot")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 80)
    print("EXTENDED COMPOUND TEST (2400 iters) + SIGMA TUNING")
    print("Skipping baseline & gaussian-only (already done at 1200i)")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = find_data_dir()
    print(f"Data: {data_dir}")
    train_data, val_data = load_data(data_dir)

    vocab_size = max(int(train_data.max()) + 1, 256)
    vocab_size = ((vocab_size + 63) // 64) * 64

    all_results = {}
    all_histories = {}

    # ── Run 1: BitNet only, 2400 iters ──────────────────────────────────
    print(f"\n{'='*70}")
    print("RUN 1/4: BITNET ONLY (softmax + ternary) — 2400 iters")
    print(f"{'='*70}")
    torch.manual_seed(42); np.random.seed(42)
    config = GPTConfig(**MODEL_CONFIG, vocab_size=vocab_size, use_gaussian_kernel=False)
    model = GPT(config).to(device)
    model = apply_bitnet_to_model(model)
    model, metrics, history = train_model(model, train_data, val_data, device, "BitNet 2400i")
    all_results['bitnet_2400'] = metrics
    all_histories['bitnet_2400'] = history
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Run 2: BitNet + Gaussian σ=5, 2400 iters ────────────────────────
    print(f"\n{'='*70}")
    print("RUN 2/4: BITNET + GAUSSIAN σ=5 — 2400 iters")
    print(f"{'='*70}")
    torch.manual_seed(42); np.random.seed(42)
    config = GPTConfig(**MODEL_CONFIG, vocab_size=vocab_size,
                       use_gaussian_kernel=True, gaussian_sigma=5.0, gaussian_sigma_learnable=False)
    model = GPT(config).to(device)
    model = apply_bitnet_to_model(model)
    model, metrics, history = train_model(model, train_data, val_data, device, "BitNet+G σ=5")
    all_results['bitnet_gaussian_s5_2400'] = metrics
    all_histories['bitnet_gaussian_s5_2400'] = history
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Run 3: BitNet + Gaussian σ=4, 2400 iters ────────────────────────
    print(f"\n{'='*70}")
    print("RUN 3/4: BITNET + GAUSSIAN σ=4 — 2400 iters")
    print(f"{'='*70}")
    torch.manual_seed(42); np.random.seed(42)
    config = GPTConfig(**MODEL_CONFIG, vocab_size=vocab_size,
                       use_gaussian_kernel=True, gaussian_sigma=4.0, gaussian_sigma_learnable=False)
    model = GPT(config).to(device)
    model = apply_bitnet_to_model(model)
    model, metrics, history = train_model(model, train_data, val_data, device, "BitNet+G σ=4")
    all_results['bitnet_gaussian_s4_2400'] = metrics
    all_histories['bitnet_gaussian_s4_2400'] = history
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Run 4: BitNet + Gaussian σ=6, 2400 iters ────────────────────────
    print(f"\n{'='*70}")
    print("RUN 4/4: BITNET + GAUSSIAN σ=6 — 2400 iters")
    print(f"{'='*70}")
    torch.manual_seed(42); np.random.seed(42)
    config = GPTConfig(**MODEL_CONFIG, vocab_size=vocab_size,
                       use_gaussian_kernel=True, gaussian_sigma=6.0, gaussian_sigma_learnable=False)
    model = GPT(config).to(device)
    model = apply_bitnet_to_model(model)
    model, metrics, history = train_model(model, train_data, val_data, device, "BitNet+G σ=6")
    all_results['bitnet_gaussian_s6_2400'] = metrics
    all_histories['bitnet_gaussian_s6_2400'] = history
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Save & Report ────────────────────────────────────────────────────
    results_path = OUTPUT_DIR / 'compound_extended_results.json'
    with open(results_path, 'w') as f:
        json.dump({'results': all_results, 'histories': all_histories,
                   'previous': PREVIOUS_RESULTS,
                   'model_config': MODEL_CONFIG, 'train_config': TRAIN_CONFIG}, f, indent=2)
    print(f"\n✓ Results saved: {results_path}")

    generate_report(all_results, OUTPUT_DIR / 'compound_extended_report.txt')
    generate_plot(all_histories, OUTPUT_DIR / 'compound_extended_plot.png')

    print(f"\n{'='*80}")
    print("DONE — All outputs in:", OUTPUT_DIR)
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
