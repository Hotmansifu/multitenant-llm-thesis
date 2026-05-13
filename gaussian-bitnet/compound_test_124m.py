#!/usr/bin/env python3
"""
compound_test_124m.py
======================
Compound error experiment at 124M parameter scale on Chicago dataset.
Mirrors bitnet_compound_test.py exactly — same structure, same functions,
same optimizer, same eval logic, same four configurations.

Only changes from bitnet_compound_test.py:
  - MODEL_CONFIG: n_layer=6,n_head=8,n_embd=768 → n_layer=12,n_head=12,n_embd=768
  - TRAIN_CONFIG: max_iters=1200 → 4800  (2x convergence artefact, then 2x safety)
  - OUTPUT_DIR: compound_test → compound_test_124m
  - SIGMA: 5.0 → 4.0  (confirmed best from extended run)

30M config was: n_layer=6,  n_head=8,  n_embd=768
124M config is: n_layer=12, n_head=12, n_embd=768

Usage:
    cd /home/coder/project
    python compound_test_124m.py

Results saved to:
    out/compound_test_124m/compound_results.json
    out/compound_test_124m/compound_report.txt
    out/compound_test_124m/compound_plot.png
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
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from model import GPTConfig, GPT


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

DATA_DIR = Path('/home/coder/project/config/data/chicago/bin')
FALLBACK_DATA_DIRS = [
    PROJECT_ROOT / 'config' / 'data' / 'chicago' / 'bin',
    PROJECT_ROOT / 'data' / 'chicago' / 'bin',
    PROJECT_ROOT / 'data' / 'chicago',
]

OUTPUT_DIR = PROJECT_ROOT / 'out' / 'compound_test_124m'

# 124M — GPT-2 medium scale. Only this changes vs 30M script.
MODEL_CONFIG = dict(
    n_layer=12, n_head=12, n_embd=768,
    block_size=256, dropout=0.10, bias=True, tie_weights=True,
)

TRAIN_CONFIG = dict(
    batch_size=8, max_iters=4800, eval_interval=200, eval_iters=50,
    learning_rate=1e-4, weight_decay=0.10, warmup_iters=100, min_lr=6e-5,
    grad_clip=1.0,
)

SIGMA = 4.0  # confirmed best from bitnet_compound_extended.py


# ═══════════════════════════════════════════════════════════════════════════
# BITNET QUANTIZATION (identical to bitnet_compound_test.py)
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
    print(f"  BitNet: replaced {replaced} Linear layers with BitLinear (ternary weights)")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# DATA & TRAINING (identical to bitnet_compound_test.py)
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
    print("ERROR: Cannot find training data. Searched:")
    print(f"  {DATA_DIR}")
    for d in FALLBACK_DATA_DIRS:
        print(f"  {d}")
    print("\nPlease edit DATA_DIR at the top of this script.")
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

    # Final eval with 2x eval_iters for accuracy
    final = estimate_loss(model, train_data, val_data, {**tc, 'eval_iters': tc['eval_iters'] * 2}, device)
    final_ppl = math.exp(final['val']) if final['val'] < 20 else float('inf')
    total_time = time.time() - t0
    print(f"  ✓ {run_name} done in {total_time:.1f}s — val_loss={final['val']:.4f}, ppl={final_ppl:.2f}")

    metrics = {'val_loss': final['val'], 'train_loss': final['train'], 'perplexity': final_ppl, 'time': total_time}
    return model, metrics, history


# ═══════════════════════════════════════════════════════════════════════════
# WEIGHT STATISTICS (identical to bitnet_compound_test.py)
# ═══════════════════════════════════════════════════════════════════════════

def weight_stats(model, run_name):
    stats = {}
    for name, param in model.named_parameters():
        if 'weight' in name and ('c_attn' in name or 'c_fc' in name):
            w = param.data.cpu()
            unique = torch.unique(w).tolist()
            stats[name] = {
                'mean': w.mean().item(),
                'std': w.std().item(),
                'min': w.min().item(),
                'max': w.max().item(),
                'num_unique': len(unique),
                'sample_unique': unique[:10] if len(unique) <= 10 else "too many",
            }
            break
    if stats:
        name, s = list(stats.items())[0]
        if s['num_unique'] <= 5:
            print(f"  [{run_name}] Weight check: {s['num_unique']} unique values — TERNARY ✓")
        else:
            print(f"  [{run_name}] Weight check: {s['num_unique']} unique values — FULL PRECISION")
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# REPORT & PLOT (identical structure to bitnet_compound_test.py)
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(all_results, output_path):
    baseline = all_results['baseline']

    lines = []
    lines.append("=" * 80)
    lines.append("COMPOUND ERROR TEST: BitNet + Gaussian Kernel Attention — 124M scale")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Thesis: Scalable and Secure Multi-Tenant LLM Deployment — RQ4")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Model: {MODEL_CONFIG['n_layer']}L / {MODEL_CONFIG['n_head']}H / {MODEL_CONFIG['n_embd']}d")
    lines.append(f"Training: {TRAIN_CONFIG['max_iters']} iterations, batch_size={TRAIN_CONFIG['batch_size']}")
    lines.append("")

    lines.append("-" * 90)
    lines.append(f"{'Run':<30} | {'Val Loss':>10} | {'Perplexity':>12} | {'Time (s)':>10} | {'Δ Ppl':>10}")
    lines.append("-" * 90)

    for name, r in all_results.items():
        delta = (r['perplexity'] - baseline['perplexity']) / baseline['perplexity'] * 100
        delta_str = "base" if name == 'baseline' else f"+{delta:.1f}%"
        lines.append(f"{name:<30} | {r['val_loss']:>10.4f} | {r['perplexity']:>12.2f} | {r['time']:>10.1f} | {delta_str:>10}")

    lines.append("-" * 90)
    lines.append("")

    gaussian_delta  = (all_results['gaussian_only']['perplexity']  - baseline['perplexity']) / baseline['perplexity'] * 100
    bitnet_delta    = (all_results['bitnet_only']['perplexity']     - baseline['perplexity']) / baseline['perplexity'] * 100
    compound_delta  = (all_results['bitnet_gaussian']['perplexity'] - baseline['perplexity']) / baseline['perplexity'] * 100
    additive_expected = gaussian_delta + bitnet_delta

    lines.append("COMPOUND ERROR ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"  Gaussian only:           +{gaussian_delta:.1f}%")
    lines.append(f"  BitNet only:             +{bitnet_delta:.1f}%")
    lines.append(f"  Expected (additive):     +{additive_expected:.1f}%")
    lines.append(f"  Actual (compound):       +{compound_delta:.1f}%")
    lines.append(f"  Interaction effect:      {compound_delta - additive_expected:+.1f}%")
    lines.append("")
    lines.append("30M reference (from thesis):")
    lines.append("  Softmax baseline: ppl 1.54 @ 2400 iters")
    lines.append("  BitNet only:      ppl 1.48 @ 2400 iters")
    lines.append("  BitNet+Gaussian:  ppl 1.48 @ 2400 iters (+0% additional cost)")
    lines.append("")

    if compound_delta < 5.0:
        lines.append("✅ COMPOUND DEGRADATION < 5% — ACCEPTABLE")
    elif compound_delta < 10.0:
        lines.append("⚠️  COMPOUND DEGRADATION 5-10% — MARGINAL, may need tuning")
    else:
        lines.append("❌ COMPOUND DEGRADATION > 10% — needs investigation")

    lines.append("")
    if compound_delta < additive_expected:
        lines.append("INTERACTION: Sub-additive (better than expected)")
    elif compound_delta > additive_expected * 1.5:
        lines.append("INTERACTION: Super-additive (errors compound)")
    else:
        lines.append("INTERACTION: Roughly additive (errors are independent)")

    lines.append("")
    lines.append("=" * 80)

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

        colors = {
            'baseline':       '#2196F3',
            'gaussian_only':  '#4CAF50',
            'bitnet_only':    '#FF9800',
            'bitnet_gaussian':'#F44336',
        }
        labels = {
            'baseline':        f'Baseline (softmax + FP)',
            'gaussian_only':   f'Gaussian σ={SIGMA} only',
            'bitnet_only':     'BitNet only',
            'bitnet_gaussian': f'BitNet + Gaussian σ={SIGMA} (compound)',
        }

        for name, history in all_histories.items():
            iters      = [h['iter']       for h in history]
            val_losses = [h['val_loss']   for h in history]
            ppls       = [h['ppl']        for h in history]

            ax1.plot(iters, val_losses, color=colors[name], label=labels[name], linewidth=2)
            ax2.plot(iters, ppls,       color=colors[name], label=labels[name], linewidth=2)

        ax1.set_xlabel('Iteration'); ax1.set_ylabel('Validation Loss')
        ax1.set_title('Convergence — Val Loss'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Iteration'); ax2.set_ylabel('Perplexity')
        ax2.set_title('Convergence — Perplexity'); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

        plt.suptitle('Compound Error Test: BitNet + Gaussian — 124M model', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot saved: {output_path}")
    except ImportError:
        print("⚠ matplotlib not available, skipping plot")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN (identical structure to bitnet_compound_test.py)
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 80)
    print("COMPOUND ERROR TEST: BitNet + Gaussian Kernel — 124M scale")
    print("Testing 4 combinations to measure interaction effects")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = find_data_dir()
    print(f"Data: {data_dir}")
    train_data, val_data = load_data(data_dir)

    # Identical vocab_size derivation to original
    vocab_size = max(int(train_data.max()) + 1, 256)
    vocab_size = ((vocab_size + 63) // 64) * 64

    all_results  = {}
    all_histories = {}

    # ── Run 1: Baseline (softmax + FP weights) ───────────────────────────
    print(f"\n{'='*70}")
    print("RUN 1/4: BASELINE (softmax + full precision weights)")
    print(f"{'='*70}")
    torch.manual_seed(42); np.random.seed(42)
    config = GPTConfig(**MODEL_CONFIG, vocab_size=vocab_size, use_gaussian_kernel=False)
    model = GPT(config).to(device)
    model, metrics, history = train_model(model, train_data, val_data, device, "Baseline")
    weight_stats(model, "Baseline")
    all_results['baseline']   = metrics
    all_histories['baseline'] = history
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Run 2: Gaussian only (σ=SIGMA + FP weights) ──────────────────────
    print(f"\n{'='*70}")
    print(f"RUN 2/4: GAUSSIAN σ={SIGMA} ONLY (full precision weights)")
    print(f"{'='*70}")
    torch.manual_seed(42); np.random.seed(42)
    config = GPTConfig(**MODEL_CONFIG, vocab_size=vocab_size,
                       use_gaussian_kernel=True, gaussian_sigma=SIGMA,
                       gaussian_sigma_learnable=False)
    model = GPT(config).to(device)
    model, metrics, history = train_model(model, train_data, val_data, device, f"Gaussian σ={SIGMA}")
    weight_stats(model, f"Gaussian σ={SIGMA}")
    all_results['gaussian_only']   = metrics
    all_histories['gaussian_only'] = history
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Run 3: BitNet only (softmax + ternary weights) ────────────────────
    print(f"\n{'='*70}")
    print("RUN 3/4: BITNET ONLY (softmax + ternary weights)")
    print(f"{'='*70}")
    torch.manual_seed(42); np.random.seed(42)
    config = GPTConfig(**MODEL_CONFIG, vocab_size=vocab_size, use_gaussian_kernel=False)
    model = GPT(config).to(device)
    model = apply_bitnet_to_model(model)
    model, metrics, history = train_model(model, train_data, val_data, device, "BitNet only")
    all_results['bitnet_only']   = metrics
    all_histories['bitnet_only'] = history
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Run 4: BitNet + Gaussian (THE COMPOUND TEST) ──────────────────────
    print(f"\n{'='*70}")
    print(f"RUN 4/4: BITNET + GAUSSIAN σ={SIGMA} (COMPOUND TEST)")
    print(f"{'='*70}")
    torch.manual_seed(42); np.random.seed(42)
    config = GPTConfig(**MODEL_CONFIG, vocab_size=vocab_size,
                       use_gaussian_kernel=True, gaussian_sigma=SIGMA,
                       gaussian_sigma_learnable=False)
    model = GPT(config).to(device)
    model = apply_bitnet_to_model(model)
    model, metrics, history = train_model(model, train_data, val_data, device, f"BitNet+Gaussian σ={SIGMA}")
    all_results['bitnet_gaussian']   = metrics
    all_histories['bitnet_gaussian'] = history
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Save ──────────────────────────────────────────────────────────────
    results_path = OUTPUT_DIR / 'compound_results.json'
    with open(results_path, 'w') as f:
        json.dump({'results': all_results, 'histories': all_histories,
                   'model_config': MODEL_CONFIG, 'train_config': TRAIN_CONFIG}, f, indent=2)
    print(f"\n✓ Results saved: {results_path}")

    generate_report(all_results,  OUTPUT_DIR / 'compound_report.txt')
    generate_plot(all_histories,  OUTPUT_DIR / 'compound_plot.png')

    print(f"\n{'='*80}")
    print("DONE — All outputs in:", OUTPUT_DIR)
    print(f"{'='*80}")


if __name__ == "__main__":
    main()