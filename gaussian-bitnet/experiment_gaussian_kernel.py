#!/usr/bin/env python3
"""
experiment_gaussian_kernel.py
=============================
Gaussian Kernel Attention Experiment for Luka's Thesis (RQ4)


This script:
  1. Trains a baseline model with standard softmax attention
  2. Trains an identical model with Gaussian kernel attention
  3. Compares validation loss, perplexity, training time, and convergence
  4. Outputs a comparison table + loss curve plot for professor

Usage:
  python experiment_gaussian_kernel.py

Requirements:
  - model.py (modified version with Gaussian kernel support)
  - Chicago dataset prepared at data/chicago/bin/
  - Or any nanoGPT-compatible dataset

Outputs:
  - gaussian_experiment_results.json   (raw numbers)
  - gaussian_experiment_plot.png        (convergence curves)
  - gaussian_experiment_report.txt      (text summary for professor)
"""

import os
import sys
import math
import time
import json
import numpy as np
import torch
from pathlib import Path

# ── Adjust this if your project structure differs ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from model import GPTConfig, GPT


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these to match your setup
# ═══════════════════════════════════════════════════════════════════════════

# Data paths (adjust to your project)
DATA_DIR = Path('/home/coder/project/data/chicago/bin')
# Check both chicago/ and chicago/bin/ for the bin files
if (DATA_DIR / 'train.bin').exists():
    TRAIN_BIN = DATA_DIR / 'train.bin'
    VAL_BIN   = DATA_DIR / 'val.bin'
elif (DATA_DIR / 'bin' / 'train.bin').exists():
    TRAIN_BIN = DATA_DIR / 'bin' / 'train.bin'
    VAL_BIN   = DATA_DIR / 'bin' / 'val.bin'
else:
    TRAIN_BIN = DATA_DIR / 'train.bin'  # fallback, will error with helpful message
    VAL_BIN   = DATA_DIR / 'val.bin'

# Model config (matches your train_base.py)
MODEL_CONFIG = dict(
    n_layer=6,
    n_head=8,
    n_embd=768,
    block_size=256,
    dropout=0.10,
    bias=True,
    tie_weights=True,
)

# Training config
TRAIN_CONFIG = dict(
    batch_size=8,
    max_iters=1200,        # Same as your train_base.py smoke run
    eval_interval=100,     # Evaluate every N steps (more granular for comparison)
    eval_iters=50,         # Batches per evaluation
    learning_rate=1e-4,
    weight_decay=0.10,
    warmup_iters=100,
    min_lr=6e-5,
    grad_clip=1.0,
)

# Gaussian kernel variants to test
SIGMA_VARIANTS = [
    # (name, sigma_value, learnable)
    ("sqrt_d",    0.0,   False),   # Auto = sqrt(head_dim) = sqrt(96) ≈ 9.8
    ("sigma_1",   1.0,   False),   # Fixed sigma = 1.0
    ("sigma_5",   5.0,   False),   # Fixed sigma = 5.0
    ("learnable", 0.0,   True),    # Learnable sigma, init = sqrt(head_dim)
]

# Output
OUTPUT_DIR = PROJECT_ROOT / 'out' / 'gaussian_experiment'


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(train_path, val_path):
    """Load memory-mapped training and validation data."""
    print(f"Loading data...")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")

    train_data = np.memmap(str(train_path), dtype=np.uint16, mode='r')
    val_data   = np.memmap(str(val_path),   dtype=np.uint16, mode='r')

    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens:   {len(val_data):,}")
    return train_data, val_data


def get_batch(data, batch_size, block_size, device):
    """Sample a random batch from data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix
    ])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device):
    """Estimate train and val loss over eval_iters batches."""
    model.eval()
    results = {}
    for split_name, data in [('train', train_data), ('val', val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, targets=y)
            losses.append(loss.item())
        results[split_name] = np.mean(losses)
    model.train()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════

def get_lr(it, warmup_iters, max_iters, learning_rate, min_lr):
    """Cosine decay with linear warmup (matches nanoGPT)."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train_model(config, train_data, val_data, device, run_name="model"):
    """
    Train a model and return metrics history.

    Returns:
        dict with keys: final_train_loss, final_val_loss, final_perplexity,
                        history (list of {iter, train_loss, val_loss}),
                        total_time_sec, config_summary
    """
    tc = TRAIN_CONFIG

    print(f"\n{'='*70}")
    print(f"TRAINING: {run_name}")
    print(f"{'='*70}")

    # Detect vocab size from data
    vocab_size = int(train_data.max()) + 1
    vocab_size = max(vocab_size, 256)  # At minimum 256 for char-level
    # Round up to nearest multiple of 64 for efficiency
    vocab_size = ((vocab_size + 63) // 64) * 64
    config.vocab_size = vocab_size

    model = GPT(config)
    model = model.to(device)

    optimizer = model.configure_optimizers(
        weight_decay=tc['weight_decay'],
        learning_rate=tc['learning_rate'],
        betas=(0.9, 0.95),
        device_type=str(device),
    )

    history = []
    best_val_loss = float('inf')
    t_start = time.time()

    model.train()
    for it in range(tc['max_iters']):
        # LR schedule
        lr = get_lr(it, tc['warmup_iters'], tc['max_iters'],
                     tc['learning_rate'], tc['min_lr'])
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Evaluate periodically
        if it % tc['eval_interval'] == 0 or it == tc['max_iters'] - 1:
            losses = estimate_loss(
                model, train_data, val_data,
                tc['eval_iters'], tc['batch_size'], config.block_size, device,
            )
            train_loss = losses['train']
            val_loss = losses['val']
            perplexity = math.exp(val_loss) if val_loss < 20 else float('inf')

            history.append({
                'iter': it,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'perplexity': perplexity,
                'lr': lr,
            })

            marker = " *" if val_loss < best_val_loss else ""
            best_val_loss = min(best_val_loss, val_loss)
            elapsed = time.time() - t_start
            print(f"  iter {it:5d} | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| ppl {perplexity:8.2f} | lr {lr:.2e} | {elapsed:.0f}s{marker}")

            # Log sigma if Gaussian and learnable
            if config.use_gaussian_kernel and config.gaussian_sigma_learnable:
                sigmas = [
                    block.attn.sigma.mean().item()
                    for block in model.transformer.h
                ]
                print(f"         sigma (mean across layers): {np.mean(sigmas):.4f}")

        # Forward + backward
        x, y = get_batch(train_data, tc['batch_size'], config.block_size, device)
        _, loss = model(x, targets=y)
        loss.backward()

        # Gradient clipping
        if tc['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc['grad_clip'])

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    total_time = time.time() - t_start

    # Final evaluation
    final_losses = estimate_loss(
        model, train_data, val_data,
        tc['eval_iters'] * 2, tc['batch_size'], config.block_size, device,
    )

    result = {
        'run_name': run_name,
        'final_train_loss': final_losses['train'],
        'final_val_loss': final_losses['val'],
        'final_perplexity': math.exp(final_losses['val']) if final_losses['val'] < 20 else float('inf'),
        'best_val_loss': best_val_loss,
        'best_perplexity': math.exp(best_val_loss) if best_val_loss < 20 else float('inf'),
        'total_time_sec': total_time,
        'history': history,
        'config': config.to_dict(),
    }

    print(f"\n  ✓ {run_name} complete in {total_time:.1f}s")
    print(f"    Final val loss: {result['final_val_loss']:.4f}")
    print(f"    Final perplexity: {result['final_perplexity']:.2f}")
    print(f"    Best val loss: {result['best_val_loss']:.4f}")

    # Clean up GPU memory
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def create_comparison_plot(all_results, output_path):
    """Create convergence comparison plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

    # Plot 1: Validation Loss
    ax = axes[0]
    for i, result in enumerate(all_results):
        iters = [h['iter'] for h in result['history']]
        vals  = [h['val_loss'] for h in result['history']]
        color = colors[i % len(colors)]
        ax.plot(iters, vals, '-o', color=color, label=result['run_name'],
                markersize=4, linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Perplexity
    ax = axes[1]
    for i, result in enumerate(all_results):
        iters = [h['iter'] for h in result['history']]
        perps = [h['perplexity'] for h in result['history']]
        # Cap perplexity for plot readability
        perps = [min(p, 5000) for p in perps]
        color = colors[i % len(colors)]
        ax.plot(iters, perps, '-o', color=color, label=result['run_name'],
                markersize=4, linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Perplexity Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(all_results, output_path):
    """Generate text report for professor."""
    baseline = all_results[0]  # First result is always softmax baseline

    lines = []
    lines.append("=" * 80)
    lines.append("GAUSSIAN KERNEL ATTENTION EXPERIMENT REPORT")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("OBJECTIVE: Test if Gaussian kernel attention can replace softmax")
    lines.append("           in our GPT model without significant quality degradation.")
    lines.append("           (Prerequisite for encrypted inference — RQ4)")
    lines.append("")
    lines.append(f"Model: {MODEL_CONFIG['n_layer']}L / {MODEL_CONFIG['n_head']}H / {MODEL_CONFIG['n_embd']}d")
    lines.append(f"Training: {TRAIN_CONFIG['max_iters']} iterations, batch_size={TRAIN_CONFIG['batch_size']}")
    lines.append(f"Block size: {MODEL_CONFIG['block_size']}")
    lines.append("")

    # Results table
    lines.append("-" * 80)
    header = f"{'Run':<30} | {'Val Loss':>10} | {'Perplexity':>12} | {'Time (s)':>10} | {'Δ Ppl':>8}"
    lines.append(header)
    lines.append("-" * 80)

    baseline_ppl = baseline['final_perplexity']
    for r in all_results:
        ppl = r['final_perplexity']
        delta_ppl = ((ppl - baseline_ppl) / baseline_ppl * 100) if baseline_ppl > 0 else 0
        delta_str = f"{delta_ppl:+.1f}%" if r != baseline else "base"
        lines.append(
            f"{r['run_name']:<30} | {r['final_val_loss']:>10.4f} | "
            f"{ppl:>12.2f} | {r['total_time_sec']:>10.1f} | {delta_str:>8}"
        )

    lines.append("-" * 80)
    lines.append("")

    # Analysis
    lines.append("ANALYSIS")
    lines.append("-" * 40)

    best_gaussian = None
    best_gaussian_delta = float('inf')
    for r in all_results[1:]:
        delta = abs(r['final_perplexity'] - baseline_ppl) / baseline_ppl * 100
        if delta < best_gaussian_delta:
            best_gaussian_delta = delta
            best_gaussian = r

    if best_gaussian:
        lines.append(f"Best Gaussian variant: {best_gaussian['run_name']}")
        lines.append(f"  Perplexity delta vs baseline: {best_gaussian_delta:.1f}%")
        lines.append("")

        if best_gaussian_delta < 5:
            lines.append("✓ RECOMMENDATION: Gaussian kernel is VIABLE (<5% degradation)")
            lines.append("  → Proceed to encryption testing (combine with BitNet + BiMPC)")
        elif best_gaussian_delta < 15:
            lines.append("⚠ RECOMMENDATION: Moderate degradation (5-15%)")
            lines.append("  → May be acceptable; discuss with professor")
            lines.append("  → Consider tuning sigma or training longer")
        else:
            lines.append("✗ RECOMMENDATION: Significant degradation (>15%)")
            lines.append("  → Gaussian kernel may not be suitable for this model size")
            lines.append("  → Investigate: longer training, different sigma, model scaling")

    lines.append("")
    lines.append("=" * 80)

    report_text = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\n{report_text}")
    print(f"\n  ✓ Report saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 80)
    print("GAUSSIAN KERNEL ATTENTION EXPERIMENT")
    print("Thesis: Scalable and Secure Multi-Tenant LLM Deployment")
    print("RQ4: 1-Bit LLM with Partial Homomorphic Encryption")
    print("=" * 80)

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Check data exists
    if not TRAIN_BIN.exists() or not VAL_BIN.exists():
        print(f"\n✗ Data not found at {DATA_DIR}")
        print(f"  Expected: {TRAIN_BIN}")
        print(f"  Expected: {VAL_BIN}")
        print(f"\n  Please adjust DATA_DIR at the top of this script,")
        print(f"  or prepare your dataset first.")
        sys.exit(1)

    train_data, val_data = load_data(TRAIN_BIN, VAL_BIN)

    all_results = []

    # ── 1. Softmax Baseline ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: SOFTMAX BASELINE")
    print("=" * 70)

    config_softmax = GPTConfig(
        **MODEL_CONFIG,
        use_gaussian_kernel=False,
    )

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    result_softmax = train_model(
        config_softmax, train_data, val_data, device,
        run_name="Softmax (baseline)"
    )
    all_results.append(result_softmax)

    # ── 2. Gaussian Kernel Variants ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: GAUSSIAN KERNEL VARIANTS")
    print("=" * 70)

    for variant_name, sigma_val, learnable in SIGMA_VARIANTS:
        torch.manual_seed(42)
        np.random.seed(42)

        config_gaussian = GPTConfig(
            **MODEL_CONFIG,
            use_gaussian_kernel=True,
            gaussian_sigma=sigma_val,
            gaussian_sigma_learnable=learnable,
        )

        run_name = f"Gaussian σ={variant_name}"
        result = train_model(
            config_gaussian, train_data, val_data, device,
            run_name=run_name,
        )
        all_results.append(result)

    # ── 3. Save Results ─────────────────────────────────────────────────────
    results_path = OUTPUT_DIR / 'gaussian_experiment_results.json'
    # Convert numpy types for JSON serialization
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Raw results saved: {results_path}")

    # ── 4. Generate Plot ────────────────────────────────────────────────────
    plot_path = OUTPUT_DIR / 'gaussian_experiment_plot.png'
    create_comparison_plot(all_results, plot_path)

    # ── 5. Generate Report ──────────────────────────────────────────────────
    report_path = OUTPUT_DIR / 'gaussian_experiment_report.txt'
    generate_report(all_results, report_path)

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs in: {OUTPUT_DIR}/")
    print(f"  - gaussian_experiment_results.json  (raw data)")
    print(f"  - gaussian_experiment_plot.png       (convergence curves)")
    print(f"  - gaussian_experiment_report.txt     (summary for professor)")


if __name__ == "__main__":
    main()
