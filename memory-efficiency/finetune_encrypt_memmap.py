#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: finetune_encrypt_memmap_enhanced_with_stats.py
# Project: "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas"
#
# ENHANCED FEATURES (NEW):
# ✅ WandB integration for online loss tracking
# ✅ Statistics collection at EVERY training step:
#    - Number/percentage of pages modified
#    - Per-weight differences
#    - Size of modifications
#    - Layer-wise change analysis
# ✅ CSV and JSON output of all statistics
# ✅ Layer freezing for selective training
#
# EXISTING FEATURES:
# - Validation loss tracking during training
# - Text generation before/after fine-tuning (proof of learning)
# - Training progress monitoring
# - Encrypted delta storage
# - Page-level threshold filtering
# -----------------------------------------------------------------------------------
from __future__ import annotations
import os, sys, argparse, time, json
from pathlib import Path
from typing import Optional, Dict, Set, List, Tuple

import numpy as np
import torch

# --- Setup paths for imports ---
_THIS = Path(__file__).resolve()
sys.path.append(str(_THIS.parent))           # src/
sys.path.append(str(_THIS.parent.parent))    # project root

# --- Import model ---
try:
    from model import GPTConfig, GPT
except ImportError:
    from src.model import GPTConfig, GPT

# --- Import modularized components ---
try:
    # Try direct import (if run from project root)
    from src.crypto_utils import is_crypto_available, derive_key
    from src.delta_serializer import compute_and_store_deltas
    from src.delta_applicator import apply_deltas
    from src.checkpoint_manager import load_checkpoint, save_checkpoint
    from src.data_loader import get_batch
    from src.validation import estimate_loss
    from src.text_generation import generate_text, print_generation_header, print_generation_footer
except ImportError:
    # Fallback if run from src/
    from crypto_utils import is_crypto_available, derive_key
    from delta_serializer import compute_and_store_deltas
    from delta_applicator import apply_deltas
    from checkpoint_manager import load_checkpoint, save_checkpoint
    from data_loader import get_batch
    from validation import estimate_loss
    from text_generation import generate_text, print_generation_header, print_generation_footer

# --- Optional imports ---
try:
    from src.parameter_counter import print_parameter_summary, count_parameters
except ImportError:
    def print_parameter_summary(model): pass
    def count_parameters(model): return {'total': 0, 'trainable': 0, 'non_trainable': 0}

# === NEW: Import StatisticsTracker ===
try:
    from src.statistics_tracker import StatisticsTracker
except ImportError:
    try:
        from statistics_tracker import StatisticsTracker
    except ImportError:
        print("WARNING: StatisticsTracker not found. Statistics collection disabled.")
        StatisticsTracker = None

# === NEW: Import layer freezing ===
try:
    from src.layer_freezing import freeze_bottom_layers, print_freeze_summary
except ImportError:
    try:
        from layer_freezing import freeze_bottom_layers, print_freeze_summary
    except ImportError:
        print("WARNING: layer_freezing not found. Layer freezing disabled.")
        freeze_bottom_layers = None
        print_freeze_summary = None

try:
    from src.delta_tracker import get_page_tracker
except ImportError:
    get_page_tracker = None

try:
    from src.plotting import plot_training_curves, plot_loss_comparison, print_loss_summary
except ImportError:
    def plot_training_curves(*args, **kwargs): return False
    def plot_loss_comparison(*args, **kwargs): return False
    def print_loss_summary(*args, **kwargs): pass

try:
    from src.page_threshold import filter_deltas_by_page_threshold
except ImportError:
    filter_deltas_by_page_threshold = None


# ------------------------- Argument Parsing -------------------------
def str2bool(v: str) -> bool:
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n", ""}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune GPT with encrypted delta storage + statistics tracking")
    
    # Model and checkpoint
    parser.add_argument("--base_ckpt", type=str, default="out/checkpoints/base_final.pt",
                        help="Path to base checkpoint")
    parser.add_argument("--out_dir", type=str, default="out/deltas",
                        help="Output directory for deltas")
    
    # Data
    parser.add_argument("--dataset", type=str, default="chicago",
                        help="Dataset name (chicago or openwebtext_toy)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Evaluate validation loss every N iterations")
    parser.add_argument("--eval_iters", type=int, default=5,
                        help="Number of batches to use for validation")
    
    # Training hyperparameters
    parser.add_argument("--max_iters", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping")
    
    # Layer freezing (selective training)
    parser.add_argument("--freeze_ratio", type=float, default=0.0,
                        help="Freeze bottom N%% of transformer layers (0.0-1.0). E.g., 0.5 = freeze bottom 50%%")
    
    # === NEW: Statistics tracking ===
    parser.add_argument("--enable_stats", type=str2bool, default=True,
                        help="Enable detailed statistics tracking")
    parser.add_argument("--stats_dir", type=str, default="training_stats",
                        help="Directory for statistics output")
    parser.add_argument("--enable_wandb", type=str2bool, default=False,
                        help="Enable Weights & Biases online tracking")
    parser.add_argument("--wandb_project", type=str, default="llm-finetuning",
                        help="WandB project name")
    
    # Delta storage
    parser.add_argument("--delta_file", type=str, default="out/deltas/deltas_chicago.bin",
                        help="Path to delta file")
    parser.add_argument("--delta_store", type=str, default="file",
                        choices=["ram", "file", "shared"],
                        help="Storage mode: ram, file, or shared")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Skip deltas where max(abs(delta)) < threshold")
    
    # Encryption
    parser.add_argument("--encrypt", type=str2bool, default=False,
                        help="Enable AES-CBC+HMAC encryption")
    parser.add_argument("--key_hex", type=str, default=None,
                        help="Hex-encoded 32-byte encryption key")
    
    # Page tracking (optional)
    parser.add_argument("--track_pages", type=str2bool, default=False,
                        help="Enable page-level dirty tracking")
    parser.add_argument("--tracker_backend", type=str, default="hash",
                        choices=["hash", "mprotect"],
                        help="Page tracker backend")
    parser.add_argument("--page_size", type=int, default=4096,
                        help="Page size for tracking")
    
    # Page-level threshold filtering
    parser.add_argument("--use_page_threshold", type=str2bool, default=False,
                        help="Enable page-level threshold filtering to reduce storage")
    parser.add_argument("--page_threshold", type=float, default=1e-6,
                        help="Skip pages where max(|delta|) < threshold")
    
    # Text generation
    parser.add_argument("--generate_samples", type=str2bool, default=True,
                        help="Generate text samples before/after training")
    parser.add_argument("--generation_prompt", type=str, default="The city of Chicago",
                        help="Prompt for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=200,
                        help="Top-k sampling")
    
    # Testing/debug
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Test prompt file for generation")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    
    return parser.parse_args()


# ------------------------- Training Loop (ENHANCED WITH STATISTICS) -------------------------
def train_model(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: np.memmap,
    val_data: Optional[np.memmap],
    args,
    device: torch.device,
    tracker=None,
    handle_by_name: Dict = None,
    stats_tracker: Optional[StatisticsTracker] = None  # NEW
) -> Tuple[Set, List[Tuple[int, float, float]]]:
    """
    Main training loop with validation tracking AND statistics collection
    
    Args:
        model: GPT model to train
        optimizer: Optimizer
        train_data: Memory-mapped training data
        val_data: Memory-mapped validation data (optional)
        args: Command-line arguments
        device: torch device
        tracker: Optional page tracker
        handle_by_name: Optional dict of tracker handles
        stats_tracker: Optional StatisticsTracker for detailed metrics (NEW)
    
    Returns:
        Tuple of (dirty parameters, training history)
        Training history: List of (iter, train_loss, val_loss)
    """
    model.train()
    
    # Track which parameters change
    dirty = set()
    
    # Track training progress
    history = []
    
    # Register hooks to detect modifications
    def hook(module, grad_input, grad_output):
        for name, param in module.named_parameters(recurse=False):
            full_name = name
            for n, m in model.named_modules():
                if m is module:
                    full_name = f"{n}.{name}" if n else name
                    break
            dirty.add(full_name)
    
    # Attach hooks
    for module in model.modules():
        module.register_full_backward_hook(hook)
    
    print(f"\n{'='*70}")
    print("STARTING FINE-TUNING WITH VALIDATION TRACKING + STATISTICS")
    print(f"{'='*70}")
    print(f"Iterations: {args.max_iters}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Eval interval: {args.eval_interval}")
    print(f"Device: {device}")
    print(f"Statistics tracking: {'ENABLED' if stats_tracker else 'DISABLED'}")
    if stats_tracker and stats_tracker.wandb_enabled:
        print(f"WandB tracking: ENABLED (project: {args.wandb_project})")
    print(f"{'='*70}\n")
    
    # Training loop
    start_time = time.time()
    
    for iter_num in range(args.max_iters):
        # === NEW: Save model state BEFORE optimizer step ===
        if stats_tracker is not None:
            model_state_before = {
                name: param.data.clone()
                for name, param in model.named_parameters()
            }
        
        # Get batch
        X, Y = get_batch(
            train_data,
            block_size=model.config.block_size,
            batch_size=args.batch_size,
            device=device
        )
        
        # Forward pass
        logits, loss = model(X, Y)
        train_loss = loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Update weights
        optimizer.step()
        
        # === NEW: Save model state AFTER optimizer step and collect statistics ===
        val_loss = None
        if stats_tracker is not None:
            model_state_after = {
                name: param.data.clone()
                for name, param in model.named_parameters()
            }
            
            # Compute validation loss if it's time
            if iter_num % args.eval_interval == 0 and val_data is not None:
                val_loss = estimate_loss(
                    model, val_data, args.eval_iters, args.batch_size, device
                )
            
            # Collect statistics for this step
            stats = stats_tracker.collect_step_statistics(
                iteration=iter_num,
                model_state_before=model_state_before,
                model_state_after=model_state_after,
                train_loss=train_loss,
                val_loss=val_loss
            )
            
            # Print progress with statistics
            if iter_num % args.eval_interval == 0:
                if val_loss is not None:
                    print(f"iter {iter_num:4d}/{args.max_iters}: "
                          f"train={train_loss:.4f}, val={val_loss:.4f}, "
                          f"modified_params={stats['modified_parameters']:,} "
                          f"({stats['pct_parameters_modified']:.2f}%), "
                          f"pages={stats['modified_pages']:,} "
                          f"({stats['pct_pages_modified']:.2f}%)")
                    history.append((iter_num, train_loss, val_loss))
                else:
                    print(f"iter {iter_num:4d}/{args.max_iters}: "
                          f"loss={train_loss:.4f}, "
                          f"modified_params={stats['modified_parameters']:,} "
                          f"({stats['pct_parameters_modified']:.2f}%)")
                    history.append((iter_num, train_loss, None))
            elif iter_num % 10 == 0:
                print(f"iter {iter_num:4d}/{args.max_iters}: "
                      f"loss={train_loss:.4f}, "
                      f"modified={stats['modified_parameters']:,}")
        else:
            # Original behavior without statistics tracking
            if iter_num % args.eval_interval == 0:
                if val_data is not None:
                    val_loss = estimate_loss(
                        model, val_data, args.eval_iters, args.batch_size, device
                    )
                    history.append((iter_num, train_loss, val_loss))
                    print(f"iter {iter_num:4d}/{args.max_iters}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
                else:
                    history.append((iter_num, train_loss, None))
                    print(f"iter {iter_num:4d}/{args.max_iters}: train loss {train_loss:.4f}")
            elif iter_num % 10 == 0:
                print(f"iter {iter_num:4d}/{args.max_iters}: loss {train_loss:.4f}")
    
    end_time = time.time()
    print(f"\n⏱  Fine-tuning time: {end_time - start_time:.2f} seconds")
    
    return dirty, history


# ------------------------- Main Function (ENHANCED) -------------------------
def main():
    args = parse_args()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print("FINE-TUNING WITH VALIDATION, TEXT GENERATION, AND STATISTICS")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Base checkpoint: {args.base_ckpt}")
    print(f"Dataset: {args.dataset}")
    print(f"Encryption: {args.encrypt}")
    print(f"Delta storage: {args.delta_store}")
    print(f"Text generation: {args.generate_samples}")
    print(f"Statistics tracking: {args.enable_stats}")
    if args.enable_stats:
        print(f"WandB: {args.enable_wandb}")
    print(f"Layer freezing: {args.freeze_ratio*100:.0f}% of bottom layers" if args.freeze_ratio > 0 else "Disabled")
    print(f"{'='*70}\n")
    
    # === NEW: Initialize Statistics Tracker ===
    stats_tracker = None
    if args.enable_stats and StatisticsTracker is not None:
        stats_tracker = StatisticsTracker(
            output_dir=args.stats_dir,
            page_size=args.page_size,
            enable_wandb=args.enable_wandb
        )
        
        if args.enable_wandb:
            stats_tracker.initialize_wandb(
                config={
                    'model': 'GPT',
                    'dataset': args.dataset,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'weight_decay': args.weight_decay,
                    'max_iters': args.max_iters,
                    'eval_interval': args.eval_interval,
                    'page_size': args.page_size,
                    'encrypt': args.encrypt,
                    'threshold': args.threshold,
                    'freeze_ratio': args.freeze_ratio
                },
                project_name=args.wandb_project
            )
        
        print(f"✓ Statistics tracker initialized (output: {args.stats_dir})")
    elif args.enable_stats and StatisticsTracker is None:
        print("⚠  Statistics tracking requested but StatisticsTracker not available")
    
    # Check encryption setup
    if args.encrypt:
        if not is_crypto_available():
            print("❌ Encryption requested but pycryptodome not installed")
            print("Install with: pip install pycryptodome")
            sys.exit(1)
        
        if args.key_hex:
            base_key = bytes.fromhex(args.key_hex)
            if len(base_key) != 32:
                print("❌ Key must be 32 bytes (64 hex chars)")
                sys.exit(1)
        else:
            base_key = os.urandom(32)
            print(f"Generated key: {base_key.hex()}")
    else:
        base_key = None
    
    # Load base checkpoint
    print("Loading base checkpoint...")
    ckpt = torch.load(args.base_ckpt, map_location='cpu')
    
    # Create model from config
    if 'model_args' not in ckpt:
        print("❌ Checkpoint missing 'model_args'")
        sys.exit(1)
    
    config = GPTConfig(**ckpt['model_args'])
    model = GPT(config)
    
    # Load state dict
    state_dict = ckpt.get('model_state_dict') or ckpt.get('model')
    if state_dict is None:
        print("❌ Checkpoint missing model weights")
        sys.exit(1)
    
    # Normalize keys (remove _orig_mod., module. prefixes)
    normalized = {}
    for k, v in state_dict.items():
        nk = k.replace('_orig_mod.', '').replace('module.', '')
        normalized[nk] = v
    
    model.load_state_dict(normalized, strict=False)
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Show parameter summary
    param_counts = count_parameters(model)
    if param_counts['total'] > 0:
        print_parameter_summary(model)
    
    # Move to device BEFORE generating baseline text
    model = model.to(device)
    
    # Generate text BEFORE fine-tuning (baseline)
    if args.generate_samples:
        print_generation_header(
            "TEXT GENERATION - BEFORE FINE-TUNING (BASELINE)",
            args.generation_prompt
        )
        
        baseline_text = generate_text(
            model,
            prompt=args.generation_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        print(baseline_text)
        print_generation_footer()
    
    # Save original state
    orig_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    # ============================================================
    # === LAYER FREEZING (BEFORE OPTIMIZER CREATION) ===
    # ============================================================
    freeze_stats = None
    if args.freeze_ratio > 0.0:
        if freeze_bottom_layers is not None:
            print(f"\n{'='*70}")
            print(f"APPLYING SELECTIVE LAYER TRAINING")
            print(f"{'='*70}")
            freeze_stats = freeze_bottom_layers(model, freeze_ratio=args.freeze_ratio)
            print_freeze_summary(freeze_stats)
        else:
            print(f"⚠  Layer freezing requested but layer_freezing module not available")
            print(f"   Run: cp /mnt/user-data/outputs/layer_freezing.py .")
    else:
        print("\nℹ  No layer freezing (training all layers)")
        print("   Try --freeze_ratio 0.5 for 50% reduction")
    
    # Setup optimizer (AFTER freezing so it only tracks trainable params)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        fused=True  
        )
    
    # Load training data
    data_path = Path(args.data_dir) / args.dataset / "bin" / "train.bin"
    if not data_path.exists():
        print(f"❌ Training data not found: {data_path}")
        sys.exit(1)
    
    train_data = np.memmap(data_path, dtype=np.uint16, mode='r')
    print(f"✓ Training data loaded: {len(train_data):,} tokens")
    
    # Load validation data
    val_path = Path(args.data_dir) / args.dataset / "bin" / "val.bin"
    if val_path.exists():
        val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
        print(f"✓ Validation data loaded: {len(val_data):,} tokens")
    else:
        val_data = None
        print("⚠  No validation data found, skipping validation loss tracking")
    
    # Optional: Setup page tracker
    tracker = None
    handle_by_name = {}
    
    if args.track_pages and get_page_tracker is not None:
        print(f"\nEnabling page tracker (backend: {args.tracker_backend}, page_size: {args.page_size})")
        tracker = get_page_tracker(backend=args.tracker_backend, page_size=args.page_size)
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            ptr = param.data.data_ptr()
            nbytes = param.numel() * param.element_size()
            h = tracker.watch_buffer(ptr, nbytes, name=name)
            handle_by_name[name] = h
        
        print(f"✓ Tracking {len(handle_by_name)} parameters")
    
    # === Train model with statistics tracking ===
    dirty, history = train_model(
        model, optimizer, train_data, val_data, args, device,
        tracker, handle_by_name,
        stats_tracker=stats_tracker
    )
    
    # === NEW: Save and analyze statistics ===
    if stats_tracker is not None:
        print(f"\n{'='*70}")
        print("STATISTICS SUMMARY")
        print(f"{'='*70}")
        
        # Save complete summary
        stats_tracker.save_summary()
        
        # Print delta size breakdown
        stats_tracker.print_delta_size_breakdown()
        
        # Get top changing layers
        top_layers = stats_tracker.get_top_changing_layers(top_k=10)
        print(f"\nTop 10 Most Changed Layers:")
        for i, (layer_name, avg_change) in enumerate(top_layers, 1):
            print(f"{i:2d}. {layer_name:50s} L2={avg_change:.6f}")
        
        # Finalize WandB
        stats_tracker.finish()
        print(f"✓ Statistics tracking complete")
    
    # Print training summary
    if history:
        print_loss_summary(history)
        
        # Generate plots
        plot_training_curves(
            history,
            output_path="out/training_curves.png",
            title=f"Fine-tuning Progress - {args.dataset}"
        )
        
        plot_loss_comparison(
            history,
            output_path="out/loss_comparison.png"
        )
    
    # Generate text AFTER fine-tuning
    if args.generate_samples:
        print_generation_header(
            "TEXT GENERATION - AFTER FINE-TUNING (IMPROVED)",
            args.generation_prompt
        )
        
        finetuned_text = generate_text(
            model,
            prompt=args.generation_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        print(finetuned_text)
        print_generation_footer()
    
    # Fallback diff pass (catch anything hooks missed)
    new_state_now = model.state_dict()
    for name, tensor in new_state_now.items():
        if name not in orig_state:
            continue
        if name not in dirty:
            a = orig_state[name]
            b = tensor.detach().cpu()
            if args.threshold <= 0.0:
                if not torch.equal(a, b):
                    dirty.add(name)
            else:
                if torch.max(torch.abs(a - b)).item() > args.threshold:
                    dirty.add(name)
    
    # Log page-level dirty info (optional)
    dirty_log: Dict[str, dict] = {}
    if tracker is not None:
        for name, param in model.named_parameters():
            h = handle_by_name.get(name)
            if h is None:
                continue
            try:
                pages = sorted(tracker.dirty_pages(h))
                spans = tracker.dirty_spans(h)
                if pages:
                    dirty_log[name] = {"pages": pages, "spans": spans}
                    print(f"[tracker] {name}: dirty pages: {len(pages)}")
            except Exception as e:
                print(f"[tracker] query failed for {name}: {e}")
    
    # Apply page-level threshold filtering
    filtered_state = None
    page_filter_stats = {}
    
    if args.use_page_threshold and filter_deltas_by_page_threshold is not None:
        print(f"\nApplying page-level threshold filtering (threshold={args.page_threshold:.2e})...")
        
        filtered_deltas, page_filter_stats = filter_deltas_by_page_threshold(
            orig_state=orig_state,
            new_state=model.state_dict(),
            dirty_params=dirty,
            page_size=args.page_size,
            threshold=args.page_threshold
        )
        
        # Create filtered state by applying sparse deltas to original
        filtered_state = {}
        for name in dirty:
            if name in filtered_deltas:
                filtered_state[name] = orig_state[name] + filtered_deltas[name]
            else:
                # Parameter was completely filtered out
                filtered_state[name] = orig_state[name]
        
        print(f"✓ Page filtering complete: {len(filtered_deltas)}/{len(dirty)} parameters kept")
    
    # Compute and store deltas
    print(f"\nComputing {'encrypted' if args.encrypt else 'plaintext'} deltas for {len(dirty)} dirty tensors...")
    file_path = Path(args.delta_file) if args.delta_store in ("file", "shared") else None
    
    # Use filtered state if page threshold was applied, otherwise use original fine-tuned state
    state_to_save = filtered_state if filtered_state is not None else model.state_dict()
    
    mm, deltas, backing, layout = compute_and_store_deltas(
        orig_state=orig_state,
        new_state=state_to_save,
        dirty_params=dirty,
        encrypt=args.encrypt,
        base_key=base_key,
        store=args.delta_store,
        file_path=file_path,
        threshold=args.threshold,
    )
    
    # Emit JSON metadata (includes training history AND freeze stats)
    meta_path = Path("out/deltas.meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    idx = {
        "encrypted": bool(args.encrypt),
        "threshold": float(args.threshold),
        "store_type": args.delta_store,
        "delta_file": str(file_path) if file_path else None,
        "training_history": [
            {"iter": it, "train_loss": tl, "val_loss": vl}
            for it, tl, vl in history
        ],
        "layer_freezing": freeze_stats if freeze_stats else {"enabled": False},
        "page_filtering": {
            "enabled": args.use_page_threshold,
            "threshold": args.page_threshold if args.use_page_threshold else None,
            "page_size": args.page_size if args.use_page_threshold else None,
        } if args.use_page_threshold else None,
        "tensors": {}
    }
    
    for name, (off, ln, shape, dtype) in deltas.items():
        rec = {"off": off, "len": ln, "shape": list(shape), "dtype": dtype}
        if name in dirty_log:
            rec["dirty_pages"] = dirty_log[name]["pages"]
            rec["dirty_spans"] = dirty_log[name]["spans"]
        if name in page_filter_stats:
            rec["page_filtering"] = page_filter_stats[name]
        idx["tensors"][name] = rec
    
    meta_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    
    # Validate encryption (if enabled)
    if args.encrypt and deltas and mm is not None:
        first_name = next(iter(sorted(deltas.keys())))
        off, ln, _, _ = deltas[first_name]
        from crypto_utils import decrypt_blob
        _ = decrypt_blob(mm[off : off + ln], derive_key(first_name, base_key))
        print("✓ Encryption validated")
    
    # Print summary
    print(f"\n{'='*70}")
    print("DELTA SUMMARY")
    print(f"{'='*70}")
    print(f"Total model parameters: {param_counts['total']:,}")
    print(f"Tensors touched by hooks: {len(dirty)}")
    
    # === LAYER FREEZING IMPACT ===
    if freeze_stats:
        print(f"\nLayer Freezing Impact:")
        print(f"  Frozen layers: {freeze_stats['frozen_layers']}/{freeze_stats['total_layers']}")
        print(f"  Trainable params: {freeze_stats['trainable_params_after']:,} ({100 - freeze_stats['reduction_percentage']:.1f}%)")
        print(f"  Frozen params: {freeze_stats['trainable_params_before'] - freeze_stats['trainable_params_after']:,} ({freeze_stats['reduction_percentage']:.1f}%)")
    
    num_effective = len(deltas)
    total_bytes = sum(ln for (_, ln, _, _) in deltas.values())
    print(f"\nDeltas above threshold: {num_effective}")
    print(f"Total delta size: {total_bytes / 1024:.2f} KB ({total_bytes / (1024*1024):.2f} MB)")
    
    # Calculate percentage modified
    if num_effective and param_counts['total'] > 0:
        modified_params = 0
        counted_param_ids = set()
        
        for name in deltas.keys():
            if name in model.state_dict():
                param = model.state_dict()[name]
                param_id = id(param.storage())
                
                if param_id not in counted_param_ids:
                    counted_param_ids.add(param_id)
                    modified_params += param.numel()
        
        percentage_modified = (modified_params / param_counts['total']) * 100
        print(f"Modified parameters: {modified_params:,} ({percentage_modified:.2f}%)")
        
        if freeze_stats:
            expected = freeze_stats['reduction_percentage']
            actual = 100 - percentage_modified
            print(f"  Expected from freezing: {expected:.1f}%, Actual: {actual:.1f}%")
        
        if len(deltas) > len(counted_param_ids):
            num_tied = len(deltas) - len(counted_param_ids)
            print(f"  (Note: {num_tied} tied weight(s) detected)")
    
    print(f"\n✓ Delta file: {file_path}")
    print(f"✓ JSON index: {meta_path}")
    print(f"{'='*70}\n")
    
    # Cleanup
    if backing is not None and args.delta_store == "file":
        mm.flush()
        backing.flush()
        os.fsync(backing.fileno())
        backing.close()
        print(f"✓ Delta file flushed and closed")
    
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU RAM: {peak/1024**3:.2f} GB")


if __name__ == "__main__":
    main()