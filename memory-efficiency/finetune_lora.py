#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: finetune_lora.py
# Project: "Multi-Tier Access Control for LLM Fine-Tuning"
# Purpose: Fine-tune GPT using LoRA (Low-Rank Adaptation) instead of full fine-tuning
#
# DIFFERENCES FROM FULL FINE-TUNING:
# - Only trains small low-rank adapter matrices (8-32 MB vs 180 MB deltas)
# - Base model weights frozen (never modified)
# - Adapters saved as separate small files
# - Much better memory efficiency for multi-user deployment
# -----------------------------------------------------------------------------------
from __future__ import annotations
import os, sys, argparse, time, json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn

# Setup paths
_THIS = Path(__file__).resolve()
sys.path.append(str(_THIS.parent))
sys.path.append(str(_THIS.parent.parent))

# Import model
try:
    from model import GPTConfig, GPT
except ImportError:
    from src.model import GPTConfig, GPT

# Import utilities
try:
    from src.data_loader import get_batch
    from src.validation import estimate_loss
    from src.text_generation import generate_text
except ImportError:
    from data_loader import get_batch
    from validation import estimate_loss
    from text_generation import generate_text

# Import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    print("❌ PEFT library not installed!")
    print("Install with: pip install peft")
    PEFT_AVAILABLE = False
    sys.exit(1)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune GPT using LoRA")
    
    # Model and checkpoint
    parser.add_argument("--base_ckpt", type=str, default="out/checkpoints/base_final.pt",
                        help="Path to base checkpoint")
    parser.add_argument("--out_dir", type=str, default="out/lora_adapters",
                        help="Output directory for LoRA adapters")
    
    # Data
    parser.add_argument("--dataset", type=str, default="chicago",
                        help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Evaluate every N iterations")
    parser.add_argument("--eval_iters", type=int, default=5,
                        help="Number of eval batches")
    
    # Training
    parser.add_argument("--max_iters", type=int, default=100,
                        help="Training iterations")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate (typically same as full fine-tuning)")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (4, 8, 16, 32). Higher = larger adapter but better quality")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha (scaling factor, typically 2*r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="c_attn,c_proj",
                        help="Comma-separated list of modules to apply LoRA (c_attn,c_proj,c_fc)")
    
    # Text generation
    parser.add_argument("--generate_samples", type=bool, default=True,
                        help="Generate text before/after")
    parser.add_argument("--generation_prompt", type=str, default="The city of Chicago",
                        help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Max tokens to generate")
    
    # Testing
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    
    return parser.parse_args()


def print_trainable_parameters(model):
    """Print number of trainable parameters (LoRA only)"""
    trainable_params = 0
    all_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    percentage = 100 * trainable_params / all_params
    
    print(f"\n{'='*70}")
    print("PARAMETER SUMMARY")
    print(f"{'='*70}")
    print(f"Total parameters: {all_params:,}")
    print(f"Trainable (LoRA): {trainable_params:,}")
    print(f"Frozen (base): {all_params - trainable_params:,}")
    print(f"Trainable percentage: {percentage:.4f}%")
    print(f"{'='*70}\n")
    
    return {
        'total': all_params,
        'trainable': trainable_params,
        'frozen': all_params - trainable_params,
        'percentage': percentage
    }


def train_lora(
    model,
    optimizer,
    train_data: np.memmap,
    val_data: Optional[np.memmap],
    args,
    device: torch.device
) -> List[Tuple[int, float, Optional[float]]]:
    """
    Train LoRA adapters
    
    Returns:
        List of (iteration, train_loss, val_loss)
    """
    model.train()
    history = []
    
    print(f"\n{'='*70}")
    print("STARTING LoRA FINE-TUNING")
    print(f"{'='*70}")
    print(f"Iterations: {args.max_iters}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"Target modules: {args.lora_target_modules}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for iter_num in range(args.max_iters):
        # Get batch
        X, Y = get_batch(
            train_data,
            block_size=model.config.block_size,
            batch_size=args.batch_size,
            device=device
        )
        
        # Forward pass
        # Forward pass (use keyword arguments for PEFT compatibility)
        logits, loss = model(input_ids=X, labels=Y)
        train_loss = loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Update weights (only LoRA adapters)
        optimizer.step()
        
        # Evaluation
        if iter_num % args.eval_interval == 0:
            if val_data is not None:
                val_loss = estimate_loss(
                    model, val_data, args.eval_iters, args.batch_size, device
                )
                history.append((iter_num, train_loss, val_loss))
                print(f"iter {iter_num:4d}/{args.max_iters}: "
                      f"train={train_loss:.4f}, val={val_loss:.4f}")
            else:
                history.append((iter_num, train_loss, None))
                print(f"iter {iter_num:4d}/{args.max_iters}: "
                      f"train={train_loss:.4f}")
        elif iter_num % 10 == 0:
            print(f"iter {iter_num:4d}/{args.max_iters}: loss={train_loss:.4f}")
    
    end_time = time.time()
    print(f"\n⏱  Training time: {end_time - start_time:.2f} seconds")
    
    return history


def main():
    if not PEFT_AVAILABLE:
        print("❌ PEFT library required!")
        print("Install with: pip install peft")
        sys.exit(1)
    
    args = parse_args()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print("LoRA FINE-TUNING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Base checkpoint: {args.base_ckpt}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.out_dir}")
    print(f"{'='*70}\n")
    
    # Load base checkpoint
    print("Loading base checkpoint...")
    ckpt = torch.load(args.base_ckpt, map_location='cpu')
    
    if 'model_args' not in ckpt:
        print("❌ Checkpoint missing 'model_args'")
        sys.exit(1)
    
    # Create base model
    config = GPTConfig(**ckpt['model_args'])
    base_model = GPT(config)
    
    # Load weights
    state_dict = ckpt.get('model_state_dict') or ckpt.get('model')
    if state_dict is None:
        print("❌ Checkpoint missing model weights")
        sys.exit(1)
    
    # Normalize keys
    normalized = {}
    for k, v in state_dict.items():
        nk = k.replace('_orig_mod.', '').replace('module.', '')
        normalized[nk] = v
    
    base_model.load_state_dict(normalized, strict=False)
    print(f"✓ Base model loaded: {sum(p.numel() for p in base_model.parameters()):,} parameters")
    
    # Move to device
    base_model = base_model.to(device)
    
    # Generate baseline text BEFORE LoRA
    if args.generate_samples:
        print(f"\n{'='*70}")
        print("TEXT GENERATION - BEFORE LoRA (BASELINE)")
        print(f"{'='*70}")
        print(f"Prompt: {args.generation_prompt}\n")
        
        baseline_text = generate_text(
            base_model,
            prompt=args.generation_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_k=200,
            device=device
        )
        print(baseline_text)
        print(f"{'='*70}\n")
    
    # ============================================================
    # APPLY LoRA TO MODEL
    # ============================================================
    print("\nApplying LoRA configuration...")
    
    # Parse target modules
    target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",  # Don't apply LoRA to biases
        inference_mode=False  # Training mode
    )
    
    # Wrap model with LoRA
    model = get_peft_model(base_model, lora_config)
    
    print(f"✓ LoRA applied to modules: {target_modules}")
    
    # Print parameter summary
    param_stats = print_trainable_parameters(model)
    
    # Setup optimizer (only trains LoRA parameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Load training data
    data_path = Path(args.data_dir) / args.dataset / "bin" / "train.bin"
    if not data_path.exists():
        print(f"❌ Training data not found: {data_path}")
        sys.exit(1)
    
    train_data = np.memmap(data_path, dtype=np.uint16, mode='r')
    print(f"✓ Training data: {len(train_data):,} tokens")
    
    # Load validation data
    val_path = Path(args.data_dir) / args.dataset / "bin" / "val.bin"
    if val_path.exists():
        val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
        print(f"✓ Validation data: {len(val_data):,} tokens")
    else:
        val_data = None
        print("⚠  No validation data")
    
    # Train LoRA adapters
    history = train_lora(
        model, optimizer, train_data, val_data, args, device
    )
    
    # Generate text AFTER LoRA
    if args.generate_samples:
        print(f"\n{'='*70}")
        print("TEXT GENERATION - AFTER LoRA (IMPROVED)")
        print(f"{'='*70}")
        print(f"Prompt: {args.generation_prompt}\n")
        
        finetuned_text = generate_text(
            model,
            prompt=args.generation_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_k=200,
            device=device
        )
        print(finetuned_text)
        print(f"{'='*70}\n")
    
    # ============================================================
    # SAVE LoRA ADAPTERS
    # ============================================================
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save adapter weights
    adapter_path = out_dir / "adapter_model.bin"
    model.save_pretrained(str(out_dir))
    
    print(f"\n{'='*70}")
    print("SAVING LoRA ADAPTERS")
    print(f"{'='*70}")
    
    # Calculate adapter size
    adapter_size = 0
    if adapter_path.exists():
        adapter_size = adapter_path.stat().st_size
    else:
        # Try alternative filename
        adapter_path = out_dir / "adapter_model.safetensors"
        if adapter_path.exists():
            adapter_size = adapter_path.stat().st_size
    
    print(f"Adapter file: {adapter_path}")
    print(f"Adapter size: {adapter_size / (1024*1024):.2f} MB")
    print(f"Compared to full model: {adapter_size / (param_stats['total']*4) * 100:.2f}%")
    
    # Save metadata
    meta = {
        "lora_config": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules
        },
        "parameter_stats": param_stats,
        "adapter_size_bytes": adapter_size,
        "adapter_size_mb": adapter_size / (1024*1024),
        "training_history": [
            {"iter": it, "train_loss": tl, "val_loss": vl}
            for it, tl, vl in history
        ],
        "base_checkpoint": str(args.base_ckpt),
        "dataset": args.dataset
    }
    
    meta_path = out_dir / "lora_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    
    print(f"Metadata: {meta_path}")
    print(f"{'='*70}\n")
    
    # Final summary
    print(f"{'='*70}")
    print("LORA TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Trainable parameters: {param_stats['trainable']:,} ({param_stats['percentage']:.4f}%)")
    print(f"Adapter size: {adapter_size / (1024*1024):.2f} MB")
    print(f"Output directory: {out_dir}")
    
    # Compare with full fine-tuning
    print(f"\n📊 COMPARISON:")
    print(f"  Full fine-tuning delta: ~180-188 MB (no freezing)")
    print(f"  75% layer freezing delta: ~82-85 MB")
    print(f"  LoRA adapter (rank {args.lora_r}): {adapter_size / (1024*1024):.2f} MB")
    print(f"  LoRA reduction: {(1 - adapter_size/(180*1024*1024)) * 100:.1f}% vs full fine-tuning")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()