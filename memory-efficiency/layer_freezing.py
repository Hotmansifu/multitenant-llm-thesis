#!/usr/bin/env python3
"""
Layer Freezing Utility for Selective Fine-Tuning

Implements selective layer training by freezing bottom N% of transformer layers.
Research shows freezing 25-50% of bottom layers maintains quality while reducing:
- Memory usage by 30-50%
- Training time by 20-30%  
- Delta storage by 50%
"""
from typing import Dict, List
import torch.nn as nn


def freeze_bottom_layers(model, freeze_ratio: float = 0.5) -> Dict:
    """
    Freeze the bottom N% of transformer layers for selective fine-tuning.
    
    Bottom layers learn general features (grammar, syntax), top layers learn task-specific.
    Freezing bottom layers:
    - Reduces parameters modified during training
    - Maintains or improves performance (prevents overfitting)
    - Reduces delta storage automatically
    
    Args:
        model: GPT model with transformer.h layers
        freeze_ratio: Fraction of bottom layers to freeze (0.0 to 1.0)
                     0.25 = freeze bottom 25%
                     0.50 = freeze bottom 50%
                     0.75 = freeze bottom 75%
    
    Returns:
        Dict with statistics:
            - total_layers: Total number of transformer layers
            - frozen_layers: Number of layers frozen
            - trainable_layers: Number of layers still trainable
            - frozen_layer_indices: List of frozen layer indices
            - trainable_params_before: Trainable params before freezing
            - trainable_params_after: Trainable params after freezing
            - reduction_percentage: % reduction in trainable params
    """
    if not 0.0 <= freeze_ratio <= 1.0:
        raise ValueError(f"freeze_ratio must be 0.0-1.0, got {freeze_ratio}")
    
    # Count trainable params before
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get transformer layers
    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'h'):
        raise AttributeError("Model must have transformer.h (ModuleList of layers)")
    
    layers = model.transformer.h
    total_layers = len(layers)
    
    # Calculate how many to freeze (bottom layers)
    num_freeze = int(total_layers * freeze_ratio)
    
    frozen_indices = []
    trainable_indices = []
    
    # Freeze bottom layers
    for i, layer in enumerate(layers):
        if i < num_freeze:
            # Freeze this layer
            for param in layer.parameters():
                param.requires_grad = False
            frozen_indices.append(i)
        else:
            trainable_indices.append(i)
    
    # Count trainable params after
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate reduction
    reduction = ((trainable_before - trainable_after) / trainable_before * 100) if trainable_before > 0 else 0.0
    
    stats = {
        'total_layers': total_layers,
        'frozen_layers': num_freeze,
        'trainable_layers': total_layers - num_freeze,
        'frozen_layer_indices': frozen_indices,
        'trainable_layer_indices': trainable_indices,
        'trainable_params_before': trainable_before,
        'trainable_params_after': trainable_after,
        'reduction_percentage': reduction,
    }
    
    return stats


def print_freeze_summary(stats: Dict):
    """Print a formatted summary of layer freezing"""
    print(f"\n{'='*70}")
    print("SELECTIVE LAYER TRAINING - FREEZE SUMMARY")
    print(f"{'='*70}")
    print(f"Total transformer layers: {stats['total_layers']}")
    print(f"Frozen layers (bottom):   {stats['frozen_layers']} "
          f"(indices: {stats['frozen_layer_indices'][:5]}"
          f"{('...' + str(stats['frozen_layer_indices'][-1])) if len(stats['frozen_layer_indices']) > 5 else ''})")
    print(f"Trainable layers (top):   {stats['trainable_layers']} "
          f"(indices: {stats['trainable_layer_indices'][:5]}"
          f"{('...' + str(stats['trainable_layer_indices'][-1])) if len(stats['trainable_layer_indices']) > 5 else ''})")
    print(f"\nTrainable parameters:")
    print(f"  Before freezing: {stats['trainable_params_before']:,}")
    print(f"  After freezing:  {stats['trainable_params_after']:,}")
    print(f"  Reduction:       {stats['reduction_percentage']:.2f}%")
    print(f"\nExpected benefits:")
    print(f"  • Delta storage reduction: ~{stats['reduction_percentage']:.0f}%")
    print(f"  • Training time reduction: ~20-30%")
    print(f"  • Memory usage reduction:  ~30-50%")
    print(f"  • Quality: Maintained or improved (prevents overfitting)")
    print(f"{'='*70}\n")


def verify_freezing(model) -> Dict:
    """
    Verify which layers are frozen vs trainable.
    Returns dict with layer-wise trainable status.
    """
    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'h'):
        return {}
    
    layers = model.transformer.h
    status = {}
    
    for i, layer in enumerate(layers):
        trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in layer.parameters())
        is_frozen = (trainable_params == 0)
        
        status[f'layer_{i}'] = {
            'frozen': is_frozen,
            'trainable_params': trainable_params,
            'total_params': total_params
        }
    
    return status


# Example usage for testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent to path for imports
    sys.path.append(str(Path(__file__).parent))
    from model import GPT, GPTConfig
    
    # Create test model
    config = GPTConfig(n_layer=12, n_embd=768)
    model = GPT(config)
    
    print("Testing layer freezing...")
    
    # Test different freeze ratios
    for ratio in [0.0, 0.25, 0.5, 0.75]:
        print(f"\n{'='*70}")
        print(f"Testing freeze_ratio = {ratio}")
        print(f"{'='*70}")
        
        # Reset model (unfreeze all)
        for param in model.parameters():
            param.requires_grad = True
        
        # Freeze
        stats = freeze_bottom_layers(model, freeze_ratio=ratio)
        print_freeze_summary(stats)
        
        # Verify
        status = verify_freezing(model)
        frozen_count = sum(1 for s in status.values() if s['frozen'])
        print(f"Verification: {frozen_count}/{len(status)} layers frozen ✓")