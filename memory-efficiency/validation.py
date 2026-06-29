"""
Validation utilities for model evaluation during training
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np

if TYPE_CHECKING:
    from model import GPT


@torch.no_grad()
def estimate_loss(
    model: 'GPT',
    data: np.memmap,
    eval_iters: int,
    batch_size: int,
    device: torch.device,
    block_size: int = None
) -> float:
    """
    Estimate loss on a dataset
    
    Args:
        model: GPT model
        data: Memory-mapped data
        eval_iters: Number of batches to evaluate
        batch_size: Batch size
        device: torch device
        block_size: Override model's block_size if provided
    
    Returns:
        Average loss
    """
    from data_loader import get_batch
    
    model.eval()
    losses = []
    
    if block_size is None:
        block_size = model.config.block_size
    
    for _ in range(eval_iters):
        X, Y = get_batch(
            data,
            block_size=block_size,
            batch_size=batch_size,
            device=device
        )
        logits, loss = model(input_ids=X, labels=Y)
        losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses)