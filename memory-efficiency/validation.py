#!/usr/bin/env python3
"""
Validation utilities for model evaluation during training.
"""
from __future__ import annotations

import torch
import numpy as np

from data_loader import get_batch


@torch.no_grad()
def estimate_loss(
    model,
    data: np.memmap,
    eval_iters: int,
    batch_size: int,
    device: torch.device,
    block_size: int,
) -> float:
    """Estimate mean loss over `eval_iters` random batches from `data`."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))
