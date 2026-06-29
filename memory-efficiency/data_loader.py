#!/usr/bin/env python3
"""
Data Loader Utilities
======================
Simple batch loading for memory-mapped training data.
"""
from typing import Tuple

import numpy as np
import torch


def get_batch(
    data: np.memmap,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch (x, y) from memory-mapped token data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x_list = []
    y_list = []
    for i in ix:
        i = int(i)
        x_seq = torch.from_numpy(data[i : i + block_size].astype(np.int64))
        y_seq = torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
        x_list.append(x_seq)
        y_list.append(y_seq)

    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    return x, y
