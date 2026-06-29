#!/usr/bin/env python3
"""
Data Loader Utilities
======================
Simple batch loading for training.
"""
import numpy as np
import torch
from typing import Tuple


def get_batch(
    data: np.memmap,
        block_size: int,
            batch_size: int,
                device: torch.device
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    """Sample random batch from memory-mapped data."""
                        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
                            
                                x_list = []
                                    y_list = []
                                        
                                            for i in ix.cpu():
                                                    i = int(i)
                                                            x_seq = torch.from_numpy(data[i : i + block_size].astype(np.int64))
                                                                    x_list.append(x_seq)
                                                                            
                                                                                    y_seq = torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
                                                                                            y_list.append(y_seq)
                                                                                                
                                                                                                    x = torch.stack(x_list).to(device)
                                                                                                        y = torch.stack(y_list).to(device)
                                                                                                            
                                                                                                                return x, y
                                                                                                                