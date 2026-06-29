"""
Text generation utilities for model evaluation and demonstration
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import tiktoken

if TYPE_CHECKING:
    from model import GPT


@torch.no_grad()
def generate_text(
    model: 'GPT',
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    device: torch.device = None,
    encoding: str = "gpt2"
) -> str:
    """
    Generate text from a prompt
    
    Args:
        model: GPT model
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        device: torch device (auto-detect if None)
        encoding: Tokenizer encoding to use
    
    Returns:
        Generated text (includes original prompt)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Encode prompt
    enc = tiktoken.get_encoding(encoding)
    tokens = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate tokens
    model.eval()
    
    for _ in range(max_new_tokens):
        # Crop to block_size if needed
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        
        # Forward pass
        logits, _ = model(idx_cond)
        
        # Get logits for last token
        logits = logits[:, -1, :] / temperature
        
        # Top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Apply softmax and sample
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    model.train()
    
    # Decode
    generated_tokens = idx[0].tolist()
    generated_text = enc.decode(generated_tokens)
    
    return generated_text


def print_generation_header(title: str, prompt: str):
    """Print formatted header for text generation"""
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"Prompt: \"{prompt}\"")
    print(f"-" * 70)


def print_generation_footer():
    """Print formatted footer for text generation"""
    print(f"{'='*70}\n")