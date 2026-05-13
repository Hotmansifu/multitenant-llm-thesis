# Scalable and Secure Multi-Tenant LLM Deployment

**Bachelor Thesis — Constructor University**  
**Author:** Luka Andghuladze  
**Supervisor:** Prof. A. Tormasov  
**Submitted:** May 2026

## Overview

This repository contains the implementation code for the thesis
*"Scalable and Secure Multi-Tenant LLM Deployment: Memory Efficiency, Encryption, and Quantization Approaches"*.

The thesis investigates three approaches to secure and scalable multi-tenant LLM hosting,
addressing the threat of an honest-but-curious server owner with physical access to shared hardware.

## Repository Structure

| Folder | Research Questions | Description |
|---|---|---|
| `memory-efficiency/` | RQ1, RQ2 | LoRA, layer freezing, and full fine-tuning comparison |
| `gaussian-bitnet/` | RQ4 | Gaussian kernel attention and BitNet ternary weights |
| `split-process/` | RQ3 | Split-process architecture with Paillier homomorphic encryption |

## Key Results

| Finding | Result |
|---|---|
| LoRA vs layer freezing (30M) | 109× less memory, 16% quality loss |
| LoRA vs layer freezing (774M) | 17× less memory |
| Split-process shared memory (D1) | ~40% overhead |
| Split-process TCP (D2) | ~560% overhead |
| Full Paillier HE hidden state (D3) | ~56,000× overhead (extrapolated) |
| LoRA adapter path encryption | 16,383× overhead (measured) |
| Gaussian kernel vs softmax | 0.06% quality difference |
| BitNet + Gaussian compound effect | 0% interaction at both tested scales |