#!/usr/bin/env python3
"""
paillier_additive_setup_ckpt_v2.py
Tightened measurement of the Paillier additive-PHE setup cost on the REAL checkpoint.

Changes vs v1 (addresses the "3-sample per-term is noisy" issue):
  - per-term homomorphic cost measured over N_NEURON=30 real neurons across 2 layers
    (60 real rows total), and the spread (min/mean/max, std) is reported so you can
    see it is stable before trusting the extrapolation.
  - encryption rate measured on a full real rho vector (C and 4C), not a 32-sample.
  - everything else identical: correctness check, real nonzero counts from your
    weights, same extrapolation method the thesis uses for D3.

Setup step (matches the protocol diagram):
    client encrypts secret masks rho ; server computes E(W.rho)=sum_i E(rho_i)*w_i
    homomorphically (ternary w -> additions only), never decrypting, never seeing rho;
    client decrypts -> stores W.rho. Online inference is separate, ~1.0x, no crypto.

Run:
    cd /home/coder/project/nanoGPT
    PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
        ~/.pyenv/versions/3.11.9/bin/python paillier_additive_setup_ckpt_v2.py
Optional: checkpoint path as argv[1].
"""

import os, sys, time
import numpy as np
import torch

try:
    import phe as paillier
except ImportError:
    print("ERROR: python-paillier not installed.  ~/.pyenv/.../pip install phe")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model_bitnet import GPTConfig, GPT

CKPT_PATH = sys.argv[1] if len(sys.argv) > 1 else "out/bitnet_gaussian_demo/ckpt.pt"
KEY_BITS  = 2048
N_NEURON  = 30          # neurons timed per layer for per-term cost
N_LAYERS_SAMPLED = 2    # how many layers to sample neurons from

rng = np.random.default_rng(0)


def ternary(w):
    return np.sign(w.detach().float().numpy()).astype(np.int64)


def correction_neuron(enc_rho, w_row):
    """E((W.rho)_j) = sum_i E(rho_i)*w_i -- server side, no decryption, skip zeros."""
    acc = None
    for e, w in zip(enc_rho, w_row):
        if w == 0:
            continue
        term = e * int(w)
        acc = term if acc is None else acc + term
    return acc


def main():
    print("paillier_additive_setup_ckpt_v2.py - tightened real-checkpoint setup cost")
    print("=" * 70)
    if not os.path.exists(CKPT_PATH):
        print(f"ERROR: checkpoint not found: {CKPT_PATH}"); sys.exit(1)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model_args = ckpt["model_args"]
    if "config" in ckpt:
        model_args.update({k: v for k, v in ckpt["config"].items()
                           if k not in model_args})
    model_args["use_bitnet"] = True
    cfg = GPTConfig(**model_args)
    gpt = GPT(cfg)
    sd  = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    gpt.load_state_dict(sd, strict=False)
    gpt.eval()

    C, C4, L = cfg.n_embd, 4 * cfg.n_embd, cfg.n_layer
    print(f"Model: {L} layers, C={C}, 4C={C4}, key={KEY_BITS}-bit\n")

    # real ternary weights + density
    W_fc1, W_fc2, nnz_fc1, nnz_fc2 = [], [], 0, 0
    for i in range(L):
        w1, w2 = ternary(gpt.transformer.h[i].mlp.c_fc.weight), ternary(gpt.transformer.h[i].mlp.c_proj.weight)
        W_fc1.append(w1); W_fc2.append(w2)
        nnz_fc1 += int(np.count_nonzero(w1)); nnz_fc2 += int(np.count_nonzero(w2))
    total_elems = L * (C4 * C + C * C4)
    total_nnz   = nnz_fc1 + nnz_fc2
    print("-- Weight density (real checkpoint) --")
    print(f"  total ternary weights : {total_elems:,}")
    print(f"  nonzeros              : {total_nnz:,}")
    print(f"  zero fraction         : {(1-total_nnz/total_elems)*100:.1f}%\n")

    t0 = time.perf_counter()
    pub, priv = paillier.generate_paillier_keypair(n_length=KEY_BITS)
    print(f"Keygen: {time.perf_counter()-t0:.2f}s\n")

    # correctness on real layer-0 fc1 (8 neurons)
    print("-- Correctness check (real layer-0 fc1, 8 neurons) --")
    rho0 = rng.standard_normal(C)
    enc0 = [pub.encrypt(float(v)) for v in rho0]
    got  = np.array([priv.decrypt(correction_neuron(enc0, W_fc1[0][j])) for j in range(8)])
    want = W_fc1[0][:8].astype(np.float64) @ rho0
    err  = np.abs(got - want).max()
    print(f"  max error {err:.2e}   {'PASS' if err < 1e-6 else 'FAIL'}\n")

    # encryption rate on FULL real vectors (C and 4C)
    print("-- Encryption rate (full real vectors) --")
    t0 = time.perf_counter(); enc_rho1 = [pub.encrypt(float(v)) for v in rng.standard_normal(C)]
    t_enc_c = time.perf_counter() - t0
    t0 = time.perf_counter(); enc_rho2 = [pub.encrypt(float(v)) for v in rng.standard_normal(C4)]
    t_enc_c4 = time.perf_counter() - t0
    enc_rate = (C + C4) / (t_enc_c + t_enc_c4)
    print(f"  encrypt {C}: {t_enc_c:.1f}s   encrypt {C4}: {t_enc_c4:.1f}s   "
          f"-> {enc_rate:.1f} elem/s")

    # decryption rate on C4 corrections
    t0 = time.perf_counter(); _ = [priv.decrypt(e) for e in enc_rho2]
    dec_rate = C4 / (time.perf_counter() - t0)
    print(f"  decrypt rate: {dec_rate:.1f} elem/s\n")

    # per-term cost over N_NEURON neurons across N_LAYERS_SAMPLED layers
    print(f"-- Per-term homomorphic cost ({N_NEURON} neurons x {N_LAYERS_SAMPLED} layers) --")
    per_term_samples = []
    layers = list(range(min(N_LAYERS_SAMPLED, L)))
    for li in layers:
        for j in range(N_NEURON):
            row = W_fc1[li][j]                      # fc1 row: dot over C terms
            nz  = int(np.count_nonzero(row))
            if nz == 0:
                continue
            t0 = time.perf_counter(); correction_neuron(enc_rho1, row)
            per_term_samples.append((time.perf_counter() - t0) / nz)
        for j in range(N_NEURON):
            row = W_fc2[li][j]                      # fc2 row: dot over C4 terms
            nz  = int(np.count_nonzero(row))
            if nz == 0:
                continue
            t0 = time.perf_counter(); correction_neuron(enc_rho2, row)
            per_term_samples.append((time.perf_counter() - t0) / nz)
    pts = np.array(per_term_samples)
    per_term = float(pts.mean())
    print(f"  samples: {len(pts)}")
    print(f"  per-term ms: mean {pts.mean()*1000:.3f}  std {pts.std()*1000:.3f}  "
          f"min {pts.min()*1000:.3f}  max {pts.max()*1000:.3f}")
    print(f"  relative spread (std/mean): {pts.std()/pts.mean()*100:.1f}%\n")

    # extrapolate full setup
    enc_total = L * (C + C4) / enc_rate
    dec_total = L * (C4 + C) / dec_rate
    mat_total = total_nnz * per_term
    setup_total = enc_total + mat_total + dec_total
    # +/- band from per-term std
    band = total_nnz * pts.std()
    print("-- Projected full setup cost (extrapolated from measured rates + real nnz) --")
    print(f"  encrypt rho (all layers)  : {enc_total:9.1f} s")
    print(f"  homomorphic matmul        : {mat_total:9.1f} s  (+/- {band:.0f}s from per-term std)")
    print(f"  decrypt corrections       : {dec_total:9.1f} s")
    print(f"  TOTAL one-time setup      : {setup_total:9.1f} s  ({setup_total/60:.1f} min)")
    print(f"  honest range              : {(setup_total-band)/60:.0f}-{(setup_total+band)/60:.0f} min")
    print()
    print("  One-time per mask set. Online ~1.0x over ternary baseline (no crypto online).")
    print("  Fresh mask per inference for security -> this setup recurs (does not amortize).")


if __name__ == "__main__":
    main()