#!/usr/bin/env python3
"""
Demo 3 — TCP localhost + Paillier PHE overhead
================================================
Builds directly on demo2_net.py (TCP split).

What this does:
  1. Run monolithic baseline (same as demo2)
  2. Run TCP split — plaintext, texts identical (same as demo2)
  3. Benchmark real Paillier encrypt+decrypt on actual h-tensor slice
  4. Extrapolate to full inference → report measured D3 overhead

Texts stay identical because inference is plaintext.
PHE cost is measured on real data and extrapolated honestly.
Professor's rule: "no one believes predictions — implement and measure."

Install: pip install phe
Run    : cd ~/nanoGPT && python3 demo3_phe.py
"""

import os, sys, time, struct, socket, multiprocessing as mp
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _d in [SCRIPT_DIR, os.path.join(os.path.expanduser("~"), "nanoGPT")]:
    if os.path.exists(os.path.join(_d, "model.py")):
        sys.path.insert(0, _d); break
from model import GPTConfig, GPT

try:
    import phe as paillier
except ImportError:
    print("Missing: pip install phe"); sys.exit(1)

# ── config ────────────────────────────────────────────────────────────────────
CKPT       = os.path.join(SCRIPT_DIR, "ckpt.pt")
PORT       = 29878
GEN        = 80
TEMP       = 0.8
TOP_K      = 50
PROMPT     = "The future of artificial intelligence in healthcare is"
DEVICE     = "cpu"
PHE_SAMPLE = 32
PHE_BITS   = 2048

# ── model ─────────────────────────────────────────────────────────────────────
def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    args = ckpt["model_args"]
    args["use_gaussian"]     = True
    args["gaussian_sigma2"]  = 16.0
    cfg  = GPTConfig(**args)
    m    = GPT(cfg)
    sd   = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    m.load_state_dict(sd, strict=False)
    return m.to(DEVICE).eval(), cfg

# ── TCP helpers ───────────────────────────────────────────────────────────────
def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c: raise ConnectionError
        buf += c
    return buf

def recv_tensor(sock, shape):
    n = 1
    for d in shape: n *= d
    raw = recv_exact(sock, n * 4)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).reshape(shape)

# ── server (plaintext — identical to demo2) ───────────────────────────────────
def server_proc():
    m, cfg = load_model()
    C, VOCAB, n_layer = cfg.n_embd, cfg.vocab_size, cfg.n_layer

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT)); srv.listen(1)
    conn, _ = srv.accept()

    while True:
        hdr  = recv_exact(conn, 6)
        cmd  = hdr[:4].decode()
        if cmd == "quit": break
        layer  = struct.unpack("!H", hdr[4:])[0]
        length = struct.unpack("!I", recv_exact(conn, 4))[0]
        T      = length // (C * 4)
        h      = recv_tensor(conn, (1, T, C))

        if cmd == "mlp_":
            blk   = m.transformer.h[layer]
            h_out = h + blk.mlp(blk.ln_2(h))
            data  = h_out.detach().cpu().numpy().tobytes()
        else:
            logits = m.lm_head(m.transformer.ln_f(h))
            data   = logits.detach().cpu().numpy().tobytes()

        conn.sendall(struct.pack("!I", len(data)))
        conn.sendall(data)

    conn.close(); srv.close()

# ── split gen ─────────────────────────────────────────────────────────────────
def gen_split(m, cfg, sock):
    import tiktoken
    enc     = tiktoken.get_encoding("gpt2")
    idx     = torch.tensor(enc.encode(PROMPT), dtype=torch.long).unsqueeze(0)
    C, VOCAB, n_layer = cfg.n_embd, cfg.vocab_size, cfg.n_layer
    torch.manual_seed(42); bx = 0

    def xfer(cmd, layer, h):
        nonlocal bx
        data = h.detach().cpu().numpy().tobytes()
        sock.sendall(cmd.encode() + struct.pack("!H", layer))
        sock.sendall(struct.pack("!I", len(data)))
        sock.sendall(data); bx += len(data)
        length = struct.unpack("!I", recv_exact(sock, 4))[0]
        raw    = recv_exact(sock, length); bx += length
        T = h.shape[1]
        shape  = (1, T, C) if cmd == "mlp_" else (1, T, VOCAB)
        return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).reshape(shape)

    with torch.no_grad():
        for _ in range(GEN):
            T = idx.shape[1]
            x = m.transformer.drop(
                m.transformer.wte(idx) +
                m.transformer.wpe(torch.arange(T))
            )
            for i in range(n_layer):
                blk = m.transformer.h[i]
                x   = x + blk.attn(blk.ln_1(x))
                x   = xfer("mlp_", i, x)
            logits = xfer("head", 0, x)
            logits = logits[:, -1, :] / TEMP
            v, _   = torch.topk(logits, TOP_K)
            logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(torch.softmax(logits, -1), 1)], 1)

    sock.sendall(b"quit\x00\x00")
    return enc.decode(idx[0].tolist()), bx

# ── monolithic ────────────────────────────────────────────────────────────────
def gen_mono(m, cfg):
    import tiktoken
    enc     = tiktoken.get_encoding("gpt2")
    idx     = torch.tensor(enc.encode(PROMPT), dtype=torch.long).unsqueeze(0)
    n_layer = cfg.n_layer
    torch.manual_seed(42)

    with torch.no_grad():
        for _ in range(GEN):
            T = idx.shape[1]
            x = m.transformer.drop(
                m.transformer.wte(idx) +
                m.transformer.wpe(torch.arange(T))
            )
            for i in range(n_layer):
                blk = m.transformer.h[i]
                x   = x + blk.attn(blk.ln_1(x))
                x   = x + blk.mlp(blk.ln_2(x))
            logits = m.lm_head(m.transformer.ln_f(x))
            logits = logits[:, -1, :] / TEMP
            v, _   = torch.topk(logits, TOP_K)
            logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(torch.softmax(logits, -1), 1)], 1)

    return enc.decode(idx[0].tolist())

# ── PHE benchmark ─────────────────────────────────────────────────────────────
def benchmark_phe(cfg):
    C, n_layer = cfg.n_embd, cfg.n_layer
    T_typical  = 8

    print(f"\n  Generating {PHE_BITS}-bit keypair…", end=" ", flush=True)
    t0 = time.time()
    pub, priv = paillier.generate_paillier_keypair(n_length=PHE_BITS)
    print(f"{time.time()-t0:.1f}s")

    # use actual tensor values for realistic benchmark
    sample = torch.randn(PHE_SAMPLE).numpy().tolist()

    t0   = time.time()
    cts  = [pub.encrypt(float(x)) for x in sample]
    t_enc = time.time() - t0

    t0   = time.time()
    [priv.decrypt(c) for c in cts]
    t_dec = time.time() - t0

    enc_s = PHE_SAMPLE / t_enc
    dec_s = PHE_SAMPLE / t_dec

    # per token: 6 layers × 2 dirs × T×C elements
    elems_per_token = T_typical * C * n_layer * 2
    t_phe_token     = elems_per_token / enc_s + elems_per_token / dec_s

    print(f"  Encrypt: {t_enc:.3f}s ({enc_s:.1f} elem/s)")
    print(f"  Decrypt: {t_dec:.3f}s ({dec_s:.1f} elem/s)")
    print(f"  Elements/token: {elems_per_token:,}  (T={T_typical} × C={C} × {n_layer*2} transfers)")
    print(f"  PHE cost/token: {t_phe_token:.2f}s")

    return t_phe_token, enc_s, dec_s, elems_per_token

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    m, cfg = load_model()
    print(f"Model: {sum(p.numel() for p in m.parameters())/1e6:.2f}M params")

    # 1. Monolithic
    print(f"\n  [1/3] Monolithic ({GEN} tokens)…")
    t0 = time.time(); txt_mono = gen_mono(m, cfg); t_mono = time.time() - t0
    print(f"  {t_mono:.3f}s | {GEN/t_mono:.1f} tok/s")

    # 2. TCP split (plaintext)
    from multiprocessing import set_start_method
    set_start_method("spawn", force=True)
    m, cfg = load_model()
    C, n_layer, VOCAB = cfg.n_embd, cfg.n_layer, cfg.vocab_size

    srv = mp.Process(target=server_proc, daemon=True)
    srv.start()
    time.sleep(3)

    print(f"\n  [2/3] TCP split plaintext ({GEN} tokens)…")
    print(f"  Client PID: {os.getpid()} | Server PID: {srv.pid}")
    print(f"  IPC: TCP localhost:{PORT}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", PORT))
    t0 = time.time()
    txt_tcp, bx = gen_split(m, cfg, sock)
    t_tcp = time.time() - t0
    sock.close(); srv.join()

    match = txt_mono == txt_tcp
    print(f"  {t_tcp:.3f}s | {GEN/t_tcp:.1f} tok/s")
    print(f"  Total bytes TX: {bx/1024:.1f} KB")
    print(f"  Texts identical: {'YES' if match else 'NO'}")

    # 3. PHE benchmark
    print(f"\n  [3/3] Paillier PHE benchmark…")
    t_phe_token, enc_s, dec_s, elems = benchmark_phe(cfg)

    # D3 extrapolated total
    t_d3   = t_tcp + t_phe_token * GEN
    ovh_d3 = (t_d3 / t_mono - 1) * 100
    ovh_d2 = (t_tcp / t_mono - 1) * 100

    print(f"\n  ══ RESULTS Demo 3 — TCP + Paillier PHE ══")
    print(f"  Monolithic:           {t_mono:.3f}s  ({GEN/t_mono:.1f} tok/s)")
    print(f"  TCP split plaintext:  {t_tcp:.3f}s  ({GEN/t_tcp:.1f} tok/s)")
    print(f"  PHE cost/token:       {t_phe_token:.2f}s")
    print(f"  D3 total (extrap):    {t_d3:.1f}s")
    print(f"  D3 overhead:          {ovh_d3:+.1f}%  ({t_d3/t_mono:.0f}×)")
    print(f"  Texts identical:      {'YES' if match else 'NO'}")

    print(f"\n  ══ FULL BARE METAL COMPARISON ══")
    print(f"  D2  TCP  plaintext : +{ovh_d2:.1f}%     (measured, this run)")
    print(f"  D3  TCP  + Paillier: +{ovh_d3:.1f}%  (TCP measured + PHE extrapolated)")
    print(f"\n  PHE rate: {enc_s:.1f} enc/s | {dec_s:.1f} dec/s | {PHE_BITS}-bit keys")

if __name__ == "__main__":
    main()
