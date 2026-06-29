import sys, os, time, socket, struct, torch, tiktoken
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from model import GPTConfig, GPT
from multiprocessing import Process, set_start_method

CKPT   = os.path.join(SCRIPT_DIR, 'ckpt.pt')
GEN    = 80
PROMPT = "The future of artificial intelligence in healthcare is"
TEMP   = 0.9
TOP_K  = 40
PORT   = 29876

def load_model():
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    args = ckpt['model_args']
    args['use_gaussian'] = True
    args['gaussian_sigma2'] = 16.0
    from model import GPTConfig, GPT
    cfg = GPTConfig(**args)
    m = GPT(cfg)
    state = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    m.load_state_dict(state, strict=False)
    m.eval()
    return m, cfg

def send_tensor(sock, t):
    data = t.numpy().tobytes()
    sock.sendall(struct.pack('!I', len(data)))
    sock.sendall(data)

def recv_tensor(sock, shape, dtype=torch.float32):
    n = struct.unpack('!I', sock.recv(4))[0]
    buf = b''
    while len(buf) < n:
        buf += sock.recv(n - len(buf))
    return torch.frombuffer(bytearray(buf), dtype=dtype).reshape(shape).clone()

def server_fn():
    import sys, socket, struct, torch
    sys.path.insert(0, '/home/sifu/nanoGPT')
    m, cfg = load_model()
    C = cfg.n_embd
    VOCAB = cfg.vocab_size

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', PORT))
    srv.listen(1)
    conn, _ = srv.accept()

    with torch.no_grad():
        while True:
            hdr = conn.recv(6)
            if not hdr or hdr == b'quit__':
                break
            cmd = hdr[0:4].decode().strip()
            layer = struct.unpack('!H', hdr[4:6])[0]
            n = struct.unpack('!I', conn.recv(4))[0]
            buf = b''
            while len(buf) < n:
                buf += conn.recv(n - len(buf))

            if cmd == 'mlp_':
                T = n // (C*4)
                h = torch.frombuffer(bytearray(buf), dtype=torch.float32).reshape(1,T,C).clone()
                block = m.transformer.h[layer]
                out = h + block.mlp(block.ln_2(h))
                send_tensor(conn, out)
            elif cmd == 'head':
                T = n // (C*4)
                h = torch.frombuffer(bytearray(buf), dtype=torch.float32).reshape(1,T,C).clone()
                out = m.lm_head(m.transformer.ln_f(h))
                send_tensor(conn, out)

    conn.close(); srv.close()

def main():
    set_start_method('spawn', force=True)
    m, cfg = load_model()
    C, n_layer, VOCAB = cfg.n_embd, cfg.n_layer, cfg.vocab_size

    proc = Process(target=server_fn)
    proc.start()
    time.sleep(3)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('127.0.0.1', PORT))

    print(f"\n  Client PID: {os.getpid()} | Server PID: {proc.pid}")
    print(f"  IPC: TCP localhost:{PORT}")
    print(f"  Per pass (T=8): {n_layer*2*8*C*4/1024:.1f} KB")

    enc = tiktoken.get_encoding('gpt2')

    def xfer(cmd, layer, h):
        T = h.shape[1]
        data = h.numpy().tobytes()
        hdr = cmd.ljust(4).encode() + struct.pack('!H', layer)
        sock.sendall(hdr)
        sock.sendall(struct.pack('!I', len(data)))
        sock.sendall(data)
        bx = len(data)
        if cmd == 'mlp_':
            out = recv_tensor(sock, (1,T,C))
            return out, bx + T*C*4
        else:
            out = recv_tensor(sock, (1,T,VOCAB))
            return out, bx + T*VOCAB*4

    def gen(split):
        idx = torch.tensor(enc.encode(PROMPT), dtype=torch.long).unsqueeze(0)
        torch.manual_seed(42); bx = 0
        with torch.no_grad():
            for _ in range(GEN):
                T = idx.shape[1]
                x = m.transformer.drop(m.transformer.wte(idx) + m.transformer.wpe(torch.arange(T)))
                for i in range(n_layer):
                    blk = m.transformer.h[i]
                    x = x + blk.attn(blk.ln_1(x))
                    if split:
                        x, b = xfer('mlp_', i, x); bx += b
                    else:
                        x = x + blk.mlp(blk.ln_2(x))
                if split:
                    logits, b = xfer('head', 0, x); bx += b
                else:
                    logits = m.lm_head(m.transformer.ln_f(x))
                logits = logits[:,-1,:]/TEMP
                v,_ = torch.topk(logits, TOP_K)
                logits[logits < v[:,[-1]]] = float('-inf')
                idx = torch.cat([idx, torch.multinomial(torch.softmax(logits,-1),1)],1)
        return enc.decode(idx[0].tolist()), bx

    print(f"\n  [1/2] Monolithic ({GEN} tokens)...")
    t0=time.time(); tm,_ = gen(False); t_mono=time.time()-t0
    print(f"  {t_mono:.3f}s | {GEN/t_mono:.1f} tok/s")

    print(f"\n  [2/2] Split TCP localhost ({GEN} tokens)...")
    t0=time.time(); ts,bx = gen(True); t_split=time.time()-t0
    print(f"  {t_split:.3f}s | {GEN/t_split:.1f} tok/s")

    sock.sendall(b'quit__')
    proc.join()
    match = tm == ts

    print(f"\n  ══ RESULTS Demo 2 — Network TCP (bare metal) ══")
    print(f"  Monolithic:      {t_mono:.3f}s  ({GEN/t_mono:.1f} tok/s)")
    print(f"  Split TCP local: {t_split:.3f}s  ({GEN/t_split:.1f} tok/s)")
    print(f"  Overhead:        {(t_split/t_mono-1)*100:+.1f}%")
    print(f"  Total bytes TX:  {bx/1024:.1f} KB")
    print(f"  Texts identical: {'YES' if match else 'NO'}")
    print(f"\n  [mono]  {tm[len(PROMPT):len(PROMPT)+100]}")
    print(f"  [split] {ts[len(PROMPT):len(PROMPT)+100]}")

if __name__ == '__main__':
    main()
