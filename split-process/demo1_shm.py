import sys, os, time, mmap, torch, tiktoken
pythonSCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from model import GPTConfig, GPT
from multiprocessing import Process, Pipe, set_start_method

CKPT   = os.path.join(SCRIPT_DIR, 'ckpt.pt')
GEN    = 80
PROMPT = "The future of artificial intelligence in healthcare is"
TEMP   = 0.9
TOP_K  = 40

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

def server_fn(conn, shm_path, shm_size):
    import sys, mmap, torch
    sys.path.insert(0, '/home/sifu/nanoGPT')
    m, cfg = load_model()
    C = cfg.n_embd
    VOCAB = cfg.vocab_size
    f  = open(shm_path, 'r+b')
    mm = mmap.mmap(f.fileno(), shm_size)
    conn.send('ready')
    with torch.no_grad():
        while True:
            msg = conn.recv()
            if msg == 'quit':
                break
            cmd, layer, T = msg
            if cmd == 'mlp':
                h = torch.frombuffer(bytearray(mm[0:T*C*4]), dtype=torch.float32).reshape(1,T,C).clone()
                block = m.transformer.h[layer]
                out = h + block.mlp(block.ln_2(h))
                mm[0:T*C*4] = out.numpy().tobytes()
            elif cmd == 'head':
                h = torch.frombuffer(bytearray(mm[0:T*C*4]), dtype=torch.float32).reshape(1,T,C).clone()
                out = m.lm_head(m.transformer.ln_f(h))
                mm[0:T*VOCAB*4] = out.numpy().tobytes()
            mm.flush()
            conn.send('done')
    mm.close(); f.close()

def main():
    set_start_method('spawn', force=True)
    m, cfg = load_model()
    C, n_layer, VOCAB = cfg.n_embd, cfg.n_layer, cfg.vocab_size
    SHM  = '/dev/shm/bm_demo1'
    SIZE = max(cfg.block_size*C*4, cfg.block_size*VOCAB*4)

    with open(SHM, 'wb') as f: f.write(b'\x00'*SIZE)

    pc, cc = Pipe()
    proc = Process(target=server_fn, args=(cc, SHM, SIZE))
    proc.start()

    f  = open(SHM, 'r+b')
    mm = mmap.mmap(f.fileno(), SIZE)
    assert pc.recv() == 'ready'
    print(f"  Client PID: {os.getpid()} | Server PID: {proc.pid}")
    print(f"  /dev/shm | Per pass (T=8): {n_layer*2*8*C*4/1024:.1f} KB")

    enc = tiktoken.get_encoding('gpt2')

    def xfer(cmd, layer, h):
        T = h.shape[1]
        mm[0:T*C*4] = h.numpy().tobytes(); mm.flush()
        pc.send((cmd, layer, T)); pc.recv()
        if cmd == 'mlp':
            return torch.frombuffer(bytearray(mm[0:T*C*4]), dtype=torch.float32).reshape(1,T,C).clone(), T*C*4*2
        data = bytes(mm[0:T*VOCAB*4])
        return torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(1,T,VOCAB).clone(), T*C*4+T*VOCAB*4

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
                        x, b = xfer('mlp', i, x); bx += b
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

    print(f"\n  [2/2] Split /dev/shm ({GEN} tokens)...")
    t0=time.time(); ts,bx = gen(True); t_split=time.time()-t0
    print(f"  {t_split:.3f}s | {GEN/t_split:.1f} tok/s")

    pc.send('quit'); proc.join()
    match = tm == ts

    print(f"\n  ══ RESULTS Demo 1 — Shared Memory (bare metal) ══")
    print(f"  Monolithic:     {t_mono:.3f}s  ({GEN/t_mono:.1f} tok/s)")
    print(f"  Split /dev/shm: {t_split:.3f}s  ({GEN/t_split:.1f} tok/s)")
    print(f"  Overhead:       {(t_split/t_mono-1)*100:+.1f}%")
    print(f"  Total bytes TX: {bx/1024:.1f} KB")
    print(f"  Texts identical: {'YES' if match else 'NO'}")
    print(f"\n  [mono]  {tm[len(PROMPT):len(PROMPT)+100]}")
    print(f"  [split] {ts[len(PROMPT):len(PROMPT)+100]}")

    mm.close(); f.close()
    if os.path.exists(SHM): os.unlink(SHM)

if __name__ == '__main__':
    main()
