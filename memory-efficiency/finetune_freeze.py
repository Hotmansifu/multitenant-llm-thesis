import torch, argparse, time, numpy as np, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig
from layer_freezing import freeze_bottom_layers, print_freeze_summary

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device):
    model.eval()
    results = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, targets=y)
            losses.append(loss.item())
        results[split] = float(np.mean(losses))
    model.train()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_ckpt', default='../out/checkpoints/base_final.pt')
    parser.add_argument('--data_dir', default='../data')
    parser.add_argument('--dataset', default='chicago')
    parser.add_argument('--freeze_ratio', type=float, default=0.75)
    parser.add_argument('--max_iters', type=int, default=2000)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--eval_iters', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--block_size', type=int, default=256)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print(f"Loading: {args.base_ckpt}")
    ckpt = torch.load(args.base_ckpt, map_location='cpu')
    model_args = ckpt['model_args']
    print(f"Model args: {model_args}")

    cfg = GPTConfig(**model_args)
    model = GPT(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    stats = freeze_bottom_layers(model, args.freeze_ratio)
    print_freeze_summary(stats)

    data_path = f"{args.data_dir}/{args.dataset}/bin"
    train_data = np.memmap(f"{data_path}/train.bin", dtype=np.uint16, mode='r')
    val_data   = np.memmap(f"{data_path}/val.bin",   dtype=np.uint16, mode='r')
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.1
    )

    print(f"\nTraining {args.max_iters} iters, freeze_ratio={args.freeze_ratio}")
    print("="*60)
    t0 = time.time()

    for i in range(args.max_iters + 1):
        if i % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data,
                                   args.block_size, args.batch_size,
                                   args.eval_iters, device)
            print(f"iter {i:4d}/{args.max_iters}: train={losses['train']:.4f}, val={losses['val']:.4f}")
        if i == args.max_iters:
            break
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if i % 50 == 0 and i % args.eval_interval != 0:
            print(f"iter {i:4d}/{args.max_iters}: loss={loss.item():.4f}")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Frozen {stats['frozen_layers']}/{stats['total_layers']} layers")
    print(f"Trainable params: {stats['trainable_params_after']:,}")

    # Save model
    save_path = f'../out/checkpoints/freeze_{int(args.freeze_ratio*100)}.pt'
    torch.save({'model_state_dict': model.state_dict(), 'model_args': model_args, 'freeze_ratio': args.freeze_ratio, 'max_iters': args.max_iters}, save_path)
    print(f'Saved to {save_path}')

if __name__ == '__main__':
    main()
