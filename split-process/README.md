# split-process/ — RQ3

Two-process split inference. The transformer is split so that the trusted client
holds the embeddings, the first LayerNorm, and attention, while the server holds
the second LayerNorm, the MLP, the final LayerNorm, and the language-model head.
The client never reveals raw tokens to the server, and the server produces the
output logits.

Three transports are measured, in increasing order of protection and cost:

- **D1 — shared memory.** Client and server exchange hidden states through
  `/dev/shm` on the same machine. Lowest overhead.
- **D2 — TCP sockets.** Same split, hidden states sent over a socket, modelling
  client and server on different hosts.
- **D3 — TCP + Paillier PHE.** Builds on D2; hidden-state transfers are
  encrypted and the server computes on ciphertext. Highest overhead.

## What each script does

- `demo1_shm.py` — **D1.** Shared-memory split inference; loads `ckpt.pt` from
  the same folder and launches the worker process itself.
- `demo2_net.py` — **D2.** The same split over TCP sockets.
- `demo3_phe.py` — **D3.** Builds on `demo2_net.py`, adding Paillier encryption
  on the transfers; runs the monolithic baseline and the plaintext TCP split for
  comparison.
- `model.py` — the model definition the client and worker both load.

## Headline numbers

Overheads relative to the plaintext single-process baseline:

- D1 (shared memory): ~45.8% +/- 2.6%.
- D2 (TCP sockets): ~491.1% +/- 11.5%.
- D3 (Paillier PHE): ~56,000x (extrapolated from the measured per-element
  encrypt/decrypt rate and the number of transferred elements; see the thesis).

The cheaper LoRA-only encryption alternative (~16,383x, measured) lives in
`gaussian-bitnet/` (`paillier_s3_demo.py`), since it depends on the LoRA path.

## Running

```
python demo1_shm.py     # D1: launches its own worker
python demo2_net.py     # D2
python demo3_phe.py     # D3
```

All three load `ckpt.pt` from this folder (a trained BitNet + Gaussian
checkpoint). Place the checkpoint here, or edit the `CKPT` path at the top of
the script. `demo3_phe.py` requires `phe` (python-paillier). The D1/D2 timing
numbers in the thesis were taken on bare metal to avoid VM scheduling noise on
those latency-sensitive measurements.
