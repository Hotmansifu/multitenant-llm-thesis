# Train a bigger Gaussian attention model on OpenWebText toy data
# Target: ~124M params (GPT-2 small scale) with Gaussian kernel attention
# Hardware: H100 96GB — this will train fast
#
# Run with:
#   cd /home/coder/project/nanoGPT
#   PYTHONPATH=. PYTHONNOUSERSITE=1 ~/.pyenv/versions/3.11.9/bin/python train.py config/train_owt_gaussian_big.py
#

out_dir = 'out/owt_gaussian_big'
dataset = 'openwebtext_toy'
init_from = 'scratch'

# Model — GPT-2 small scale with Gaussian attention
n_layer = 12
n_head = 12
n_embd = 768
block_size = 256        # keep same as small model (toy data is small)
dropout = 0.0
bias = False
use_gaussian = True
gaussian_sigma2 = 16.0

# Training — adjusted for toy data size
batch_size = 64
gradient_accumulation_steps = 1
max_iters = 5000        # same as small model for fair comparison
eval_interval = 200
eval_iters = 50
log_interval = 100

# Optimizer
learning_rate = 3e-4
warmup_iters = 100
min_lr = 3e-5
lr_decay_iters = 5000
weight_decay = 0.10

# System
compile = False         # avoid compilation issues
dtype = 'float32'       # match small model for FHE comparison
device = 'cuda'

# Checkpointing
always_save_checkpoint = False

# Logging
wandb_log = False