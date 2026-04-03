# H100 Environment Setup (from scratch)

> **Applies to: NVIDIA H100 instances only** (80GB HBM3, sm_90 / compute capability 9.0).
> Tested on Ubuntu 22.04 with Driver 580.x / CUDA 13.0, toolkit 12.8, Python 3.10.

This guide covers bootstrapping a bare H100 instance into a working training environment with Flash Attention 3 support.

## Prerequisites

Verify you're on an H100 before proceeding:

```bash
nvidia-smi | head -3          # confirm H100 + driver version
nvcc --version                 # confirm CUDA toolkit (12.8+ recommended)
python3 --version              # confirm Python 3.10+
```

Expected: NVIDIA H100 80GB HBM3, compute capability (9, 0).

## 1. Bootstrap pip

The instance may not have pip pre-installed:

```bash
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --user
export PATH="$HOME/.local/bin:$PATH"
pip3 --version
```

Add the PATH export to your shell profile (`~/.bashrc` or `~/.zshrc`) so it persists:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

## 2. Install PyTorch with CUDA 12.8

```bash
pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
# Expected: 2.11.0+cu128 12.8 NVIDIA H100 80GB HBM3
```

## 3. Install Python 3.10 dev headers (for Triton / torch.compile)

Triton's JIT compiler needs `Python.h`. If `python3.10-dev` is not installed and you lack sudo, extract the headers from the .deb manually:

```bash
cd /tmp
apt-get download libpython3.10-dev
apt-get download python3.10-dev

mkdir -p /tmp/pydev
dpkg-deb -x libpython3.10-dev_*.deb /tmp/pydev
dpkg-deb -x python3.10-dev_*.deb /tmp/pydev

# Copy headers to user-local include
mkdir -p ~/.local/include/python3.10
cp -r /tmp/pydev/usr/include/python3.10/* ~/.local/include/python3.10/

mkdir -p ~/.local/include/x86_64-linux-gnu/python3.10
cp -r /tmp/pydev/usr/include/x86_64-linux-gnu/python3.10/* ~/.local/include/x86_64-linux-gnu/python3.10/
```

Set the include paths (add to `~/.bashrc`):

```bash
export C_INCLUDE_PATH="$HOME/.local/include:$HOME/.local/include/python3.10"
export CPATH="$HOME/.local/include:$HOME/.local/include/python3.10"
```

Verify Triton can compile:

```bash
python3 -c "from triton.backends.nvidia.driver import CudaUtils; CudaUtils(); print('OK')"
```

## 4. Install Flash Attention 3

FA3 is Hopper-only (sm_90) and ships as a separate package from FA2. There are no official PyPI wheels — use the community prebuilt wheels:

```bash
pip3 install --user flash_attn_3 \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch2110
```

This installs `flash_attn_3` along with `einops` and `ninja`.

Verify:

```bash
python3 -c "
import torch, flash_attn_interface
q = torch.randn(2, 256, 8, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(2, 256, 8, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(2, 256, 8, 64, device='cuda', dtype=torch.bfloat16)
out = flash_attn_interface.flash_attn_func(q, k, v, causal=True)
print(f'FA3 OK — output shape: {out[0].shape}')
"
```

**FA3 API notes:**
- Import: `import flash_attn_interface`
- Input shape: `(batch, seqlen, nheads, headdim)` — NOT the `(batch, nheads, seqlen, headdim)` that PyTorch SDPA uses
- Supports `bf16` and `fp16`
- The function returns a tuple; the attention output is `out[0]`

## 5. Install project dependencies

```bash
pip3 install --user huggingface_hub sentencepiece pyyaml pytest
```

## 6. Download training data

Download 10 training shards + validation data + tokenizer:

```bash
python3 data/cached_challenge_fineweb.py --train-shards 10
```

This fetches from the HuggingFace dataset `willdepueoai/parameter-golf` and places files in:
- `data/datasets/fineweb10B_sp1024/` — train/val `.bin` shards
- `data/tokenizers/` — `fineweb_1024_bpe.model` + `.vocab`

## 7. Verify with a short training run

Run the baseline for 50 steps to confirm everything works end-to-end:

```bash
ITERATIONS=50 VAL_LOSS_EVERY=25 TRAIN_LOG_EVERY=10 MAX_WALLCLOCK_SECONDS=120 \
  python3 train_gpt.py
```

Expected output (approximate):
- ~327 ms/step on H100
- ~10.9 GB peak VRAM
- val_bpb ~2.58 at step 50

## Quick reference: all environment variables

Add these to `~/.bashrc` for a persistent setup:

```bash
export PATH="$HOME/.local/bin:$PATH"
export C_INCLUDE_PATH="$HOME/.local/include:$HOME/.local/include/python3.10"
export CPATH="$HOME/.local/include:$HOME/.local/include/python3.10"
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Python.h: No such file or directory` during training | Missing Python dev headers, Triton can't JIT-compile | Step 3 — extract headers from .deb |
| `x86_64-linux-gnu/python3.10/pyconfig.h: No such file` | Arch-specific headers not extracted | Ensure the `x86_64-linux-gnu/python3.10/` copy in Step 3 was done |
| `No module named 'torch'` | PyTorch not installed or wrong python | Step 2 — ensure `--user` install and PATH includes `~/.local/bin` |
| `flash_attn_interface` import error | FA3 not installed or wrong CUDA/torch version | Step 4 — check wheel URL matches your CUDA + torch versions |
| FA3 returns wrong shapes | Using SDPA layout `(B, H, S, D)` | FA3 expects `(B, S, H, D)` |
