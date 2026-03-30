#!/usr/bin/env bash
# Environment setup for H100 validation experiments
# Run: bash experiments/scripts/00_setup_env.sh
set -euo pipefail

LOG="/root/parameter-golf/experiments/logs/00_setup.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== Setup started at $(date -Iseconds) ==="

cd /root/parameter-golf

# --- Install Python packages ---
PIP="pip3"
echo "[1/4] Installing PyTorch + CUDA..."
$PIP install --quiet torch numpy sentencepiece 2>&1
echo "[1/4] Done."

echo "[2/4] Installing flash-attn (may take a while to compile)..."
$PIP install --quiet flash-attn --no-build-isolation 2>&1 || {
    echo "WARN: flash-attn standard install failed, trying without isolation flag..."
    $PIP install --quiet flash-attn 2>&1 || echo "ERROR: flash-attn install failed. Some submissions may not work."
}
echo "[2/4] Done."

echo "[3/4] Installing optional deps..."
$PIP install --quiet zstandard scipy 2>&1 || true
echo "[3/4] Done."

# --- Download data ---
echo "[4/4] Downloading FineWeb data (sp1024)..."
python3 data/cached_challenge_fineweb.py --variant sp1024 2>&1
echo "[4/4] Done."

# --- Verify ---
echo ""
echo "=== Verification ==="
python3 -c "
import torch, glob, sentencepiece
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
shards = sorted(glob.glob('data/datasets/fineweb10B_sp1024/fineweb_train_*.bin'))
val = sorted(glob.glob('data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
print(f'Train shards: {len(shards)}')
print(f'Val shards: {len(val)}')
tok = sentencepiece.SentencePieceProcessor()
tok.Load('data/tokenizers/fineweb_1024_bpe.model')
print(f'Tokenizer vocab: {tok.GetPieceSize()}')
print('ALL CHECKS PASSED')
"
echo ""
echo "=== Setup completed at $(date -Iseconds) ==="
echo "SETUP_DONE" > /root/parameter-golf/experiments/logs/.setup_done
