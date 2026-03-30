#!/usr/bin/env bash
# Master orchestrator: runs all experiment phases sequentially.
# Survives SSH disconnects — launched via nohup.
#
# Usage:
#   nohup bash experiments/scripts/run_all.sh > experiments/logs/master.log 2>&1 &
#   # Check progress:
#   bash experiments/scripts/check_progress.sh
#   tail -f experiments/logs/master.log
#
set -uo pipefail

cd /root/parameter-golf

LOGDIR="experiments/logs"
RESULTS="experiments/results"
STATUS="experiments/logs/status.txt"
mkdir -p "$LOGDIR" "$RESULTS"

timestamp() { date -Iseconds; }

update_status() {
    echo "[$(timestamp)] $1" >> "$STATUS"
    echo "[$(timestamp)] $1"
}

wait_for_deps() {
    # Wait for all dependencies to be ready (data + packages)
    local MAX_WAIT=3600  # 1 hour max
    local WAITED=0
    local INTERVAL=30

    while [ $WAITED -lt $MAX_WAIT ]; do
        local READY=true

        # Check torch
        python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { READY=false; update_status "Waiting for torch+CUDA..."; }

        # Check data
        local SHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
        local VAL=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
        if [ "$SHARDS" -lt 5 ] || [ "$VAL" -lt 1 ]; then
            READY=false
            update_status "Waiting for data download... (train shards: $SHARDS, val: $VAL)"
        fi

        # Check sentencepiece
        python3 -c "import sentencepiece" 2>/dev/null || { READY=false; update_status "Waiting for sentencepiece..."; }

        # Check tokenizer
        if [ ! -f "data/tokenizers/fineweb_1024_bpe.model" ]; then
            READY=false
            update_status "Waiting for tokenizer file..."
        fi

        if $READY; then
            update_status "All dependencies ready!"
            return 0
        fi

        sleep $INTERVAL
        WAITED=$((WAITED + INTERVAL))
    done

    update_status "TIMEOUT waiting for dependencies after ${MAX_WAIT}s"
    return 1
}

check_flash_attn() {
    # flash_attn_interface is needed for SOTA script. Check if available.
    python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null
}

update_status "=== MASTER ORCHESTRATOR STARTED ==="
update_status "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"

# ======================================
# PHASE 0: Wait for environment
# ======================================
update_status "PHASE 0: Waiting for environment setup..."
if ! wait_for_deps; then
    update_status "PHASE 0: FAILED - dependencies not available"
    exit 1
fi

# Check flash-attn (may still be compiling)
if ! check_flash_attn; then
    update_status "PHASE 0: flash_attn_interface not yet available, waiting up to 30min..."
    for i in $(seq 1 60); do
        sleep 30
        if check_flash_attn; then
            update_status "PHASE 0: flash_attn_interface now available!"
            break
        fi
        update_status "PHASE 0: Still waiting for flash-attn... (${i}/60)"
    done
    if ! check_flash_attn; then
        update_status "PHASE 0: WARNING - flash_attn_interface not available. Will try fallback SDPA."
    fi
fi

update_status "PHASE 0: Environment ready"
python3 -c "
import torch, glob
print(f'  torch={torch.__version__} cuda={torch.version.cuda}')
print(f'  GPU={torch.cuda.get_device_name(0)}')
print(f'  Train shards: {len(glob.glob(\"data/datasets/fineweb10B_sp1024/fineweb_train_*.bin\"))}')
print(f'  Val shards: {len(glob.glob(\"data/datasets/fineweb10B_sp1024/fineweb_val_*.bin\"))}')
try:
    from flash_attn_interface import flash_attn_func
    print('  flash_attn_interface: YES')
except:
    print('  flash_attn_interface: NO (will need fallback)')
"

SOTA_SCRIPT="records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py"

# Common env vars for 64M matched-token budget
# 82 steps × 786,432 = 64.4M tokens
COMMON_ENV=(
    "ITERATIONS=82"
    "TRAIN_BATCH_TOKENS=786432"
    "TRAIN_SEQ_LEN=2048"
    "EVAL_SEQ_LEN=2048"
    "MAX_WALLCLOCK_SECONDS=0"
    "TTT_ENABLED=0"
    "WARMDOWN_ITERS=25"
    "WARMUP_STEPS=5"
    "VAL_LOSS_EVERY=80"
    "TRAIN_LOG_EVERY=10"
    "SEED=42"
)

run_experiment() {
    local NAME="$1"
    local SCRIPT="$2"
    shift 2
    # Remaining args are extra env overrides
    local EXPDIR="${RESULTS}/${NAME}"
    mkdir -p "$EXPDIR"

    if [ -f "${EXPDIR}/.done" ] && grep -q "^DONE" "${EXPDIR}/.done" 2>/dev/null; then
        update_status "  $NAME: already done (skipping)"
        return 0
    fi

    update_status "  $NAME: STARTING"

    # Build env
    local ENV_CMD=""
    for kv in "${COMMON_ENV[@]}"; do
        ENV_CMD="$ENV_CMD $kv"
    done
    for kv in "$@"; do
        ENV_CMD="$ENV_CMD $kv"
    done

    # Run with env vars
    env $ENV_CMD python3 "$SCRIPT" > "${EXPDIR}/train.log" 2>&1
    local EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        update_status "  $NAME: COMPLETED (exit 0)"
        grep -E "(pre_quant_val|final_int8|step_avg|Serialized|model_params)" \
            "${EXPDIR}/train.log" | tail -10 > "${EXPDIR}/metrics.txt" 2>/dev/null
        echo "DONE" > "${EXPDIR}/.done"
    else
        update_status "  $NAME: FAILED (exit $EXIT_CODE)"
        echo "FAILED:${EXIT_CODE}" > "${EXPDIR}/.done"
        # Save last 30 lines for debugging
        tail -30 "${EXPDIR}/train.log" > "${EXPDIR}/error.txt" 2>/dev/null
    fi

    # Move any artifacts
    for f in final_model.pt final_model.int8.ptz final_model.int6.lzma; do
        [ -f "$f" ] && mv "$f" "${EXPDIR}/" 2>/dev/null
    done

    return $EXIT_CODE
}

# ======================================
# PHASE 1: Baseline at 64M tokens
# ======================================
update_status "PHASE 1: Running baseline..."
run_experiment "baseline_64m" "$SOTA_SCRIPT" || true

if [ -f "$RESULTS/baseline_64m/metrics.txt" ]; then
    BPB=$(grep -o "pre_quant_val_bpb:[0-9.]*" "$RESULTS/baseline_64m/metrics.txt" 2>/dev/null | tail -1 | cut -d: -f2)
    update_status "PHASE 1: Baseline BPB=$BPB"
fi

# ======================================
# PHASE 2: Tier 0 compression analysis
# ======================================
if [ -f "$RESULTS/baseline_64m/final_model.pt" ]; then
    update_status "PHASE 2: Running compression analysis..."
    python3 experiments/scripts/02_tier0_compression.py 2>&1 | tee "${RESULTS}/tier0_compression/run.log"
    if [ -f "$RESULTS/tier0_compression/results.json" ]; then
        echo "DONE" > "$RESULTS/tier0_compression/.done"
        update_status "PHASE 2: Compression analysis complete"
    fi
else
    update_status "PHASE 2: Skipping (no baseline checkpoint)"
fi

# ======================================
# PHASE 3: Tier 1 env-var experiments
# ======================================
update_status "PHASE 3: Running Tier 1 env-var experiments..."

# Exp 1: Multi-token prediction (already implemented in SOTA)
run_experiment "exp_mtp_64m" "$SOTA_SCRIPT" \
    "MTP_NUM_HEADS=1" "MTP_LOSS_WEIGHT=0.15" || true

# Exp 2: Bigger bigram hash table
run_experiment "exp_bigram3k_64m" "$SOTA_SCRIPT" \
    "BIGRAM_VOCAB_SIZE=3072" || true

# Exp 3: 12 layers (more depth)
run_experiment "exp_12layer_64m" "$SOTA_SCRIPT" \
    "NUM_LAYERS=12" || true

# Exp 4: Wider model 576-dim
run_experiment "exp_wider576_64m" "$SOTA_SCRIPT" \
    "MODEL_DIM=576" "NUM_HEADS=9" "NUM_KV_HEADS=3" || true

# Exp 5: Longer warmdown
run_experiment "exp_warmdown40_64m" "$SOTA_SCRIPT" \
    "WARMDOWN_ITERS=40" || true

# Exp 6: More XSA layers
run_experiment "exp_xsa6_64m" "$SOTA_SCRIPT" \
    "XSA_LAST_N=6" || true

# Exp 7: Bigger VE dimension
run_experiment "exp_ve256_64m" "$SOTA_SCRIPT" \
    "VE_DIM=256" || true

echo "DONE" > "$RESULTS/.tier1_env_done"
update_status "PHASE 3: Tier 1 env-var experiments complete"

# ======================================
# PHASE 4: Tier 1 code-mod experiments
# ======================================
update_status "PHASE 4: Running Tier 1 code-mod experiments..."

FORKS="experiments/forks"
mkdir -p "$FORKS"

# Fork 1: SwiGLU (silu instead of leaky_relu)
if [ ! -f "$RESULTS/exp_swiglu_64m/.done" ]; then
    cp "$SOTA_SCRIPT" "$FORKS/train_gpt_swiglu.py"
    sed -i 's/x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)/x = F.silu(F.linear(x, up_w.to(x.dtype)))/' \
        "$FORKS/train_gpt_swiglu.py"
    run_experiment "exp_swiglu_64m" "$FORKS/train_gpt_swiglu.py" || true
fi

# Fork 2: GELU² activation
if [ ! -f "$RESULTS/exp_gelu2_64m/.done" ]; then
    cp "$SOTA_SCRIPT" "$FORKS/train_gpt_gelu2.py"
    sed -i 's/x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)/x = F.gelu(F.linear(x, up_w.to(x.dtype)))/' \
        "$FORKS/train_gpt_gelu2.py"
    run_experiment "exp_gelu2_64m" "$FORKS/train_gpt_gelu2.py" || true
fi

# Fork 3: Learnable negative slope
if [ ! -f "$RESULTS/exp_learned_act_64m/.done" ]; then
    cp "$SOTA_SCRIPT" "$FORKS/train_gpt_learned_act.py"
    python3 << 'PYEOF'
with open("experiments/forks/train_gpt_learned_act.py", "r") as f:
    code = f.read()

old_mlp = '''class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        # No CastedLinear -- weights come from banks
    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
        return F.linear(x.square(), down_w.to(x.dtype))'''

new_mlp = '''class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        mlp_dim = int(dim * mlp_mult)
        self.neg_slope = nn.Parameter(torch.full((mlp_dim,), 0.5, dtype=torch.float32))
    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        h = F.linear(x, up_w.to(x.dtype))
        slope = self.neg_slope.to(dtype=h.dtype)
        h = torch.where(h >= 0, h, h * slope)
        return F.linear(h.square(), down_w.to(x.dtype))'''

if old_mlp in code:
    code = code.replace(old_mlp, new_mlp)
    with open("experiments/forks/train_gpt_learned_act.py", "w") as f:
        f.write(code)
    print("Learnable activation fork created")
else:
    print("WARNING: Could not find MLP class to patch")
PYEOF
    run_experiment "exp_learned_act_64m" "$FORKS/train_gpt_learned_act.py" || true
fi

echo "DONE" > "$RESULTS/.tier1_code_done"
update_status "PHASE 4: Tier 1 code-mod experiments complete"

# ======================================
# FINAL: Summary
# ======================================
update_status ""
update_status "=========================================="
update_status "ALL PHASES COMPLETE"
update_status "=========================================="

echo ""
echo "=== FINAL RESULTS SUMMARY ==="
echo ""
printf "%-28s| %-18s| %s\n" "Experiment" "pre_quant_val_bpb" "Status"
echo "----------------------------|------------------|--------"
for dir in "$RESULTS"/baseline_64m "$RESULTS"/exp_*_64m; do
    [ -d "$dir" ] || continue
    NAME=$(basename "$dir")
    if [ -f "${dir}/.done" ]; then
        STAT=$(cat "${dir}/.done" | head -1)
    else
        STAT="INCOMPLETE"
    fi
    BPB="—"
    if [ -f "${dir}/metrics.txt" ]; then
        BPB=$(grep -o "pre_quant_val_bpb:[0-9.]*" "${dir}/metrics.txt" 2>/dev/null | tail -1 | cut -d: -f2)
    fi
    printf "%-28s| %-18s| %s\n" "$NAME" "${BPB:-—}" "$STAT"
done | tee "$RESULTS/final_summary.txt"

echo ""
update_status "Results saved to $RESULTS/final_summary.txt"
update_status "Master orchestrator finished"
