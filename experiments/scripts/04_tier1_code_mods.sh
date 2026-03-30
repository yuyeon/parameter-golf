#!/usr/bin/env bash
# Tier 1 experiments that require code modifications to the SOTA script.
# Creates forked copies with specific changes, then runs them.
#
# Usage: nohup bash experiments/scripts/04_tier1_code_mods.sh > experiments/logs/tier1_code_mods.log 2>&1 &
set -uo pipefail

cd /root/parameter-golf

SOTA_SCRIPT="records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py"
RESULTS_BASE="experiments/results"
FORKS="experiments/forks"

# Common budget env
export ITERATIONS=82
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export MAX_WALLCLOCK_SECONDS=0
export TTT_ENABLED=0
export WARMDOWN_ITERS=25
export WARMUP_STEPS=5
export VAL_LOSS_EVERY=80
export TRAIN_LOG_EVERY=10
export SEED=42

timestamp() { date -Iseconds; }

# ---- Fork 1: SwiGLU activation ----
# Replace LeakyReLU(0.5)² with SwiGLU: silu(x_gate) * x_up
# SwiGLU needs a gate projection, but we can repurpose the up_w bank:
# Split up_w into two halves: gate and up.
# This changes MLP dimensions, so we need to adjust mlp_up_bank shape.
# SIMPLER: Use the existing up_w as gating and approximate SwiGLU via:
#   x = F.silu(F.linear(x, up_w)) * F.linear(x_orig, up_w)
# But that doubles compute. Instead, try GEGLU-like: split hidden dim in half.
#
# Actually the cleanest test: replace activation only, keep same dimensions.
# Test: F.silu(F.linear(x, up_w)).square() vs F.leaky_relu(..., 0.5).square()

echo "[$(timestamp)] Creating SwiGLU fork..."
mkdir -p "$FORKS"
cp "$SOTA_SCRIPT" "$FORKS/train_gpt_swiglu.py"
# Replace the activation line
sed -i 's/x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)/x = F.silu(F.linear(x, up_w.to(x.dtype)))/' \
    "$FORKS/train_gpt_swiglu.py"

echo "[$(timestamp)] Running SwiGLU experiment..."
EXPDIR="${RESULTS_BASE}/exp_swiglu_64m"
mkdir -p "$EXPDIR"
python3 "$FORKS/train_gpt_swiglu.py" 2>&1 | tee "${EXPDIR}/train.log"
EXIT_CODE=${PIPESTATUS[0]}
echo "[$(timestamp)] SwiGLU done (exit $EXIT_CODE)"
grep -E "(pre_quant_val|model_params)" "${EXPDIR}/train.log" | tail -5 > "${EXPDIR}/metrics.txt" 2>/dev/null
[ -f final_model.pt ] && mv final_model.pt "${EXPDIR}/"
echo "DONE:${EXIT_CODE}" > "${EXPDIR}/.done"


# ---- Fork 2: Learnable negative slope ----
# Replace fixed 0.5 with per-layer learnable parameter
echo ""
echo "[$(timestamp)] Creating learnable activation fork..."
cp "$SOTA_SCRIPT" "$FORKS/train_gpt_learned_act.py"

# We need to add a parameter to MLP.__init__ and use it in forward
python3 << 'PYEOF'
import re

with open("experiments/forks/train_gpt_learned_act.py", "r") as f:
    code = f.read()

# Replace MLP class with version that has learnable neg_slope
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
        # Learnable per-neuron negative slope, initialized to 0.5
        self.neg_slope = nn.Parameter(torch.full((mlp_dim,), 0.5, dtype=torch.float32))
    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        h = F.linear(x, up_w.to(x.dtype))
        # Learnable leaky relu: positive stays, negative scaled by learned slope
        slope = self.neg_slope.to(dtype=h.dtype)
        h = torch.where(h >= 0, h, h * slope)
        return F.linear(h.square(), down_w.to(x.dtype))'''

code = code.replace(old_mlp, new_mlp)

with open("experiments/forks/train_gpt_learned_act.py", "w") as f:
    f.write(code)

print("Learnable activation fork created successfully")
PYEOF

echo "[$(timestamp)] Running learnable activation experiment..."
EXPDIR="${RESULTS_BASE}/exp_learned_act_64m"
mkdir -p "$EXPDIR"
python3 "$FORKS/train_gpt_learned_act.py" 2>&1 | tee "${EXPDIR}/train.log"
EXIT_CODE=${PIPESTATUS[0]}
echo "[$(timestamp)] Learnable activation done (exit $EXIT_CODE)"
grep -E "(pre_quant_val|model_params)" "${EXPDIR}/train.log" | tail -5 > "${EXPDIR}/metrics.txt" 2>/dev/null
[ -f final_model.pt ] && mv final_model.pt "${EXPDIR}/"
echo "DONE:${EXIT_CODE}" > "${EXPDIR}/.done"


# ---- Fork 3: GELU² activation ----
echo ""
echo "[$(timestamp)] Creating GELU² fork..."
cp "$SOTA_SCRIPT" "$FORKS/train_gpt_gelu2.py"
sed -i 's/x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)/x = F.gelu(F.linear(x, up_w.to(x.dtype)))/' \
    "$FORKS/train_gpt_gelu2.py"

echo "[$(timestamp)] Running GELU² experiment..."
EXPDIR="${RESULTS_BASE}/exp_gelu2_64m"
mkdir -p "$EXPDIR"
python3 "$FORKS/train_gpt_gelu2.py" 2>&1 | tee "${EXPDIR}/train.log"
EXIT_CODE=${PIPESTATUS[0]}
echo "[$(timestamp)] GELU² done (exit $EXIT_CODE)"
grep -E "(pre_quant_val|model_params)" "${EXPDIR}/train.log" | tail -5 > "${EXPDIR}/metrics.txt" 2>/dev/null
[ -f final_model.pt ] && mv final_model.pt "${EXPDIR}/"
echo "DONE:${EXIT_CODE}" > "${EXPDIR}/.done"


# ---- Summary ----
echo ""
echo "================================================================"
echo "[$(timestamp)] CODE-MOD EXPERIMENTS COMPLETE"
echo "================================================================"
echo ""
for dir in ${RESULTS_BASE}/exp_swiglu_64m ${RESULTS_BASE}/exp_learned_act_64m ${RESULTS_BASE}/exp_gelu2_64m; do
    NAME=$(basename "$dir")
    if [ -f "${dir}/metrics.txt" ]; then
        BPB=$(grep -o "pre_quant_val_bpb:[0-9.]*" "${dir}/metrics.txt" 2>/dev/null | tail -1 | cut -d: -f2)
        printf "%-25s| %s\n" "$NAME" "${BPB:-MISSING}"
    else
        printf "%-25s| %s\n" "$NAME" "NO METRICS"
    fi
done
echo "CODE_MODS_DONE" > "${RESULTS_BASE}/.code_mods_done"
