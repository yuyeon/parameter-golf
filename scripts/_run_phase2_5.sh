#!/bin/bash
# Phase 3: Build proxy_val subsets + SmearGate/MixedQuant inversion study
# Runs under nohup. Survives terminal disconnect.

cd /workspace/parameter-golf
mkdir -p logs

LOGFILE="logs/phase3_$(date +%Y%m%d_%H%M%S).log"
echo "=== Phase 3 started: $(date) ===" | tee "$LOGFILE"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"

# -----------------------------------------------------------------------
# Step 1: Profile anchor models on full val set
# -----------------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "--- Step 1: Profile 4 anchors on full val set ---" | tee -a "$LOGFILE"

# Build manifest for profiling
cat > /tmp/profile_manifest.json << 'MANIFEST'
[
  {
    "name": "NaiveBaseline",
    "model_script": "records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py",
    "checkpoint": "artifacts/phase2_5_sweep/dispersed_5sh_seed42/2026-03-17_NaiveBaseline/final_model.pt",
    "seed": 1337
  },
  {
    "name": "SmearGate3x",
    "model_script": "records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py",
    "checkpoint": "artifacts/phase2_5_sweep/dispersed_5sh_seed42/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/final_model.pt",
    "seed": 1337
  },
  {
    "name": "MixedQuant",
    "model_script": "records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/train_gpt.py",
    "checkpoint": "artifacts/phase2_5_sweep/dispersed_5sh_seed42/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/final_model.pt",
    "seed": 1337
  },
  {
    "name": "Baseline10L",
    "model_script": "records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py",
    "checkpoint": "artifacts/phase2_5_sweep/dispersed_5sh_seed42/Baseline10L/final_model.pt",
    "seed": 1337
  }
]
MANIFEST

python -u scripts/profile_full_val.py \
    --manifest /tmp/profile_manifest.json \
    --output-dir artifacts/profiling \
    --batch-seqs 8 \
    --max-gb 10.0 \
    --seq-len 1024 \
    2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "--- Step 1 done: $(date) ---" | tee -a "$LOGFILE"

# -----------------------------------------------------------------------
# Step 2: Build proxy_val subsets
# -----------------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "--- Step 2: Build proxy_val_tune, proxy_val_audit, proxy_val_long ---" | tee -a "$LOGFILE"

python -u scripts/build_proxy_val.py \
    --profile-dir artifacts/profiling \
    --output-dir proxy_data \
    --n-tune 2000 \
    --n-audit 2000 \
    --n-long 500 \
    --strategy mixed \
    --seed 42 \
    --seq-len 1024 \
    --train-finalists artifacts/phase2_5_sweep/selection/phase2_5_report.json \
    2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "--- Step 2 done: $(date) ---" | tee -a "$LOGFILE"

# -----------------------------------------------------------------------
# Step 3: SmearGate vs MixedQuant inversion study
# Train both at 16M and 32M tokens on dispersed_5sh_seed42
# -----------------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "--- Step 3: Inversion study (SmearGate vs MixedQuant at 16M and 32M) ---" | tee -a "$LOGFILE"

python -u << 'PYEOF' 2>&1 | tee -a "$LOGFILE"
import sys, time, json
sys.path.insert(0, ".")
from pathlib import Path
from proxy_framework.train_subset_search import load_candidates
from proxy_framework.budget import BudgetSpec
from scripts.sweep_train_subsets import _run_one

candidates = {c.candidate_id: c for c in load_candidates("artifacts/train_subsets")}
cand = candidates["dispersed_5sh_seed42"]
source = Path("data/datasets/fineweb10B_sp1024").resolve()

# Check if proxy_val_tune was built
pv_manifest = "proxy_data/proxy_val_tune.json"
if not Path(pv_manifest).exists():
    pv_manifest = "proxy_data/provisional_val_tune.json"
    print(f"Using provisional val: {pv_manifest}")
else:
    print(f"Using final val: {pv_manifest}")

scripts = {
    "SmearGate3x": "records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py",
    "MixedQuant": "records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/train_gpt.py",
}

ref_bpb = {"SmearGate3x": 1.1458, "MixedQuant": 1.1630}
print(f"Reference: SmearGate3x ({ref_bpb['SmearGate3x']}) < MixedQuant ({ref_bpb['MixedQuant']})")
print(f"SmearGate should be BETTER (lower BPB) than MixedQuant")
print()

results = {}
for budget_tokens in [16_000_000, 32_000_000]:
    budget = BudgetSpec(mode="tokens", value=budget_tokens, batch_tokens=32768)
    target_steps = int(budget_tokens / 32768)
    print(f"=== Budget: {budget_tokens/1e6:.0f}M tokens ({target_steps} steps) ===")

    for model_name, script in scripts.items():
        tag = f"{model_name}_{budget_tokens//1000000}M"
        print(f"  Running {tag}...", flush=True)
        t0 = time.time()
        result = _run_one(
            candidate=cand,
            script_path=script,
            model_name=tag,
            output_dir=Path("artifacts/inversion_study"),
            source_data_dir=source,
            budget_spec=budget,
            mem_fraction=1.0,
            conda_env="parameter-golf",
            seed=1337,
            timeout_sec=3600,
            proxy_val_manifest=pv_manifest,
        )
        elapsed = time.time() - t0
        tbpb = result.get("train_proxy_bpb")
        ebpb = result.get("eval_proxy_bpb")
        ts = f"train={tbpb:.4f}" if tbpb else "no-train"
        es = f"eval={ebpb:.4f}" if ebpb else "no-eval"
        vram = result.get("vram_status", "?")
        steps = result.get("steps_completed", 0)
        print(f"    {ts}, {es}, steps={steps}, {elapsed:.0f}s, vram={vram}", flush=True)
        results[tag] = {"train": tbpb, "eval": ebpb, "steps": steps}

    # Compare
    sg_key = f"SmearGate3x_{budget_tokens//1000000}M"
    mq_key = f"MixedQuant_{budget_tokens//1000000}M"
    sg_t = results[sg_key]["train"]
    mq_t = results[mq_key]["train"]
    sg_e = results[sg_key]["eval"]
    mq_e = results[mq_key]["eval"]

    print()
    print(f"  Train-proxy: SmearGate={sg_t:.4f} vs MixedQuant={mq_t:.4f}", end="")
    if sg_t and mq_t:
        if sg_t < mq_t:
            print("  -> SmearGate wins (CORRECT)")
        else:
            print(f"  -> MixedQuant wins (INVERTED, gap={sg_t-mq_t:.4f})")

    if sg_e and mq_e:
        print(f"  Eval-proxy:  SmearGate={sg_e:.4f} vs MixedQuant={mq_e:.4f}", end="")
        if sg_e < mq_e:
            print("  -> SmearGate wins (CORRECT)")
        else:
            print(f"  -> MixedQuant wins (INVERTED, gap={sg_e-mq_e:.4f})")
    print()

# Summary
print()
print("=" * 70)
print("INVERSION STUDY SUMMARY")
print("=" * 70)
for budget_tokens in [16_000_000, 32_000_000]:
    label = f"{budget_tokens//1000000}M"
    sg = results[f"SmearGate3x_{label}"]
    mq = results[f"MixedQuant_{label}"]
    train_winner = "SmearGate" if sg["train"] < mq["train"] else "MixedQuant"
    eval_winner = "SmearGate" if (sg["eval"] and mq["eval"] and sg["eval"] < mq["eval"]) else "MixedQuant" if (sg["eval"] and mq["eval"]) else "?"
    train_gap = abs(sg["train"] - mq["train"]) if sg["train"] and mq["train"] else 0
    print(f"  {label}: train-proxy winner={train_winner} (gap={train_gap:.4f}), eval-proxy winner={eval_winner}")

print()
print("Reference: SmearGate should win (leaderboard BPB 1.1458 vs 1.1630)")

# Save
with open("artifacts/inversion_study/summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: artifacts/inversion_study/summary.json")
PYEOF

echo "" | tee -a "$LOGFILE"
echo "=== Phase 3 complete: $(date) ===" | tee -a "$LOGFILE"
