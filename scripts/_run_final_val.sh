#!/bin/bash
# Final proxy_val construction and stability evaluation
#
# Step 1: Train NaiveBaseline + Baseline10L at 32M tokens on dispersed_5sh_seed42
# Step 2: Re-profile all 4 anchors at 32M on full val set
# Step 3: Rebuild proxy_val_tune, proxy_val_audit, proxy_val_long from 32M profiles
# Step 4: Evaluate stability: run all 4 anchors with eval-proxy on both tune and audit
# Step 5: Compare rankings: do tune and audit agree?

cd /workspace/parameter-golf
mkdir -p logs artifacts/val_stability

LOGFILE="logs/final_val_$(date +%Y%m%d_%H%M%S).log"
echo "=== Final proxy_val build + stability: $(date) ===" | tee "$LOGFILE"

# -----------------------------------------------------------------------
# Step 1: Train missing 32M checkpoints
# -----------------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "--- Step 1: Train NaiveBaseline + Baseline10L at 32M ---" | tee -a "$LOGFILE"

python -u << 'PYEOF' 2>&1 | tee -a "$LOGFILE"
import sys, time
sys.path.insert(0, ".")
from pathlib import Path
from proxy_framework.train_subset_search import load_candidates
from proxy_framework.budget import BudgetSpec
from scripts.sweep_train_subsets import _run_one

candidates = {c.candidate_id: c for c in load_candidates("artifacts/train_subsets")}
cand = candidates["dispersed_5sh_seed42"]
source = Path("data/datasets/fineweb10B_sp1024").resolve()
budget = BudgetSpec(mode="tokens", value=32_000_000, batch_tokens=32768)
timeout = 1200  # generous for sequential 32M

runs = [
    ("NaiveBaseline_32M",
     "records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py",
     None),
    ("Baseline10L_32M",
     "records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py",
     {"NUM_LAYERS": "10"}),
]

for model_name, script, extra in runs:
    out = Path("artifacts/val_stability")
    existing = out / "dispersed_5sh_seed42" / model_name / "run_summary.json"
    if existing.exists():
        print(f"  {model_name}: already exists, skipping")
        continue
    print(f"  Running {model_name}...", flush=True)
    t0 = time.time()
    result = _run_one(
        candidate=cand,
        script_path=script,
        model_name=model_name,
        output_dir=out,
        source_data_dir=source,
        budget_spec=budget,
        mem_fraction=1.0,
        conda_env="parameter-golf",
        seed=1337,
        timeout_sec=timeout,
        extra_env=extra,
    )
    elapsed = time.time() - t0
    bpb = result.get("train_proxy_bpb")
    steps = result.get("steps_completed", 0)
    print(f"    train={bpb:.4f}, steps={steps}/976, {elapsed:.0f}s", flush=True)

print("Step 1 done.", flush=True)
PYEOF

echo "--- Step 1 done: $(date) ---" | tee -a "$LOGFILE"

# -----------------------------------------------------------------------
# Step 2: Re-profile all 4 anchors at 32M
# -----------------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "--- Step 2: Profile 4 anchors (32M checkpoints) on full val ---" | tee -a "$LOGFILE"

# Build manifest pointing to 32M checkpoints
python -u -c "
import json
manifest = [
    {'name': 'NaiveBaseline_32M',
     'model_script': 'records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py',
     'checkpoint': 'artifacts/val_stability/dispersed_5sh_seed42/NaiveBaseline_32M/final_model.pt',
     'seed': 1337},
    {'name': 'Baseline10L_32M',
     'model_script': 'records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py',
     'checkpoint': 'artifacts/val_stability/dispersed_5sh_seed42/Baseline10L_32M/final_model.pt',
     'seed': 1337},
    {'name': 'SmearGate3x_32M',
     'model_script': 'records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py',
     'checkpoint': 'artifacts/inversion_study/dispersed_5sh_seed42/SmearGate3x_32M/final_model.pt',
     'seed': 1337},
    {'name': 'MixedQuant_32M',
     'model_script': 'records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/train_gpt.py',
     'checkpoint': 'artifacts/inversion_study/dispersed_5sh_seed42/MixedQuant_32M/final_model.pt',
     'seed': 1337},
]
with open('/tmp/profile_32m_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print('Manifest written')
" 2>&1 | tee -a "$LOGFILE"

# Clear old profiling data and re-profile
rm -rf artifacts/profiling_32m
mkdir -p artifacts/profiling_32m

python -u scripts/profile_full_val.py \
    --manifest /tmp/profile_32m_manifest.json \
    --output-dir artifacts/profiling_32m \
    --batch-seqs 8 \
    --max-gb 10.0 \
    --seq-len 1024 \
    2>&1 | tee -a "$LOGFILE"

echo "--- Step 2 done: $(date) ---" | tee -a "$LOGFILE"

# -----------------------------------------------------------------------
# Step 3: Rebuild proxy_val subsets from 32M profiles
# -----------------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "--- Step 3: Build proxy_val_tune/audit/long from 32M profiles ---" | tee -a "$LOGFILE"

# Save old subsets for comparison
cp proxy_data/proxy_val_tune.json proxy_data/proxy_val_tune_16m_backup.json 2>/dev/null
cp proxy_data/proxy_val_audit.json proxy_data/proxy_val_audit_16m_backup.json 2>/dev/null

python -u scripts/build_proxy_val.py \
    --profile-dir artifacts/profiling_32m \
    --output-dir proxy_data \
    --n-tune 2000 \
    --n-audit 2000 \
    --n-long 500 \
    --strategy mixed \
    --seed 42 \
    --seq-len 1024 \
    --train-finalists artifacts/phase2_5_sweep/selection/phase2_5_report.json \
    2>&1 | tee -a "$LOGFILE"

echo "--- Step 3 done: $(date) ---" | tee -a "$LOGFILE"

# -----------------------------------------------------------------------
# Step 4: Evaluate stability — eval-proxy on tune AND audit
# -----------------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "--- Step 4: Stability evaluation (tune vs audit) ---" | tee -a "$LOGFILE"

python -u << 'PYEOF' 2>&1 | tee -a "$LOGFILE"
import sys, json, subprocess, math
from pathlib import Path
sys.path.insert(0, ".")
from proxy_framework.metrics import spearman_rho, kendall_tau, pairwise_accuracy, top_1_agreement

REPO = Path(".")

ref = {
    "NaiveBaseline_32M": 1.2244,
    "Baseline10L_32M": 1.20,
    "SmearGate3x_32M": 1.1458,
    "MixedQuant_32M": 1.1630,
}

checkpoints = {
    "NaiveBaseline_32M": ("records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py",
                          "artifacts/val_stability/dispersed_5sh_seed42/NaiveBaseline_32M/final_model.pt"),
    "Baseline10L_32M": ("records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py",
                        "artifacts/val_stability/dispersed_5sh_seed42/Baseline10L_32M/final_model.pt"),
    "SmearGate3x_32M": ("records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py",
                         "artifacts/inversion_study/dispersed_5sh_seed42/SmearGate3x_32M/final_model.pt"),
    "MixedQuant_32M": ("records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/train_gpt.py",
                       "artifacts/inversion_study/dispersed_5sh_seed42/MixedQuant_32M/final_model.pt"),
}

def sn(m):
    return m.replace("_32M", "")

def eval_on_manifest(model_name, manifest_path):
    script, ckpt = checkpoints[model_name]
    out = Path(f"/tmp/eval_{model_name}_{Path(manifest_path).stem}.json")
    cmd = ["python", str(REPO / "scripts/_eval_proxy_subprocess.py"),
           "--script", script, "--checkpoint", ckpt,
           "--manifest", manifest_path, "--output", str(out), "--max-gb", "10.0"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(REPO))
    if out.exists():
        with open(out) as f:
            r = json.load(f)
        if r.get("status") == "ok":
            return r["proxy_val_bits_per_token"]
    return None

# Get train-proxy scores from run_summaries
train_scores = {}
for model in ref:
    for base in ["artifacts/val_stability", "artifacts/inversion_study"]:
        f = Path(f"{base}/dispersed_5sh_seed42/{model}/run_summary.json")
        if f.exists():
            with open(f) as fp:
                s = json.load(fp)
            if s["pre_quant_val_bpb"] > 0:
                train_scores[model] = s["pre_quant_val_bpb"]
                break

print("Train-proxy scores (from training):")
for m in sorted(ref):
    t = train_scores.get(m)
    print(f"  {sn(m):<20} {t:.4f}" if t else f"  {sn(m):<20} MISSING")

# Eval on tune and audit
print("\nEvaluating on proxy_val_tune and proxy_val_audit...")
tune_scores = {}
audit_scores = {}

for model in sorted(ref):
    print(f"  {sn(model)}: tune...", end="", flush=True)
    t = eval_on_manifest(model, "proxy_data/proxy_val_tune.json")
    tune_scores[model] = t
    print(f" {t:.4f}" if t else " FAIL", end="", flush=True)

    print(f"  audit...", end="", flush=True)
    a = eval_on_manifest(model, "proxy_data/proxy_val_audit.json")
    audit_scores[model] = a
    print(f" {a:.4f}" if a else " FAIL", flush=True)

# -----------------------------------------------------------------------
# Step 5: Compare rankings
# -----------------------------------------------------------------------
print(f"\n{'='*80}")
print("STABILITY REPORT: train-proxy vs eval-proxy-tune vs eval-proxy-audit")
print(f"{'='*80}")

print(f"\n  {'Model':<20} {'Ref BPB':>8} {'Train':>10} {'Tune':>10} {'Audit':>10}")
print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
models = sorted(ref)
for m in models:
    r = ref[m]
    t = train_scores.get(m)
    tu = tune_scores.get(m)
    au = audit_scores.get(m)
    print(f"  {sn(m):<20} {r:>8.4f} {t:>10.4f}" +
          (f" {tu:>10.4f}" if tu else "          —") +
          (f" {au:>10.4f}" if au else "          —"))

ref_order = sorted(models, key=lambda m: ref[m])
print(f"\n  Reference order: {' < '.join(sn(m) for m in ref_order)}")

for label, scores in [("Train-proxy", train_scores), ("Eval-tune", tune_scores), ("Eval-audit", audit_scores)]:
    valid = {m: s for m, s in scores.items() if s is not None}
    common = sorted(set(valid) & set(ref))
    if len(common) < 2:
        print(f"\n  {label}: insufficient data")
        continue
    p = [valid[m] for m in common]
    r = [ref[m] for m in common]
    sp = spearman_rho(p, r)
    kt = kendall_tau(p, r)
    pa = pairwise_accuracy(p, r)
    t1 = top_1_agreement(p, r)
    order = sorted(common, key=lambda m: valid[m])
    print(f"\n  {label}:")
    print(f"    Spearman={sp:+.3f}  Kendall={kt:+.3f}  Pairwise={pa:.3f}  Top1={'Y' if t1 else 'N'}")
    print(f"    Order: {' < '.join(sn(m) for m in order)}")

    # Flag proxy-fragile pairs
    sg = valid.get("SmearGate3x_32M")
    mq = valid.get("MixedQuant_32M")
    if sg and mq:
        delta = mq - sg
        correct = delta > 0
        print(f"    SmearGate vs MixedQuant: delta={delta:+.4f} {'CORRECT' if correct else 'INVERTED (proxy-fragile)'}")

# Tune vs audit agreement
print(f"\n{'='*80}")
print("TUNE vs AUDIT AGREEMENT")
print(f"{'='*80}")
tune_valid = {m: s for m, s in tune_scores.items() if s}
audit_valid = {m: s for m, s in audit_scores.items() if s}
common_ea = sorted(set(tune_valid) & set(audit_valid))
if len(common_ea) >= 2:
    tp = [tune_valid[m] for m in common_ea]
    ap = [audit_valid[m] for m in common_ea]
    sp = spearman_rho(tp, ap)
    kt = kendall_tau(tp, ap)
    tune_order = sorted(common_ea, key=lambda m: tune_valid[m])
    audit_order = sorted(common_ea, key=lambda m: audit_valid[m])
    print(f"  Spearman(tune, audit) = {sp:+.3f}")
    print(f"  Kendall(tune, audit)  = {kt:+.3f}")
    print(f"  Tune order:  {' < '.join(sn(m) for m in tune_order)}")
    print(f"  Audit order: {' < '.join(sn(m) for m in audit_order)}")
    if tune_order == audit_order:
        print(f"  STABLE: tune and audit produce identical ordering")
    else:
        print(f"  UNSTABLE: tune and audit disagree on ordering")

# Save
report = {
    "ref": ref, "train_scores": train_scores,
    "tune_scores": {k: v for k, v in tune_scores.items() if v},
    "audit_scores": {k: v for k, v in audit_scores.items() if v},
}
with open("artifacts/val_stability/stability_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved: artifacts/val_stability/stability_report.json")
PYEOF

echo "" | tee -a "$LOGFILE"
echo "=== Final proxy_val complete: $(date) ===" | tee -a "$LOGFILE"
