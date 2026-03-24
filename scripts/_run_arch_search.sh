#!/bin/bash
# Architecture search: screen all unique submissions at 32M matched tokens
#
# Uses the frozen proxy harness:
#   proxy_train = dispersed_5sh_seed42
#   proxy_val_tune = proxy_data/proxy_val_tune.json (2000 seqs)
#   budget = 32M matched tokens
#
# Sequential runs to avoid contention. Each ~10-15 min.
# 13 submissions × ~12 min avg = ~2.5 hours
#
# Reuses existing 32M results where available.

cd /workspace/parameter-golf
mkdir -p logs artifacts/arch_search

LOGFILE="logs/arch_search_$(date +%Y%m%d_%H%M%S).log"
echo "=== Architecture search: $(date) ===" | tee "$LOGFILE"

python -u << 'PYEOF' 2>&1 | tee -a "$LOGFILE"
import sys, time, json, re
sys.path.insert(0, ".")
from pathlib import Path
from collections import defaultdict
from proxy_framework.train_subset_search import load_candidates
from proxy_framework.budget import BudgetSpec
from proxy_framework.metrics import spearman_rho, kendall_tau, pairwise_accuracy, top_1_agreement
from scripts.sweep_train_subsets import _run_one

candidates = {c.candidate_id: c for c in load_candidates("artifacts/train_subsets")}
cand = candidates["dispersed_5sh_seed42"]
source = Path("data/datasets/fineweb10B_sp1024").resolve()
budget = BudgetSpec(mode="tokens", value=32_000_000, batch_tokens=32768)
target_steps = 976
pv_manifest = "proxy_data/proxy_val_tune.json"

# All unique submissions (excluding LowerLR = duplicate of NaiveBaseline)
submissions = [
    ("2026-03-17_NaiveBaseline", "baseline_variant"),
    ("2026-03-18_FP16Embed_WD3600", "baseline_variant"),
    ("2026-03-18_LongContextSeq2048", "baseline_variant"),
    ("2026-03-19_10L_MixedPrecision", "baseline_variant"),
    ("2026-03-19_TrainingOptSeq4096", "baseline_variant"),
    ("2026-03-17_LoRA_TTT", "lora_ttt"),
    ("2026-03-19_SlidingWindowEval", "sliding_only"),
    ("2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit", "10layer"),
    ("2026-03-19_Seq2048_FP16Emb_TunedLR", "10layer_tuned"),
    ("2026-03-19_MLP3x_QAT_Int6_SlidingWindow", "mlp3x_qat"),
    ("2026-03-19_WarmdownQuantization", "quant_opt"),
    ("2026-03-19_MixedQuant_Int6Int8_SlidingWindow", "mlp3x_tuned"),
    ("2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA", "smeargate"),
]

# Reference BPB from leaderboard
ref = {}
for name, _ in submissions:
    sj = Path(f"records/track_10min_16mb/{name}/submission.json")
    if sj.exists():
        with open(sj) as f:
            d = json.load(f)
        bpb = d.get("val_bpb") or d.get("mean_val_bpb")
        if bpb:
            ref[name] = float(bpb)

# Check for reusable existing results
def find_existing(model_name):
    for base in ["artifacts/val_stability", "artifacts/inversion_study",
                 "artifacts/phase2_5_sweep", "artifacts/arch_search"]:
        f = Path(f"{base}/dispersed_5sh_seed42/{model_name}/run_summary.json")
        if f.exists():
            with open(f) as fp:
                s = json.load(fp)
            if s["optimizer_steps"] == target_steps and s["pre_quant_val_bpb"] > 0:
                return {
                    "train_proxy": s["pre_quant_val_bpb"],
                    "eval_proxy": s["proxy_val_tune_bpb"] if s["proxy_val_tune_bpb"] > 0 else None,
                    "steps": s["optimizer_steps"],
                    "vram": s["peak_vram_allocated_gb"],
                }
    return None

results = {}
n_total = len(submissions)
n_reused = 0
n_new = 0

for i, (name, family) in enumerate(submissions, 1):
    print(f"\n[{i}/{n_total}] {name} ({family})")

    # Check for reusable result (try multiple model name patterns)
    existing = None
    for tag in [name, f"{name}_32M"]:
        existing = find_existing(tag)
        if existing:
            break

    if existing:
        results[name] = {**existing, "family": family}
        e_str = f"eval={existing['eval_proxy']:.4f}" if existing.get('eval_proxy') else "no-eval"
        print(f"  REUSED: train={existing['train_proxy']:.4f}, {e_str}")
        n_reused += 1
        continue

    script = f"records/track_10min_16mb/{name}/train_gpt.py"
    print(f"  Running (timeout=1200s)...", flush=True)
    t0 = time.time()
    result = _run_one(
        candidate=cand,
        script_path=script,
        model_name=name,
        output_dir=Path("artifacts/arch_search"),
        source_data_dir=source,
        budget_spec=budget,
        mem_fraction=1.0,
        conda_env="parameter-golf",
        seed=1337,
        timeout_sec=1200,
        proxy_val_manifest=pv_manifest,
    )
    elapsed = time.time() - t0
    tbpb = result.get("train_proxy_bpb")
    ebpb = result.get("eval_proxy_bpb")
    steps = result.get("steps_completed", 0)
    vram = result.get("peak_vram_mib", 0)

    results[name] = {
        "train_proxy": tbpb,
        "eval_proxy": ebpb,
        "steps": steps,
        "vram": vram / 1024 if vram else 0,
        "family": family,
    }
    ts = f"train={tbpb:.4f}" if tbpb else "no-train"
    es = f"eval={ebpb:.4f}" if ebpb else "no-eval"
    ok = "OK" if steps == target_steps else f"INCOMPLETE ({steps}/{target_steps})"
    print(f"  {ok}: {ts}, {es}, {elapsed:.0f}s", flush=True)
    n_new += 1

print(f"\n\n{'='*80}")
print(f"SEARCH COMPLETE: {n_reused} reused, {n_new} new runs")
print(f"{'='*80}")

# ===================================================================
# Results table sorted by train-proxy
# ===================================================================
print(f"\n{'='*80}")
print("RESULTS BY TRAIN-PROXY (lower = better)")
print(f"{'='*80}\n")

ranked = sorted(
    [(n, r) for n, r in results.items() if r.get("train_proxy")],
    key=lambda x: x[1]["train_proxy"]
)

print(f"  {'Rank':>4} {'Submission':<55} {'Family':<16} {'Train':>8} {'Eval':>8} {'Ref':>8} {'VRAM':>6}")
print(f"  {'-'*4} {'-'*55} {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

for rank, (name, r) in enumerate(ranked, 1):
    t = f"{r['train_proxy']:.4f}" if r["train_proxy"] else "—"
    e = f"{r['eval_proxy']:.4f}" if r.get("eval_proxy") else "—"
    rb = f"{ref.get(name, 0):.4f}" if name in ref else "—"
    v = f"{r['vram']:.2f}" if r.get("vram") and r["vram"] > 0 else "?"
    print(f"  {rank:>4} {name:<55} {r['family']:<16} {t:>8} {e:>8} {rb:>8} {v:>6}")

# ===================================================================
# Results by family
# ===================================================================
print(f"\n{'='*80}")
print("TOP CANDIDATE PER FAMILY (by train-proxy)")
print(f"{'='*80}\n")

by_family = defaultdict(list)
for name, r in results.items():
    if r.get("train_proxy"):
        by_family[r["family"]].append((name, r))

for fam in sorted(by_family):
    members = sorted(by_family[fam], key=lambda x: x[1]["train_proxy"])
    best_name, best_r = members[0]
    ref_bpb = ref.get(best_name, 0)
    print(f"  {fam}:")
    print(f"    Best: {best_name}")
    print(f"    Train-proxy: {best_r['train_proxy']:.4f}  Ref: {ref_bpb:.4f}")
    if len(members) > 1:
        others = ", ".join(f"{n.split('_')[-1]}({r['train_proxy']:.4f})" for n, r in members[1:])
        print(f"    Others: {others}")
    print()

# ===================================================================
# Proxy-fragile detection
# ===================================================================
print(f"{'='*80}")
print("PROXY-FRAGILE COMPARISONS")
print(f"{'='*80}\n")

FRAGILE_THRESHOLD = 0.02
fragile_pairs = []

for i, (n1, r1) in enumerate(ranked):
    for n2, r2 in ranked[i+1:]:
        gap = abs(r1["train_proxy"] - r2["train_proxy"])
        if gap < FRAGILE_THRESHOLD and r1["family"] != r2["family"]:
            fragile_pairs.append((n1, n2, gap, r1["family"], r2["family"]))

if fragile_pairs:
    for n1, n2, gap, f1, f2 in sorted(fragile_pairs, key=lambda x: x[2]):
        s1 = n1.split("_", 3)[-1][:30] if len(n1) > 30 else n1
        s2 = n2.split("_", 3)[-1][:30] if len(n2) > 30 else n2
        print(f"  {s1} ({f1}) vs {s2} ({f2}): gap={gap:.4f} — PROXY-FRAGILE")
else:
    print("  None detected (all cross-family gaps > 0.02)")

# ===================================================================
# Ranking fidelity
# ===================================================================
print(f"\n{'='*80}")
print("OVERALL PROXY RANKING FIDELITY")
print(f"{'='*80}\n")

common = sorted(set(r[0] for r in ranked) & set(ref))
if len(common) >= 3:
    p = [results[m]["train_proxy"] for m in common]
    r_vals = [ref[m] for m in common]
    sp = spearman_rho(p, r_vals)
    kt = kendall_tau(p, r_vals)
    pa = pairwise_accuracy(p, r_vals)
    t1 = top_1_agreement(p, r_vals)
    print(f"  Models compared: {len(common)}")
    print(f"  Spearman:  {sp:+.3f}")
    print(f"  Kendall:   {kt:+.3f}")
    print(f"  Pairwise:  {pa:.3f}")
    print(f"  Top-1:     {'YES' if t1 else 'NO'}")

# Save
output = {
    "results": {n: {k: v for k, v in r.items()} for n, r in results.items()},
    "ranked": [(n, r) for n, r in ranked],
    "reference": ref,
    "fragile_pairs": fragile_pairs,
}
with open("artifacts/arch_search/search_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved: artifacts/arch_search/search_results.json")
PYEOF

echo "" | tee -a "$LOGFILE"
echo "=== Architecture search complete: $(date) ===" | tee -a "$LOGFILE"
