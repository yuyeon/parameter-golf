#!/bin/bash
# Scaling study v3: SmearGate vs MixedQuant at 8M/16M/32M/48M/64M
#
# KEY FIX: timeout = 400 + steps*0.35 + 400 (generous, accounts for compile)
# Sequential runs only. Reuses ONLY from artifacts/inversion_study where
# steps match exactly.

cd /workspace/parameter-golf
mkdir -p logs artifacts/scaling_study

LOGFILE="logs/scaling_study_v3_$(date +%Y%m%d_%H%M%S).log"
echo "=== Scaling study v3: $(date) ===" | tee "$LOGFILE"

python -u << 'PYEOF' 2>&1 | tee -a "$LOGFILE"
import sys, time, json, os
sys.path.insert(0, ".")
from pathlib import Path
from proxy_framework.train_subset_search import load_candidates
from proxy_framework.budget import BudgetSpec
from scripts.sweep_train_subsets import _run_one

candidates = {c.candidate_id: c for c in load_candidates("artifacts/train_subsets")}
cand = candidates["dispersed_5sh_seed42"]
source = Path("data/datasets/fineweb10B_sp1024").resolve()
pv_manifest = "proxy_data/proxy_val_tune.json"

scripts = {
    "SmearGate3x": "records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py",
    "MixedQuant": "records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/train_gpt.py",
}

# Only reuse from inversion_study (those were fully completed)
def find_valid(tag, target_steps):
    f = Path(f"artifacts/inversion_study/dispersed_5sh_seed42/{tag}/run_summary.json")
    if f.exists():
        with open(f) as fp:
            s = json.load(fp)
        if s["optimizer_steps"] == target_steps and s["pre_quant_val_bpb"] > 0:
            ep = s["proxy_val_tune_bpb"] if s["proxy_val_tune_bpb"] > 0 else None
            return {"train_proxy": s["pre_quant_val_bpb"], "eval_proxy": ep, "steps": target_steps}
    return None

budgets_m = [8, 16, 32, 48, 64]
results = {}

for budget_m in budgets_m:
    budget_tokens = budget_m * 1_000_000
    budget = BudgetSpec(mode="tokens", value=budget_tokens, batch_tokens=32768)
    target_steps = int(budget_tokens / 32768)
    # Generous timeout: 400s compile + 0.35s/step + 400s eval/margin
    timeout = 400 + int(target_steps * 0.35) + 400

    print(f"\n{'='*60}")
    print(f"Budget: {budget_m}M tokens ({target_steps} steps, timeout={timeout}s)")
    print(f"{'='*60}")

    for model_name, script in scripts.items():
        tag = f"{model_name}_{budget_m}M"
        existing = find_valid(tag, target_steps)
        if existing:
            results[tag] = existing
            e_str = f"eval={existing['eval_proxy']:.4f}" if existing['eval_proxy'] else "no-eval"
            print(f"  {tag}: REUSED (train={existing['train_proxy']:.4f}, {e_str})")
            continue

        print(f"  Running {tag} (timeout={timeout}s)...", flush=True)
        t0 = time.time()
        result = _run_one(
            candidate=cand,
            script_path=script,
            model_name=tag,
            output_dir=Path("artifacts/scaling_study"),
            source_data_dir=source,
            budget_spec=budget,
            mem_fraction=1.0,
            conda_env="parameter-golf",
            seed=1337,
            timeout_sec=timeout,
            proxy_val_manifest=pv_manifest,
        )
        elapsed = time.time() - t0
        tbpb = result.get("train_proxy_bpb")
        ebpb = result.get("eval_proxy_bpb")
        steps = result.get("steps_completed", 0)

        if steps < target_steps:
            print(f"    !! INCOMPLETE: {steps}/{target_steps} steps", flush=True)

        ts = f"train={tbpb:.4f}" if tbpb else "no-train"
        es = f"eval={ebpb:.4f}" if ebpb else "no-eval"
        print(f"    {ts}, {es}, steps={steps}/{target_steps}, {elapsed:.0f}s", flush=True)
        results[tag] = {"train_proxy": tbpb, "eval_proxy": ebpb, "steps": steps}

    sg = results.get(f"SmearGate3x_{budget_m}M", {})
    mq = results.get(f"MixedQuant_{budget_m}M", {})
    if sg.get("train_proxy") and mq.get("train_proxy"):
        dt = mq["train_proxy"] - sg["train_proxy"]
        print(f"\n  Delta (train): {dt:+.4f} ({'SG wins' if dt > 0 else 'MQ wins'})")
    if sg.get("eval_proxy") and mq.get("eval_proxy"):
        de = mq["eval_proxy"] - sg["eval_proxy"]
        print(f"  Delta (eval):  {de:+.4f} ({'SG wins' if de > 0 else 'MQ wins'})")

# Summary
print(f"\n\n{'='*80}")
print("SCALING STUDY RESULTS")
print(f"{'='*80}")
print(f"\n  {'Budget':>8} {'SG train':>10} {'MQ train':>10} {'D train':>10} {'SG eval':>10} {'MQ eval':>10} {'D eval':>10} {'Steps OK':>10}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

deltas_t = []
deltas_e = []
for bm in budgets_m:
    target = int(bm * 1_000_000 / 32768)
    sg = results.get(f"SmearGate3x_{bm}M", {})
    mq = results.get(f"MixedQuant_{bm}M", {})
    sg_ok = sg.get("steps", 0) == target
    mq_ok = mq.get("steps", 0) == target
    both_ok = "YES" if sg_ok and mq_ok else f"SG={'Y' if sg_ok else 'N'} MQ={'Y' if mq_ok else 'N'}"
    sgt = sg.get("train_proxy"); mqt = mq.get("train_proxy")
    sge = sg.get("eval_proxy"); mqe = mq.get("eval_proxy")
    dt = mqt - sgt if sgt and mqt and sg_ok and mq_ok else None
    de = mqe - sge if sge and mqe and sg_ok and mq_ok else None
    if dt is not None: deltas_t.append((bm, dt))
    if de is not None: deltas_e.append((bm, de))
    print(f"  {bm:>6}M {sgt if sgt else '—':>10} {mqt if mqt else '—':>10} "
          f"{f'{dt:+.4f}' if dt is not None else '—':>10} "
          f"{f'{sge:.4f}' if sge else '—':>10} {f'{mqe:.4f}' if mqe else '—':>10} "
          f"{f'{de:+.4f}' if de is not None else '—':>10} {both_ok:>10}")

if deltas_t:
    print(f"\n  Train-proxy delta trend (only matched-budget points):")
    for b, d in deltas_t:
        bar = "#" * min(40, int(abs(d) * 200))
        side = "<MQ" if d < 0 else "SG>"
        print(f"    {b:>4}M: {d:+.4f} {side} |{bar}")
    first, last = deltas_t[0][1], deltas_t[-1][1]
    if first < 0 and last < 0:
        v = "CLOSING" if abs(last) < abs(first) else "WIDENING"
        print(f"\n  Verdict: Inversion persists. Gap {v}.")
    elif first < 0 and last > 0:
        print(f"\n  Verdict: CROSSOVER — inversion resolves at higher budget.")
    elif first > 0 and last > 0:
        if any(d < 0 for _, d in deltas_t):
            print(f"\n  Verdict: NON-MONOTONIC — correct at endpoints but inverted in middle.")
        else:
            print(f"\n  Verdict: STABLE CORRECT — SmearGate wins at all budgets.")
    else:
        print(f"\n  Verdict: Inconclusive.")

with open("artifacts/scaling_study/results_v3.json", "w") as f:
    json.dump({"results": results, "deltas_train": deltas_t, "deltas_eval": deltas_e}, f, indent=2)
print(f"\nSaved: artifacts/scaling_study/results_v3.json")
PYEOF

echo "=== Done: $(date) ===" | tee -a "$LOGFILE"
