#!/bin/bash
# Phase 2: Train-subset sweep under provisional ranking lens
# Survives terminal disconnect via nohup.
#
# Matrix: 4 anchors × 6 train subsets = 24 runs at 16M tokens
# Anchors:
#   1. NaiveBaseline (9L, MLP=2, LR=0.04)
#   2. Baseline10L (NaiveBaseline + NUM_LAYERS=10)
#   3. SmearGate_3xMLP (MLP=3, SmearGate, BigramHash, LR=0.02)
#   4. MixedQuant (MLP=3, LR=0.02, MOM=0.99)
# Subsets:
#   single_shard0, single_shard5, single_shard7,
#   contiguous_5sh_off0, odd_5sh, dispersed_5sh_seed42
#
# Estimated: ~2.5 hours on A40 (3 parallel workers)

set -e
cd /workspace/parameter-golf
mkdir -p logs

LOGFILE="logs/phase2_sweep_$(date +%Y%m%d_%H%M%S).log"
echo "=== Phase 2 sweep started at $(date) ===" | tee "$LOGFILE"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"

rm -rf artifacts/phase2_sweep

# -----------------------------------------------------------------------
# Part A: NaiveBaseline + SmearGate3x + MixedQuant (default env)
# These 3 have different baked-in defaults, no env overrides needed
# -----------------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "--- Part A: 3 anchors with baked-in defaults (18 runs) ---" | tee -a "$LOGFILE"

python scripts/sweep_train_subsets.py \
    --candidates-dir artifacts/train_subsets \
    --records-dir records/track_10min_16mb \
    --candidate-ids single_shard0 single_shard5 single_shard7 contiguous_5sh_off0 odd_5sh dispersed_5sh_seed42 \
    --anchor-scripts \
        records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
        records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py \
        records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/train_gpt.py \
    --budget-mode tokens --budget-value 16000000 \
    --output-dir artifacts/phase2_sweep \
    --timeout 1200 \
    --n-finalists 3 --seed 1337 \
    --max-workers 3 \
    2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "--- Part A completed at $(date) ---" | tee -a "$LOGFILE"

# -----------------------------------------------------------------------
# Part B: Baseline10L (NaiveBaseline + NUM_LAYERS=10 env override)
# Run sequentially since we need per-run env override
# -----------------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "--- Part B: Baseline10L (6 runs, sequential) ---" | tee -a "$LOGFILE"

python -c "
import sys, json, time
sys.path.insert(0, '.')
from pathlib import Path
from proxy_framework.train_subset_search import load_candidates
from proxy_framework.budget import BudgetSpec
from scripts.sweep_train_subsets import _run_one

candidates = {c.candidate_id: c for c in load_candidates('artifacts/train_subsets')}
source = Path('data/datasets/fineweb10B_sp1024').resolve()
budget = BudgetSpec(mode='tokens', value=16_000_000, batch_tokens=32768)

subset_ids = ['single_shard0', 'single_shard5', 'single_shard7',
              'contiguous_5sh_off0', 'odd_5sh', 'dispersed_5sh_seed42']

for i, cid in enumerate(subset_ids, 1):
    print(f'[Part B] [{i}/{len(subset_ids)}] Running {cid} × Baseline10L ...')
    t0 = time.time()
    result = _run_one(
        candidate=candidates[cid],
        script_path='records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py',
        model_name='Baseline10L',
        output_dir=Path('artifacts/phase2_sweep'),
        source_data_dir=source,
        budget_spec=budget,
        mem_fraction=1.0,
        conda_env='parameter-golf',
        seed=1337,
        timeout_sec=1200,
        extra_env={'NUM_LAYERS': '10'},
    )
    elapsed = time.time() - t0
    status = 'OK' if result['success'] else 'FAIL'
    bpb = result.get('train_proxy_bpb')
    bpb_s = f'train={bpb:.4f}' if bpb else 'no-train'
    vram = result.get('vram_status', '?')
    print(f'  {status} ({bpb_s}, {elapsed:.0f}s, vram={vram})')
    print()
" 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "--- Part B completed at $(date) ---" | tee -a "$LOGFILE"

# -----------------------------------------------------------------------
# Audit and rank
# -----------------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "=== PHASE 2 AUDIT ===" | tee -a "$LOGFILE"

python -c "
import json, math
from pathlib import Path
from collections import defaultdict

sweep = Path('artifacts/phase2_sweep')

ref = {
    '2026-03-17_NaiveBaseline': 1.2244,
    'Baseline10L': 1.20,  # estimated (10L, same LR as baseline)
    '2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA': 1.1458,
    '2026-03-19_MixedQuant_Int6Int8_SlidingWindow': 1.1630,
}
ref_order = sorted(ref, key=lambda m: ref[m])  # best first

# Collect all results
results = defaultdict(dict)
all_summaries = []
for f in sorted(sweep.rglob('run_summary.json')):
    with open(f) as fp:
        s = json.load(fp)
    parts = f.relative_to(sweep).parts
    cand, model = parts[0], parts[1]
    results[cand][model] = s
    all_summaries.append(s)

# A. Accounting
print('='*80)
print('A. ACCOUNTING')
print('='*80)
tokens_set = set(s['tokens_processed'] for s in all_summaries)
steps_set = set(s['optimizer_steps'] for s in all_summaries)
print(f'  Runs: {len(all_summaries)}')
print(f'  Token counts: {tokens_set}')
print(f'  Step counts:  {steps_set}')
print(f'  Token match: {\"PASS\" if len(tokens_set) == 1 else \"FAIL\"}')

# B. VRAM
print()
print('='*80)
print('B. VRAM')
print('='*80)
vram_unknown = 0
for s in all_summaries:
    if s['peak_vram_allocated_gb'] == 0:
        vram_unknown += 1
    if s['peak_vram_allocated_gb'] > 10.0:
        print(f'  !! VRAM CAP VIOLATION: {s[\"run_name\"]} = {s[\"peak_vram_allocated_gb\"]:.3f} GB')
max_vram = max(s['peak_vram_allocated_gb'] for s in all_summaries)
print(f'  Max VRAM: {max_vram:.3f} GB')
print(f'  Unknown VRAM: {vram_unknown} of {len(all_summaries)}')
print(f'  Status: {\"PASS\" if max_vram <= 10.0 and vram_unknown == 0 else \"NEEDS REVIEW\"}')

# C. Rankings per train subset
print()
print('='*80)
print('C. TRAIN-PROXY RANKINGS PER SUBSET')
print('='*80)

# Compute ranking metrics for each candidate
from proxy_framework.metrics import spearman_rho, kendall_tau, pairwise_accuracy, top_1_agreement

candidates_order = ['single_shard0', 'single_shard5', 'single_shard7',
                    'contiguous_5sh_off0', 'odd_5sh', 'dispersed_5sh_seed42']
models = sorted(ref.keys())

def short_name(m):
    if m == '2026-03-17_NaiveBaseline': return 'Baseline'
    if m == 'Baseline10L': return 'Base10L'
    if m.startswith('2026-03-20_Int6'): return 'SmearGt'
    if m.startswith('2026-03-19_Mixed'): return 'MixedQ'
    return m[:8]

print(f'  Reference: {\" < \".join(short_name(m) for m in ref_order)}')
print()
print(f'  {\"Candidate\":<25} {\"Spear\":>7} {\"Kend\":>7} {\"Pair%\":>7} {\"Top1\":>5}  Order')
print(f'  {\"-\"*25} {\"-\"*7} {\"-\"*7} {\"-\"*7} {\"-\"*5}  {\"-\"*40}')

eval_data = []
for cid in candidates_order:
    if cid not in results:
        print(f'  {cid:<25} MISSING')
        continue
    proxy = {}
    for m in models:
        if m in results[cid]:
            bpb = results[cid][m]['pre_quant_val_bpb']
            if bpb > 0:
                proxy[m] = bpb
    common = sorted(set(proxy) & set(ref))
    if len(common) < 2:
        print(f'  {cid:<25} only {len(common)} models')
        continue

    p = [proxy[m] for m in common]
    r = [ref[m] for m in common]
    sp = spearman_rho(p, r)
    kt = kendall_tau(p, r)
    pa = pairwise_accuracy(p, r)
    t1 = top_1_agreement(p, r)
    proxy_order = sorted(common, key=lambda m: proxy[m])
    order_str = ' < '.join(short_name(m) for m in proxy_order)
    t1_str = 'Y' if t1 else 'N'
    print(f'  {cid:<25} {sp:>+7.3f} {kt:>+7.3f} {pa:>7.3f} {t1_str:>5}  {order_str}')
    eval_data.append({
        'candidate_id': cid,
        'spearman': sp,
        'kendall': kt,
        'pairwise': pa,
        'top1': t1,
        'composite': (sp + kt + pa) / 3,
    })

# D. Finalist selection
print()
print('='*80)
print('D. FINALISTS')
print('='*80)
ranked = sorted(eval_data, key=lambda x: x['composite'], reverse=True)
for i, e in enumerate(ranked[:3], 1):
    print(f'  #{i} {e[\"candidate_id\"]}')
    print(f'      Spearman:  {e[\"spearman\"]:+.3f}')
    print(f'      Kendall:   {e[\"kendall\"]:+.3f}')
    print(f'      Pairwise:  {e[\"pairwise\"]:.3f}')
    print(f'      Top-1:     {\"YES\" if e[\"top1\"] else \"NO\"}')
    print(f'      Composite: {e[\"composite\"]:.3f}')
    print()

# E. Stability
print('='*80)
print('E. STABILITY ASSESSMENT')
print('='*80)
if len(ranked) >= 2:
    gap = ranked[0]['composite'] - ranked[1]['composite']
    print(f'  Gap between #1 and #2: {gap:.3f}')
    if gap > 0.1:
        print(f'  Assessment: CLEAR WINNER — confident selection')
    elif gap > 0.03:
        print(f'  Assessment: MODERATE gap — selection is plausible but not decisive')
    else:
        print(f'  Assessment: NOISY — top candidates are nearly tied, low confidence')

# F. Per-model BPB table
print()
print('='*80)
print('F. RAW BPB TABLE')
print('='*80)
header = f'{\"\":>25}'
for m in models:
    header += f' {short_name(m):>10}'
print(header)
print('-' * (25 + 11 * len(models)))
for cid in candidates_order:
    if cid not in results:
        continue
    row = f'{cid:>25}'
    for m in models:
        if m in results[cid]:
            bpb = results[cid][m]['pre_quant_val_bpb']
            row += f' {bpb:>10.4f}' if bpb > 0 else f' {\"N/A\":>10}'
        else:
            row += f' {\"—\":>10}'
    print(row)
ref_row = f'{\"REFERENCE\":>25}'
for m in models:
    ref_row += f' {ref[m]:>10.4f}'
print(ref_row)

# Save finalists
import json
finalists_out = {
    'phase': 'phase2_provisional',
    'ranking_lens': 'train-proxy (pre_quant_val_bpb)',
    'budget': '16M matched tokens',
    'n_anchors': len(models),
    'n_candidates': len(eval_data),
    'finalists': ranked[:3],
    'all_candidates': ranked,
    'note': 'Selection under provisional ranking lens only. '
            'Eval-proxy not yet validated for custom architectures.',
}
out_dir = Path('artifacts/phase2_sweep/selection')
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / 'finalists.json', 'w') as f:
    json.dump(finalists_out, f, indent=2)
with open(out_dir / 'report.json', 'w') as f:
    json.dump(finalists_out, f, indent=2)
print(f'\\nSaved to {out_dir}/finalists.json')
" 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Phase 2 sweep completed at $(date) ===" | tee -a "$LOGFILE"
echo "Full log: $LOGFILE"
