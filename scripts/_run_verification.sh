#!/bin/bash
set -e
cd /workspace/parameter-golf

echo "=== Verification sweep started at $(date) ===" | tee logs/verification.log

# Clean previous
rm -rf artifacts/verify_sweep

# We need to handle the 10L anchor specially: use NaiveBaseline script + NUM_LAYERS=10 env var.
# The sweep's --extra-env applies to ALL anchors. Instead, we run two separate sweeps
# and merge results.

echo "--- Part A: NaiveBaseline + SmearGate3x (default env) ---" | tee -a logs/verification.log

python scripts/sweep_train_subsets.py \
    --candidates-dir artifacts/train_subsets \
    --records-dir records/track_10min_16mb \
    --candidate-ids single_shard0 single_shard5 \
    --anchor-scripts \
        records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
        records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py \
    --budget-mode tokens --budget-value 4000000 \
    --output-dir artifacts/verify_sweep \
    --timeout 900 \
    --proxy-val-manifest proxy_data/provisional_val_tune.json \
    --n-finalists 2 --seed 1337 \
    --max-workers 2 \
    2>&1 | tee -a logs/verification.log

echo "" | tee -a logs/verification.log
echo "--- Part B: Baseline10L (NaiveBaseline + NUM_LAYERS=10) ---" | tee -a logs/verification.log

# For the 10L anchor, we create runs manually with extra env
# Run single_shard0 × Baseline10L
python -c "
import sys, os, json, time
sys.path.insert(0, '.')
from pathlib import Path
from proxy_framework.train_subset_search import TrainSubsetCandidate, load_candidates, prepare_shard_dir, cleanup_shard_dir
from proxy_framework.budget import BudgetSpec
from scripts.sweep_train_subsets import _run_one

candidates = {c.candidate_id: c for c in load_candidates('artifacts/train_subsets')}
source = Path('data/datasets/fineweb10B_sp1024').resolve()
budget = BudgetSpec(mode='tokens', value=4_000_000, batch_tokens=32768)

for cid in ['single_shard0', 'single_shard5']:
    print(f'Running {cid} × Baseline10L ...')
    result = _run_one(
        candidate=candidates[cid],
        script_path='records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py',
        model_name='Baseline10L',
        output_dir=Path('artifacts/verify_sweep'),
        source_data_dir=source,
        budget_spec=budget,
        mem_fraction=1.0,
        conda_env='parameter-golf',
        seed=1337,
        timeout_sec=900,
        proxy_val_manifest='proxy_data/provisional_val_tune.json',
        extra_env={'NUM_LAYERS': '10'},
    )
    print(f'  Result: success={result[\"success\"]} train={result.get(\"train_proxy_bpb\")} eval={result.get(\"eval_proxy_bpb\")} vram={result.get(\"vram_status\")}')
    print()
" 2>&1 | tee -a logs/verification.log

echo "=== Verification sweep completed at $(date) ===" | tee -a logs/verification.log
echo "=== Audit the results ===" | tee -a logs/verification.log

# Audit
python -c "
import json
from pathlib import Path

sweep = Path('artifacts/verify_sweep')
print()
print('='*80)
print('VERIFICATION AUDIT')
print('='*80)

# Collect all results
print()
print(f'{\"Run\":<55} {\"Steps\":>6} {\"Tokens\":>10} {\"Train BPB\":>10} {\"Eval BPB\":>10} {\"VRAM\":>12} {\"Exh\":>10}')
print('-'*115)

for f in sorted(sweep.rglob('run_summary.json')):
    with open(f) as fp:
        s = json.load(fp)
    parts = f.relative_to(sweep).parts
    name = f'{parts[0]}/{parts[1]}'
    steps = s['optimizer_steps']
    tokens = s['tokens_processed']
    train_bpb = s['pre_quant_val_bpb']
    eval_bpb = s['proxy_val_tune_bpb']
    vram_alloc = s['peak_vram_allocated_gb']
    vram_eval = s['peak_vram_reserved_gb']  # reused for eval VRAM
    exh = s['budget_exhausted']

    vram_str = f'{vram_alloc:.3f}/{vram_eval:.3f}' if vram_alloc > 0 else f'UNK/{vram_eval:.3f}' if vram_eval > 0 else 'UNK/UNK'
    eval_str = f'{eval_bpb:.4f}' if eval_bpb > 0 else 'MISSING'
    print(f'{name:<55} {steps:>6} {tokens:>10,} {train_bpb:>10.4f} {eval_str:>10} {vram_str:>12} {exh:>10}')

# Check VRAM status files
print()
print('VRAM STATUS:')
for f in sorted(sweep.rglob('vram_status.txt')):
    parts = f.relative_to(sweep).parts
    name = f'{parts[0]}/{parts[1]}'
    content = f.read_text().strip()
    print(f'  {name}: {content}')

# Check token matching
tokens_set = set()
for f in sweep.rglob('run_summary.json'):
    with open(f) as fp:
        s = json.load(fp)
    tokens_set.add(s['tokens_processed'])
print(f'\\nToken matching: {tokens_set} {\"PASS\" if len(tokens_set)==1 else \"FAIL\"} ')

# Rankings
from collections import defaultdict
ref = {
    '2026-03-17_NaiveBaseline': 1.2244,
    'Baseline10L': 1.1748,  # approximate: 10L should be between baseline and smeargate
    '2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA': 1.1458,
}
print()
print('RANKINGS (reference: SmearGate < Baseline10L < NaiveBaseline):')
for metric_name, field in [('train-proxy', 'pre_quant_val_bpb'), ('eval-proxy', 'proxy_val_tune_bpb')]:
    print(f'\\n  {metric_name}:')
    results_by_cand = defaultdict(dict)
    for f in sorted(sweep.rglob('run_summary.json')):
        with open(f) as fp:
            s = json.load(fp)
        parts = f.relative_to(sweep).parts
        cand, model = parts[0], parts[1]
        val = s.get(field, 0)
        if val > 0:
            results_by_cand[cand][model] = val
    for cand, scores in sorted(results_by_cand.items()):
        order = sorted(scores, key=lambda m: scores[m])
        order_str = ' < '.join([f'{m.split(\"_\")[-1] if len(m) > 15 else m}({scores[m]:.4f})' for m in order])
        print(f'    {cand}: {order_str}')
" 2>&1 | tee -a logs/verification.log
