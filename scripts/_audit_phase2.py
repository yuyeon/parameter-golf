#!/usr/bin/env python3
"""
Audit Phase 2 sweep results and produce finalist selection report.

Handles the mismatched-budget issue: when timeout killed runs at different
steps, we use only the step-400 val checkpoint BPB values for fair comparison
(the most common checkpoint across runs).

Also re-extracts val_bpb at a SPECIFIC step from the training logs rather
than relying on run_summary (which may record the wrong step's BPB for
timed-out runs).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proxy_framework.metrics import (
    kendall_tau,
    pairwise_accuracy,
    spearman_rho,
    top_1_agreement,
    top_k_overlap,
)


def extract_val_bpb_at_step(log_path: Path, target_step: int) -> float | None:
    """Extract val_bpb at a specific step from a training log."""
    if not log_path.exists():
        return None
    for line in log_path.read_text(errors="replace").splitlines():
        m = re.search(rf"step:{target_step}/\d+.*val_bpb:([\d.]+)", line)
        if m:
            return float(m.group(1))
    return None


def extract_all_val_checkpoints(log_path: Path) -> list[tuple[int, float]]:
    """Extract all (step, val_bpb) pairs from a training log."""
    if not log_path.exists():
        return []
    results = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = re.search(r"step:(\d+)/\d+.*val_bpb:([\d.]+)", line)
        if m:
            results.append((int(m.group(1)), float(m.group(2))))
    return results


def extract_peak_vram(log_path: Path) -> int:
    """Extract peak VRAM in MiB from training log."""
    if not log_path.exists():
        return 0
    for line in log_path.read_text(errors="replace").splitlines():
        m = re.search(r"peak memory allocated:\s*(\d+)\s*MiB", line)
        if m:
            return int(m.group(1))
    return 0


def main():
    sweep = Path("artifacts/phase2_sweep")

    # Reference ranking (leaderboard BPB)
    ref = {
        "2026-03-17_NaiveBaseline": 1.2244,
        "Baseline10L": 1.20,  # estimated
        "2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA": 1.1458,
        "2026-03-19_MixedQuant_Int6Int8_SlidingWindow": 1.1630,
    }
    ref_order = sorted(ref, key=lambda m: ref[m])

    def short(m):
        if m == "2026-03-17_NaiveBaseline": return "Baseline"
        if m == "Baseline10L": return "Base10L"
        if "SmearGate" in m: return "SmearGt"
        if "MixedQuant" in m: return "MixedQ"
        return m[:10]

    candidates = [
        "single_shard0", "single_shard5", "single_shard7",
        "contiguous_5sh_off0", "odd_5sh", "dispersed_5sh_seed42",
    ]
    models = sorted(ref.keys())

    # ===================================================================
    # Step 1: Find the most common val checkpoint step
    # ===================================================================
    step_counts: dict[int, int] = defaultdict(int)
    for cid in candidates:
        for model in models:
            log = sweep / cid / model / "train.log"
            checkpoints = extract_all_val_checkpoints(log)
            for step, _ in checkpoints:
                step_counts[step] += 1

    print("="*80)
    print("PHASE 2 AUDIT — MATCHED-BUDGET ANALYSIS")
    print("="*80)

    print("\nVal checkpoint frequency:")
    for step, count in sorted(step_counts.items()):
        print(f"  step {step}: {count} runs")

    # Use the highest step that has enough data (at least 2 models × 4 candidates)
    # Step 400 is the most common for runs that timed out
    TARGET_STEP = 400
    print(f"\nUsing step {TARGET_STEP} for matched-budget comparison")
    print(f"  (tokens at step {TARGET_STEP} = {TARGET_STEP * 32768:,})")

    # ===================================================================
    # Step 2: Extract BPB at the target step for all (candidate, model)
    # ===================================================================
    print("\n" + "="*80)
    print("RAW BPB TABLE (at step 400)")
    print("="*80)

    header = f"{'':>25}"
    for m in models:
        header += f"  {short(m):>10}"
    print(header)
    print("-" * (25 + 12 * len(models)))

    results: dict[str, dict[str, float]] = {}
    for cid in candidates:
        results[cid] = {}
        row = f"{cid:>25}"
        for model in models:
            log = sweep / cid / model / "train.log"
            bpb = extract_val_bpb_at_step(log, TARGET_STEP)
            if bpb is not None:
                results[cid][model] = bpb
                row += f"  {bpb:>10.4f}"
            else:
                row += f"  {'—':>10}"
        print(row)

    ref_row = f"{'REFERENCE':>25}"
    for m in models:
        ref_row += f"  {ref[m]:>10.4f}"
    print(ref_row)

    # ===================================================================
    # Step 3: Compute ranking metrics per candidate
    # ===================================================================
    print("\n" + "="*80)
    print("RANKING METRICS (step 400, matched budget)")
    print("="*80)

    print(f"\n  Reference order: {' < '.join(short(m) for m in ref_order)}")
    print(f"\n  {'Candidate':<25} {'Spear':>7} {'Kend':>7} {'Pair%':>7} "
          f"{'Top1':>5} {'TopK3':>6}  Proxy order")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*6}  {'-'*40}")

    eval_data = []
    for cid in candidates:
        proxy = results.get(cid, {})
        common = sorted(set(proxy) & set(ref))
        if len(common) < 2:
            print(f"  {cid:<25} only {len(common)} models — SKIP")
            continue

        p = [proxy[m] for m in common]
        r = [ref[m] for m in common]
        sp = spearman_rho(p, r)
        kt = kendall_tau(p, r)
        pa = pairwise_accuracy(p, r)
        t1 = top_1_agreement(p, r)
        tk = top_k_overlap(p, r, min(3, len(common)))

        proxy_order = sorted(common, key=lambda m: proxy[m])
        order_str = " < ".join(short(m) for m in proxy_order)
        t1_str = "Y" if t1 else "N"

        composite = (sp + kt + pa) / 3
        print(f"  {cid:<25} {sp:>+7.3f} {kt:>+7.3f} {pa:>7.3f} "
              f"{t1_str:>5} {tk:>6.3f}  {order_str}")

        eval_data.append({
            "candidate_id": cid,
            "spearman": sp,
            "kendall": kt,
            "pairwise": pa,
            "top1": t1,
            "top_k3": tk,
            "composite": composite,
            "n_models": len(common),
            "proxy_order": [short(m) for m in proxy_order],
        })

    # ===================================================================
    # Step 4: Finalist selection
    # ===================================================================
    ranked = sorted(eval_data, key=lambda x: x["composite"], reverse=True)

    print("\n" + "="*80)
    print("FINALIST SELECTION")
    print("="*80)

    for i, e in enumerate(ranked[:3], 1):
        print(f"\n  #{i} {e['candidate_id']}")
        print(f"      Spearman:  {e['spearman']:+.3f}")
        print(f"      Kendall:   {e['kendall']:+.3f}")
        print(f"      Pairwise:  {e['pairwise']:.3f}")
        print(f"      Top-1:     {'YES' if e['top1'] else 'NO'}")
        print(f"      Top-K(3):  {e['top_k3']:.3f}")
        print(f"      Composite: {e['composite']:.3f}")
        print(f"      N models:  {e['n_models']}")
        print(f"      Order:     {' < '.join(e['proxy_order'])}")

    # ===================================================================
    # Step 5: Stability assessment
    # ===================================================================
    print("\n" + "="*80)
    print("STABILITY ASSESSMENT")
    print("="*80)

    if len(ranked) >= 2:
        gap = ranked[0]["composite"] - ranked[1]["composite"]
        print(f"  Gap #1 vs #2: {gap:.3f}")
        if gap > 0.15:
            print(f"  CLEAR WINNER — high confidence in selection")
            confidence = "high"
        elif gap > 0.05:
            print(f"  MODERATE gap — selection is plausible")
            confidence = "moderate"
        else:
            print(f"  NOISY — top candidates nearly tied, low confidence")
            confidence = "low"

        # Check consistency: do the same candidates dominate across anchors?
        print(f"\n  Per-anchor consistency:")
        for model in models:
            model_scores = {}
            for cid in candidates:
                if model in results.get(cid, {}):
                    model_scores[cid] = results[cid][model]
            if model_scores:
                best_cid = min(model_scores, key=model_scores.get)
                spread = max(model_scores.values()) - min(model_scores.values())
                print(f"    {short(model)}: best subset = {best_cid} "
                      f"(spread = {spread:.4f})")
    else:
        confidence = "insufficient_data"

    # ===================================================================
    # Step 6: VRAM summary
    # ===================================================================
    print("\n" + "="*80)
    print("VRAM SUMMARY")
    print("="*80)
    for model in models:
        vrams = []
        for cid in candidates:
            log = sweep / cid / model / "train.log"
            v = extract_peak_vram(log)
            if v > 0:
                vrams.append(v)
        if vrams:
            print(f"  {short(model)}: {min(vrams)}-{max(vrams)} MiB "
                  f"({min(vrams)/1024:.2f}-{max(vrams)/1024:.2f} GB)")
        else:
            print(f"  {short(model)}: no VRAM data (timed out before logged)")

    # ===================================================================
    # Save report
    # ===================================================================
    report = {
        "phase": "phase2_provisional",
        "ranking_lens": "train-proxy (pre_quant_val_bpb at step 400)",
        "budget": f"{TARGET_STEP * 32768:,} tokens (step {TARGET_STEP})",
        "budget_note": "Most runs timed out at 1200s due to torch.compile "
                       "contention with 3 parallel workers. Step 400 is the "
                       "last common val checkpoint.",
        "n_anchors": len(models),
        "anchors": [short(m) for m in models],
        "n_candidates": len(eval_data),
        "confidence": confidence,
        "finalists": ranked[:3],
        "all_candidates": ranked,
        "note": "Selection under provisional train-proxy ranking lens only. "
                "Eval-proxy (submission-native subprocess) is now fixed but "
                "was not used in this sweep.",
    }
    out = sweep / "selection"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "phase2_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {out / 'phase2_report.json'}")


if __name__ == "__main__":
    main()
