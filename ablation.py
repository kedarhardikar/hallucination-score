"""
ablation.py
-----------
Ablation studies for the H_score metric. Produces results for paper Section 4.

Three experiments:
  1. Component ablation   — remove each component one at a time (α=0, β=0, γ=0)
  2. Weight sensitivity   — vary α, β, γ, δ across a grid, measure ROUGE-L correlation
  3. Refinement ablation  — with vs without query refinement loop

Usage:
  python ablation.py component   → component ablation
  python ablation.py weights     → weight sensitivity grid
  python ablation.py refinement  → with vs without refinement
  python ablation.py all         → run all three
"""

import json
from pathlib import Path
from datetime import datetime

import main as pipeline
from evaluate import evaluate_query, avg, rouge_l
from db import build_or_load_index, collection_exists

OUT_DIR = Path("ablation_results")
OUT_DIR.mkdir(exist_ok=True)
RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Load stress queries (always available, no download needed) ────────────────
def get_stress_data():
    from dataset import get_stress_contexts_as_documents, get_stress_queries
    docs    = None if collection_exists(collection_name="stress") \
              else get_stress_contexts_as_documents()
    index   = build_or_load_index(docs, collection_name="stress")
    queries = get_stress_queries()
    return index, queries


# ── Run one pass with custom weights ─────────────────────────────────────────
def run_with_weights(index, queries, alpha, beta, gamma, delta=0.25, no_refine=False):
    """Temporarily override H_score weights, run all queries, restore."""
    pipeline.ALPHA = alpha
    pipeline.BETA  = beta
    pipeline.GAMMA = gamma
    pipeline.DELTA = delta

    graph = pipeline.build_rag_graph(index, no_refine=no_refine)
    rows  = []
    for qid, query, reference, risk in queries:
        try:
            row = evaluate_query(graph, qid, query, reference, risk)
            rows.append(row)
        except Exception as e:
            print(f"  ⚠ {qid} failed: {e}")
    return rows


# ── Experiment 1: Component ablation ─────────────────────────────────────────
def ablation_components():
    print("\n" + "="*60)
    print("  Experiment 1: Component Ablation")
    print("="*60)

    index, queries = get_stress_data()

    configs = {
        "full        (α=0.30 β=0.30 γ=0.15 δ=0.25)": (0.30, 0.30, 0.15, 0.25),
        "no_faith    (α=0.00 β=0.45 γ=0.20 δ=0.35)": (0.00, 0.45, 0.20, 0.35),  # remove faithfulness
        "no_cov      (α=0.45 β=0.00 γ=0.20 δ=0.35)": (0.45, 0.00, 0.20, 0.35),  # remove claim coverage
        "no_contra   (α=0.40 β=0.40 γ=0.00 δ=0.20)": (0.40, 0.40, 0.00, 0.20),  # remove contradiction
        "no_relevance(α=0.45 β=0.45 γ=0.10 δ=0.00)": (0.45, 0.45, 0.10, 0.00),  # remove answer relevance
        "faith_only  (α=1.00 β=0.00 γ=0.00 δ=0.00)": (1.00, 0.00, 0.00, 0.00),
        "cov_only    (α=0.00 β=1.00 γ=0.00 δ=0.00)": (0.00, 1.00, 0.00, 0.00),
        "relevance_only(α=0.00 β=0.00 γ=0.00 δ=1.00)":(0.00, 0.00, 0.00, 1.00),
    }

    results = {}
    for label, (a, b, g, d) in configs.items():
        print(f"\n  Config: {label}")
        rows = run_with_weights(index, queries, a, b, g, delta=d)
        results[label] = {
            "alpha":            a,
            "beta":             b,
            "gamma":            g,
            "delta":            d,
            "avg_best_h_score": avg([r["best_h_score"] for r in rows]),
            "avg_rouge_l":      avg([r["rouge_l"]      for r in rows]),
            "acceptance_rate":  round(sum(r["accepted"] for r in rows) / len(rows), 4),
            "by_stress_type":   _by_risk(rows),
        }
        print(f"    avg_best_h={results[label]['avg_best_h_score']} | "
              f"avg_rouge_l={results[label]['avg_rouge_l']}")

    # Restore defaults
    pipeline.ALPHA, pipeline.BETA, pipeline.GAMMA, pipeline.DELTA = 0.30, 0.30, 0.15, 0.25

    path = OUT_DIR / f"component_ablation_{RUN_ID}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Component ablation saved: {path}")
    return results


# ── Experiment 2: Weight sensitivity grid ────────────────────────────────────
def ablation_weights():
    print("\n" + "="*60)
    print("  Experiment 2: Weight Sensitivity Grid")
    print("="*60)

    index, queries = get_stress_data()

    # Grid: α + β + γ + δ = 1, step 0.25
    # Keep grid coarse to limit API calls (5^4 would be 625 combos — too many)
    steps = [0.0, 0.25, 0.50, 0.75, 1.0]
    weight_combos = [
        (a, b, g, round(1.0 - a - b - g, 2))
        for a in steps
        for b in steps
        for g in steps
        if 0.0 <= round(1.0 - a - b - g, 2) <= 1.0
    ]

    results = []
    for a, b, g, d in weight_combos:
        rows = run_with_weights(index, queries, a, b, g, delta=d)
        results.append({
            "alpha":            a,
            "beta":             b,
            "gamma":            g,
            "delta":            d,
            "avg_best_h_score": avg([r["best_h_score"] for r in rows]),
            "avg_rouge_l":      avg([r["rouge_l"]      for r in rows]),
            "acceptance_rate":  round(sum(r["accepted"] for r in rows) / len(rows), 4),
        })
        print(f"  α={a} β={b} γ={g} δ={d}  → "
              f"h={results[-1]['avg_best_h_score']} | rouge_l={results[-1]['avg_rouge_l']}")

    # Restore defaults
    pipeline.ALPHA, pipeline.BETA, pipeline.GAMMA, pipeline.DELTA = 0.30, 0.30, 0.15, 0.25

    path = OUT_DIR / f"weight_sensitivity_{RUN_ID}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Weight sensitivity saved: {path}")
    return results


# ── Experiment 3: Refinement ablation ────────────────────────────────────────
def ablation_refinement():
    print("\n" + "="*60)
    print("  Experiment 3: Refinement Loop Ablation")
    print("="*60)

    index, queries = get_stress_data()

    print("\n  Running WITH refinement...")
    rows_with = run_with_weights(index, queries, 0.30, 0.30, 0.15, delta=0.25, no_refine=False)

    print("\n  Running WITHOUT refinement (single pass)...")
    rows_without = run_with_weights(index, queries, 0.30, 0.30, 0.15, delta=0.25, no_refine=True)

    results = {
        "with_refinement": {
            "avg_best_h_score": avg([r["best_h_score"] for r in rows_with]),
            "avg_rouge_l":      avg([r["rouge_l"]      for r in rows_with]),
            "avg_retries":      avg([r["retries"]      for r in rows_with]),
            "acceptance_rate":  round(sum(r["accepted"] for r in rows_with) / len(rows_with), 4),
            "by_stress_type":   _by_risk(rows_with),
        },
        "without_refinement": {
            "avg_best_h_score": avg([r["best_h_score"] for r in rows_without]),
            "avg_rouge_l":      avg([r["rouge_l"]      for r in rows_without]),
            "avg_retries":      avg([r["retries"]      for r in rows_without]),
            "acceptance_rate":  round(sum(r["accepted"] for r in rows_without) / len(rows_without), 4),
            "by_stress_type":   _by_risk(rows_without),
        },
    }

    print(f"\n  WITH    refinement → h={results['with_refinement']['avg_best_h_score']} | "
          f"rouge_l={results['with_refinement']['avg_rouge_l']}")
    print(f"  WITHOUT refinement → h={results['without_refinement']['avg_best_h_score']} | "
          f"rouge_l={results['without_refinement']['avg_rouge_l']}")

    path = OUT_DIR / f"refinement_ablation_{RUN_ID}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Refinement ablation saved: {path}")
    return results


# ── Helper: group rows by stress type ────────────────────────────────────────
def _by_risk(rows):
    risks = sorted({r["risk"] for r in rows})
    out   = {}
    for risk in risks:
        subset = [r for r in rows if r["risk"] == risk]
        out[risk] = {
            "avg_best_h_score": avg([r["best_h_score"] for r in subset]),
            "avg_rouge_l":      avg([r["rouge_l"]      for r in subset]),
            "acceptance_rate":  round(sum(r["accepted"] for r in subset) / len(subset), 4),
        }
    return out


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    experiment = sys.argv[1] if len(sys.argv) > 1 else "all"

    if experiment == "component":
        ablation_components()
    elif experiment == "weights":
        ablation_weights()
    elif experiment == "refinement":
        ablation_refinement()
    else:  # all
        ablation_components()
        ablation_weights()
        ablation_refinement()
        print(f"\n✅ All ablation results saved to {OUT_DIR}/")
