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
  Add --verbose / --quiet to control logging.
"""

import json
import logging
import os
import subprocess
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import main as pipeline
from evaluate import evaluate_query, avg, rouge_l, export_config
from main import (
    ALPHA, BETA, GAMMA, DELTA, THRESHOLD, MAX_RETRIES, DRIFT_CUTOFF,
    GROQ_MODEL, NLI_MODEL, EMBED_MODEL,
)
from db import build_or_load_index, collection_exists

OUT_DIR = Path("ablation_results")
OUT_DIR.mkdir(exist_ok=True)
RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


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
            logger.warning("%s failed: %s", qid, e)
    return rows


# ── Experiment 1: Component ablation ─────────────────────────────────────────
def ablation_components():
    logger.info("="*60)
    logger.info("  Experiment 1: Component Ablation")
    logger.info("="*60)

    index, queries = get_stress_data()

    configs = {
        "full        (α=0.30 β=0.30 γ=0.15 δ=0.25)": (0.30, 0.30, 0.15, 0.25),
        "no_faith    (α=0.00 β=0.45 γ=0.20 δ=0.35)": (0.00, 0.45, 0.20, 0.35),
        "no_cov      (α=0.45 β=0.00 γ=0.20 δ=0.35)": (0.45, 0.00, 0.20, 0.35),
        "no_contra   (α=0.40 β=0.40 γ=0.00 δ=0.20)": (0.40, 0.40, 0.00, 0.20),
        "no_relevance(α=0.45 β=0.45 γ=0.10 δ=0.00)": (0.45, 0.45, 0.10, 0.00),
        "faith_only  (α=1.00 β=0.00 γ=0.00 δ=0.00)": (1.00, 0.00, 0.00, 0.00),
        "cov_only    (α=0.00 β=1.00 γ=0.00 δ=0.00)": (0.00, 1.00, 0.00, 0.00),
        "relevance_only(α=0.00 β=0.00 γ=0.00 δ=1.00)":(0.00, 0.00, 0.00, 1.00),
    }

    results = {}
    for label, (a, b, g, d) in configs.items():
        logger.info("  Config: %s", label)
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
        logger.info("    avg_best_h=%.4f  avg_rouge_l=%.4f",
                    results[label]["avg_best_h_score"], results[label]["avg_rouge_l"])

    pipeline.ALPHA, pipeline.BETA, pipeline.GAMMA, pipeline.DELTA = 0.30, 0.30, 0.15, 0.25

    path = OUT_DIR / f"component_ablation_{RUN_ID}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Component ablation saved: %s", path)
    return results


# ── Experiment 2: Weight sensitivity grid ────────────────────────────────────
def ablation_weights():
    logger.info("="*60)
    logger.info("  Experiment 2: Weight Sensitivity Grid")
    logger.info("="*60)

    index, queries = get_stress_data()

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
        logger.info("  α=%.2f β=%.2f γ=%.2f δ=%.2f  → h=%.4f  rouge_l=%.4f",
                    a, b, g, d,
                    results[-1]["avg_best_h_score"], results[-1]["avg_rouge_l"])

    pipeline.ALPHA, pipeline.BETA, pipeline.GAMMA, pipeline.DELTA = 0.30, 0.30, 0.15, 0.25

    path = OUT_DIR / f"weight_sensitivity_{RUN_ID}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Weight sensitivity saved: %s", path)
    return results


# ── Experiment 3: Refinement ablation ────────────────────────────────────────
def ablation_refinement():
    logger.info("="*60)
    logger.info("  Experiment 3: Refinement Loop Ablation")
    logger.info("="*60)

    index, queries = get_stress_data()

    logger.info("  Running WITH refinement...")
    rows_with = run_with_weights(index, queries, 0.30, 0.30, 0.15, delta=0.25, no_refine=False)

    logger.info("  Running WITHOUT refinement (single pass)...")
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

    logger.info("  WITH    refinement → h=%.4f  rouge_l=%.4f",
                results["with_refinement"]["avg_best_h_score"],
                results["with_refinement"]["avg_rouge_l"])
    logger.info("  WITHOUT refinement → h=%.4f  rouge_l=%.4f",
                results["without_refinement"]["avg_best_h_score"],
                results["without_refinement"]["avg_rouge_l"])

    path = OUT_DIR / f"refinement_ablation_{RUN_ID}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Refinement ablation saved: %s", path)
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


# ── Export: Run config ────────────────────────────────────────────────────────
def _export_ablation_config(experiment: str):
    cfg = {
        "run_id":       RUN_ID,
        "git_commit":   _git_hash(),
        "alpha":        ALPHA,
        "beta":         BETA,
        "gamma":        GAMMA,
        "delta":        DELTA,
        "threshold":    THRESHOLD,
        "max_retries":  MAX_RETRIES,
        "drift_cutoff": DRIFT_CUTOFF,
        "nli_model":    NLI_MODEL,
        "embed_model":  EMBED_MODEL,
        "groq_model":   GROQ_MODEL,
        "experiment":   experiment,
        "dataset":      "stress",
    }
    path = OUT_DIR / f"config_{RUN_ID}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Config saved: %s", path)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = ArgumentParser(description="H_score ablation studies")
    parser.add_argument("experiment", nargs="?", default="all",
                        choices=["component", "weights", "refinement", "all"],
                        help="Which ablation to run (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-sentence NLI traces (DEBUG logging)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output (WARNING logging only)")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    if args.experiment == "component":
        ablation_components()
    elif args.experiment == "weights":
        ablation_weights()
    elif args.experiment == "refinement":
        ablation_refinement()
    else:
        ablation_components()
        ablation_weights()
        ablation_refinement()
        logger.info("All ablation results saved to %s/", OUT_DIR)

    _export_ablation_config(args.experiment)
