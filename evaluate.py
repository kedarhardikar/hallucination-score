"""
evaluate.py
-----------
Batch evaluation harness for the LangGraph RAG + H_score pipeline.
Runs all queries, logs per-query metrics, and exports:
  - results_<RUN_ID>.csv       → raw per-query results
  - summary_<RUN_ID>.json      → aggregated stats by hallucination_risk tier
  - paper_table_<RUN_ID>.tex   → LaTeX-ready table
  - config_<RUN_ID>.json       → full run configuration for reproducibility

Usage:
  python evaluate.py stress [--verbose] [--quiet] [--no-refine] [--seed 42] [--n-samples 50]
  python evaluate.py hotpotqa
"""

import csv
import json
import logging
import os
import subprocess
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

# Configure logging before importing pipeline modules so all child loggers inherit the level.
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from dataset import (
    get_stress_contexts_as_documents, get_stress_queries,
    load_hotpotqa, get_hotpotqa_documents, get_hotpotqa_queries,
)
from main import (
    build_rag_graph, RAGState, Settings,
    ALPHA, BETA, GAMMA, DELTA, THRESHOLD, MAX_RETRIES, DRIFT_CUTOFF,
    GROQ_MODEL, NLI_MODEL, EMBED_MODEL,
)
from db import build_or_load_index, collection_exists, save_queries, load_queries, queries_cached

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)
RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Shared utility ────────────────────────────────────────────────────────────
def avg(lst: list) -> float:
    return round(sum(lst) / len(lst), 4) if lst else 0.0


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


# ── ROUGE-L (lightweight, no extra deps) ─────────────────────────────────────
def lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l(hypothesis: str, reference: str) -> float:
    h, r = hypothesis.lower().split(), reference.lower().split()
    if not h or not r:
        return 0.0
    lcs  = lcs_length(h, r)
    prec = lcs / len(h)
    rec  = lcs / len(r)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


# ── LaTeX escaping ────────────────────────────────────────────────────────────
_LATEX_SPECIAL = str.maketrans({
    '&':  r'\&',
    '%':  r'\%',
    '$':  r'\$',
    '#':  r'\#',
    '_':  r'\_',
    '{':  r'\{',
    '}':  r'\}',
    '~':  r'\textasciitilde{}',
    '^':  r'\textasciicircum{}',
    '\\': r'\textbackslash{}',
})

def escape_latex(text: str) -> str:
    return str(text).translate(_LATEX_SPECIAL)


# ── Per-query evaluation ──────────────────────────────────────────────────────
def evaluate_query(graph, qid: str, query: str, reference: str, risk: str) -> dict:
    # Stress queries are filtered to their own source_id so each query only
    # searches its own designated context set — prevents cross-contamination
    # from other items' contexts bleeding into the retrieval results.
    # HotpotQA queries search the full collection (multi-hop requires it).
    is_stress = risk != "hotpotqa"

    initial: RAGState = {
        "query":          query,
        "original_query": query,
        "retrieved_docs": [],
        "answer":         "",
        "h_score":          0.0,
        "faithfulness":     0.0,
        "claim_coverage":   0.0,
        "contradiction":    0.0,
        "answer_relevance": 0.0,
        "retries":          0,
        "best_answer":      None,
        "best_h_score":     -1.0,
        "final_answer":     None,
        "filter_source_id": qid if is_stress else None,
        "drift_rejected":   False,
    }

    t0      = time.time()
    result  = graph.invoke(initial)
    latency = round(time.time() - t0, 2)

    answer = result["final_answer"] or result["answer"]
    rl     = rouge_l(answer, reference)

    return {
        "id":             qid,
        "query":          query,
        "risk":           risk,
        "answer":         answer,
        "reference":      reference,
        "h_score":          result["h_score"],
        "best_h_score":     result["best_h_score"],
        "faithfulness":     result["faithfulness"],
        "claim_coverage":   result["claim_coverage"],
        "contradiction":    result["contradiction"],
        "answer_relevance": result["answer_relevance"],
        "retries":          result["retries"],
        "drift_rejected":   result["drift_rejected"],
        "rouge_l":        round(rl, 4),
        "latency_s":      latency,
        "accepted":       result["best_h_score"] >= THRESHOLD,
    }


# ── Batch runner ──────────────────────────────────────────────────────────────
def run_evaluation(mode: str = "stress", hotpotqa_n: int = 50,
                   no_refine: bool = False, seed: int = 42):
    """
    mode = "stress"   → adversarial stress-test dataset
    mode = "hotpotqa" → HotpotQA distractor split (real multi-hop QA)
    mode = "both"     → stress + hotpotqa combined
    """
    logger.info("="*60)
    logger.info("  RAG Hallucination Evaluation — Run %s  [%s]", RUN_ID, mode)
    logger.info("="*60)

    if mode == "stress":
        collection_name = "stress"
        queries = get_stress_queries()
        docs    = None if collection_exists(collection_name=collection_name) \
                  else get_stress_contexts_as_documents()

    elif mode == "hotpotqa":
        collection_name = "hotpotqa"
        if collection_exists(collection_name=collection_name) and queries_cached(collection_name=collection_name):
            logger.info("[DB] Using cached index + queries — skipping HotpotQA download")
            docs    = None
            queries = load_queries(collection_name=collection_name)
        else:
            samples = load_hotpotqa(split="validation", n_samples=hotpotqa_n, seed=seed)
            docs    = get_hotpotqa_documents(samples)
            queries = get_hotpotqa_queries(samples)
            save_queries(queries, collection_name=collection_name)
            logger.info("Loaded %d HotpotQA samples → %d documents", len(samples), len(docs))

    else:  # both
        collection_name = "both"
        if collection_exists(collection_name=collection_name) and queries_cached(collection_name=collection_name):
            logger.info("[DB] Using cached index + queries — skipping HotpotQA download")
            docs    = None
            queries = load_queries(collection_name=collection_name)
        else:
            samples = load_hotpotqa(split="validation", n_samples=hotpotqa_n, seed=seed)
            docs    = get_stress_contexts_as_documents() + get_hotpotqa_documents(samples)
            queries = get_stress_queries() + get_hotpotqa_queries(samples)
            save_queries(queries, collection_name=collection_name)
            logger.info("Loaded %d stress + %d HotpotQA queries",
                        len(get_stress_queries()), len(samples))

    index = build_or_load_index(docs, collection_name=collection_name)
    graph = build_rag_graph(index, no_refine=no_refine)
    if no_refine:
        logger.info("[Ablation] Query refinement DISABLED — single-pass only")
    rows  = []

    try:
        for i, (qid, query, reference, risk) in enumerate(queries, 1):
            logger.info("[%d/%d] %s | risk=%s | %s...", i, len(queries), qid, risk, query[:60])
            try:
                row = evaluate_query(graph, qid, query, reference, risk)
                rows.append(row)
                logger.info("        best_H=%.4f  last_H=%.4f  ROUGE-L=%.4f  "
                            "retries=%d  drift_rej=%s  latency=%.1fs",
                            row["best_h_score"], row["h_score"], row["rouge_l"],
                            row["retries"], row["drift_rejected"], row["latency_s"])
            except Exception as e:
                logger.warning("Query %s failed: %s", qid, e)
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving partial results...")
    finally:
        if rows:
            export_csv(rows)
            export_summary(rows)
            export_latex(rows)
            export_config(mode=mode, n_samples=hotpotqa_n, no_refine=no_refine, seed=seed)
            logger.info("Results saved to: %s/", OUT_DIR)
        else:
            logger.warning("No results to save.")


# ── Export: CSV ───────────────────────────────────────────────────────────────
def export_csv(rows: list):
    path = OUT_DIR / f"results_{RUN_ID}.csv"
    fields = [
        "id", "risk", "h_score", "best_h_score", "faithfulness", "claim_coverage",
        "contradiction", "answer_relevance", "retries", "drift_rejected", "rouge_l",
        "latency_s", "accepted", "query", "answer", "reference",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info("CSV saved: %s", path)


# ── Export: Summary JSON ──────────────────────────────────────────────────────
def export_summary(rows: list):
    tiers   = sorted({r["risk"] for r in rows})
    summary = {"run_id": RUN_ID, "total": len(rows), "by_risk": {}}

    for tier in tiers:
        subset = [r for r in rows if r["risk"] == tier]
        if not subset:
            continue
        summary["by_risk"][tier] = {
            "count":                 len(subset),
            "avg_best_h_score":      avg([r["best_h_score"]     for r in subset]),
            "avg_h_score":           avg([r["h_score"]          for r in subset]),
            "avg_faithfulness":      avg([r["faithfulness"]     for r in subset]),
            "avg_claim_cov":         avg([r["claim_coverage"]   for r in subset]),
            "avg_contradiction":     avg([r["contradiction"]    for r in subset]),
            "avg_answer_relevance":  avg([r["answer_relevance"] for r in subset]),
            "avg_retries":           avg([r["retries"]          for r in subset]),
            "avg_rouge_l":           avg([r["rouge_l"]          for r in subset]),
            "acceptance_rate":       round(sum(r["accepted"] for r in subset) / len(subset), 4),
        }

    summary["overall"] = {
        "avg_best_h_score":     avg([r["best_h_score"]     for r in rows]),
        "avg_h_score":          avg([r["h_score"]          for r in rows]),
        "avg_answer_relevance": avg([r["answer_relevance"] for r in rows]),
        "avg_rouge_l":          avg([r["rouge_l"]          for r in rows]),
        "avg_retries":          avg([r["retries"]          for r in rows]),
        "acceptance_rate":      round(sum(r["accepted"] for r in rows) / len(rows), 4),
        "avg_latency_s":        avg([r["latency_s"]        for r in rows]),
    }

    path = OUT_DIR / f"summary_{RUN_ID}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved: %s", path)
    logger.info("%s", json.dumps(summary["overall"], indent=2))


# ── Export: LaTeX Table ───────────────────────────────────────────────────────
def export_latex(rows: list):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-query H\_score evaluation results. Stress types: irrelevant, conflicting, missing, noisy. HotpotQA samples labelled as `hotpotqa'.}",
        r"\label{tab:hscore_results}",
        r"\begin{tabular}{llccccccc}",
        r"\hline",
        r"\textbf{ID} & \textbf{Risk} & \textbf{Best H\_score} & \textbf{Faith.} "
        r"& \textbf{Cov.} & \textbf{Contr.} & \textbf{Ans.Rel.} & \textbf{ROUGE-L} & \textbf{Retries} \\",
        r"\hline",
    ]

    for row in rows:
        risk_abbr = escape_latex(row["risk"][0].upper())
        lines.append(
            f"{escape_latex(row['id'])} & {risk_abbr} & {row['best_h_score']} & {row['faithfulness']} "
            f"& {row['claim_coverage']} & {row['contradiction']} & {row['answer_relevance']} "
            f"& {row['rouge_l']} & {row['retries']} \\\\"
        )

    lines.append(r"\hline")
    for tier in ["low", "medium", "high", "irrelevant", "conflicting", "missing", "noisy", "hotpotqa"]:
        subset = [r for r in rows if r["risk"] == tier]
        if subset:
            lines.append(
                f"\\textbf{{Avg ({escape_latex(tier)})}} & & "
                f"{avg([r['best_h_score']     for r in subset])} & "
                f"{avg([r['faithfulness']     for r in subset])} & "
                f"{avg([r['claim_coverage']   for r in subset])} & "
                f"{avg([r['contradiction']    for r in subset])} & "
                f"{avg([r['answer_relevance'] for r in subset])} & "
                f"{avg([r['rouge_l']          for r in subset])} & "
                f"{avg([r['retries']          for r in subset])} \\\\"
            )

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = OUT_DIR / f"paper_table_{RUN_ID}.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("LaTeX table saved: %s", path)


# ── Export: Run config ────────────────────────────────────────────────────────
def export_config(mode: str, n_samples: int, no_refine: bool, seed: int):
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
        "mode":         mode,
        "n_samples":    n_samples,
        "no_refine":    no_refine,
        "seed":         seed,
    }
    path = OUT_DIR / f"config_{RUN_ID}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Config saved: %s", path)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = ArgumentParser(description="Batch H_score evaluation harness")
    parser.add_argument("mode", nargs="?", default="stress",
                        choices=["stress", "hotpotqa", "both"],
                        help="Evaluation dataset (default: stress)")
    parser.add_argument("--no-refine", action="store_true",
                        help="Single-pass ablation — disable query refinement loop")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of HotpotQA samples (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for HotpotQA sample selection (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-sentence NLI traces (DEBUG logging)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output (WARNING logging only)")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    run_evaluation(
        mode=args.mode,
        hotpotqa_n=args.n_samples,
        no_refine=args.no_refine,
        seed=args.seed,
    )
