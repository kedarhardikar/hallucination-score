"""
evaluate.py
-----------
Batch evaluation harness for the LangGraph RAG + H_score pipeline.
Runs all queries, logs per-query metrics, and exports:
  - results.csv       → raw per-query results (for your paper's results table)
  - summary.json      → aggregated stats by hallucination_risk tier
  - paper_table.txt   → LaTeX-ready table for direct paste into your paper
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

from llama_index.core import VectorStoreIndex
from dataset import (
    get_stress_contexts_as_documents, get_stress_queries,
    load_hotpotqa, get_hotpotqa_documents, get_hotpotqa_queries,
)
from main import build_rag_graph, RAGState, Settings, THRESHOLD

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)
RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Shared utility ────────────────────────────────────────────────────────────
def avg(lst: list) -> float:
    return round(sum(lst) / len(lst), 4) if lst else 0.0


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
    initial: RAGState = {
        "query":          query,
        "original_query": query,
        "retrieved_docs": [],
        "answer":         "",
        "h_score":        0.0,
        "faithfulness":   0.0,
        "claim_coverage": 0.0,
        "contradiction":  0.0,
        "retries":        0,
        "best_answer":    None,
        "best_h_score":   -1.0,
        "final_answer":   None,
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
        "h_score":        result["h_score"],
        "best_h_score":   result["best_h_score"],
        "faithfulness":   result["faithfulness"],
        "claim_coverage": result["claim_coverage"],
        "contradiction":  result["contradiction"],
        "retries":        result["retries"],
        "rouge_l":        round(rl, 4),
        "latency_s":      latency,
        "accepted":       result["best_h_score"] >= THRESHOLD,
    }


# ── Batch runner ──────────────────────────────────────────────────────────────
def run_evaluation(mode: str = "stress", hotpotqa_n: int = 50):
    """
    mode = "stress"   → adversarial stress-test dataset
    mode = "hotpotqa" → HotpotQA distractor split (real multi-hop QA)
    mode = "both"     → stress + hotpotqa combined

    hotpotqa_n: number of HotpotQA samples to load (default 50)
    """
    print(f"\n{'='*60}")
    print(f"  RAG Hallucination Evaluation — Run {RUN_ID}  [{mode}]")
    print(f"{'='*60}\n")

    if mode == "stress":
        docs, queries = get_stress_contexts_as_documents(), get_stress_queries()
    elif mode == "hotpotqa":
        samples = load_hotpotqa(split="validation", n_samples=hotpotqa_n)
        docs    = get_hotpotqa_documents(samples)
        queries = get_hotpotqa_queries(samples)
        print(f"  Loaded {len(samples)} HotpotQA samples → {len(docs)} documents")
    else:  # both
        samples  = load_hotpotqa(split="validation", n_samples=hotpotqa_n)
        docs     = get_stress_contexts_as_documents() + get_hotpotqa_documents(samples)
        queries  = get_stress_queries() + get_hotpotqa_queries(samples)
        print(f"  Loaded {len(get_stress_queries())} stress + {len(samples)} HotpotQA queries")

    index = VectorStoreIndex.from_documents(docs)
    graph = build_rag_graph(index)
    rows  = []

    try:
        for i, (qid, query, reference, risk) in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {qid} | risk={risk} | {query[:60]}...")
            try:
                row = evaluate_query(graph, qid, query, reference, risk)
                rows.append(row)
                print(f"         H_score={row['h_score']} | ROUGE-L={row['rouge_l']} "
                      f"| retries={row['retries']} | latency={row['latency_s']}s\n")
            except Exception as e:
                print(f"  ⚠ Query {qid} failed: {e}\n")
    except KeyboardInterrupt:
        print("\n⚠ Interrupted — saving partial results...")
    finally:
        if rows:
            export_csv(rows)
            export_summary(rows)
            export_latex(rows)
            print(f"\n✅ Results saved to: {OUT_DIR}/")
        else:
            print("No results to save.")


# ── Export: CSV ───────────────────────────────────────────────────────────────
def export_csv(rows: list):
    path = OUT_DIR / f"results_{RUN_ID}.csv"
    fields = [
        "id", "risk", "h_score", "best_h_score", "faithfulness", "claim_coverage",
        "contradiction", "retries", "rouge_l", "latency_s", "accepted",
        "query", "answer", "reference",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"📄 CSV saved: {path}")


# ── Export: Summary JSON ──────────────────────────────────────────────────────
def export_summary(rows: list):
    tiers   = sorted({r["risk"] for r in rows})
    summary = {"run_id": RUN_ID, "total": len(rows), "by_risk": {}}

    for tier in tiers:
        subset = [r for r in rows if r["risk"] == tier]
        if not subset:
            continue
        summary["by_risk"][tier] = {
            "count":             len(subset),
            "avg_h_score":       avg([r["h_score"]       for r in subset]),
            "avg_faithfulness":  avg([r["faithfulness"]  for r in subset]),
            "avg_claim_cov":     avg([r["claim_coverage"] for r in subset]),
            "avg_contradiction": avg([r["contradiction"]  for r in subset]),
            "avg_retries":       avg([r["retries"]        for r in subset]),
            "avg_rouge_l":       avg([r["rouge_l"]        for r in subset]),
            "acceptance_rate":   round(sum(r["accepted"] for r in subset) / len(subset), 4),
        }

    summary["overall"] = {
        "avg_h_score":     avg([r["h_score"]    for r in rows]),
        "avg_rouge_l":     avg([r["rouge_l"]    for r in rows]),
        "avg_retries":     avg([r["retries"]    for r in rows]),
        "acceptance_rate": round(sum(r["accepted"] for r in rows) / len(rows), 4),
        "avg_latency_s":   avg([r["latency_s"]  for r in rows]),
    }

    path = OUT_DIR / f"summary_{RUN_ID}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Summary saved: {path}")
    print(json.dumps(summary["overall"], indent=2))


# ── Export: LaTeX Table ───────────────────────────────────────────────────────
def export_latex(rows: list):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-query H\_score evaluation results. Risk tiers: L=Low, M=Medium, H=High.}",
        r"\label{tab:hscore_results}",
        r"\begin{tabular}{llcccccc}",
        r"\hline",
        r"\textbf{ID} & \textbf{Risk} & \textbf{H\_score} & \textbf{Faith.} "
        r"& \textbf{Cov.} & \textbf{Contr.} & \textbf{ROUGE-L} & \textbf{Retries} \\",
        r"\hline",
    ]

    for row in rows:
        risk_abbr = escape_latex(row["risk"][0].upper())
        lines.append(
            f"{escape_latex(row['id'])} & {risk_abbr} & {row['h_score']} & {row['faithfulness']} "
            f"& {row['claim_coverage']} & {row['contradiction']} "
            f"& {row['rouge_l']} & {row['retries']} \\\\"
        )

    lines.append(r"\hline")
    for tier in ["low", "medium", "high", "irrelevant", "conflicting", "missing", "noisy", "hotpotqa"]:
        subset = [r for r in rows if r["risk"] == tier]
        if subset:
            lines.append(
                f"\\textbf{{Avg ({escape_latex(tier)})}} & & "
                f"{avg([r['h_score']       for r in subset])} & "
                f"{avg([r['faithfulness']  for r in subset])} & "
                f"{avg([r['claim_coverage'] for r in subset])} & "
                f"{avg([r['contradiction'] for r in subset])} & "
                f"{avg([r['rouge_l']       for r in subset])} & "
                f"{avg([r['retries']       for r in subset])} \\\\"
            )

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = OUT_DIR / f"paper_table_{RUN_ID}.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📝 LaTeX table saved: {path}")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "stress"
    run_evaluation(mode=mode)
