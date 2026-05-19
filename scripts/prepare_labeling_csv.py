"""
scripts/prepare_labeling_csv.py
--------------------------------
Splits a results CSV into two files ready for Task 05 (human labeling):

  labeling/to_label_<RUN_ID>.csv   — what the human sees and fills in (NO metric scores)
  labeling/metrics_<RUN_ID>.csv    — metric scores keyed by id (merged back in Task 07)

Usage:
  python scripts/prepare_labeling_csv.py <RUN_ID>

Example:
  python scripts/prepare_labeling_csv.py 20240519_143022

The RUN_ID matches the timestamp in eval_results/results_<RUN_ID>.csv.
"""

import csv
import sys
from pathlib import Path

EVAL_DIR    = Path("eval_results")
LABEL_DIR   = Path("labeling")
LABEL_DIR.mkdir(exist_ok=True)

MAX_DOCS_CHARS = 5000   # truncate retrieved_docs_concat to keep CSV manageable


def main(run_id: str):
    src = EVAL_DIR / f"results_{run_id}.csv"
    if not src.exists():
        print(f"ERROR: {src} not found.")
        print(f"Available result files:")
        for f in sorted(EVAL_DIR.glob("results_*.csv")):
            print(f"  {f.name}")
        sys.exit(1)

    with open(src, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("ERROR: results CSV is empty.")
        sys.exit(1)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print(f"\nSanity checks on {src.name}:")
    print(f"  Rows           : {len(rows)}")

    empty_query  = [r["id"] for r in rows if not r.get("query", "").strip()]
    empty_answer = [r["id"] for r in rows if not r.get("answer", "").strip()]
    empty_docs   = [r["id"] for r in rows if not r.get("retrieved_docs_concat", "").strip()]

    print(f"  Empty query    : {len(empty_query)}  {empty_query[:5] if empty_query else ''}")
    print(f"  Empty answer   : {len(empty_answer)}  {empty_answer[:5] if empty_answer else ''}")
    print(f"  Empty docs     : {len(empty_docs)}  {empty_docs[:5] if empty_docs else ''}")

    if empty_query or empty_answer or empty_docs:
        print("\n⚠  WARNING: some rows have missing data — review before labeling.")
    else:
        print("  ✅ All rows have query, answer, and retrieved docs.")

    # ── to_label CSV (human sees this — NO metric columns) ────────────────────
    label_path = LABEL_DIR / f"to_label_{run_id}.csv"
    label_fields = [
        "id", "query", "retrieved_docs_concat", "answer", "reference",
        "label_binary", "label_grade", "notes",
    ]

    with open(label_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=label_fields)
        w.writeheader()
        for r in rows:
            docs = r.get("retrieved_docs_concat", "")
            if len(docs) > MAX_DOCS_CHARS:
                docs = docs[:MAX_DOCS_CHARS] + "\n[truncated]"
            w.writerow({
                "id":                   r["id"],
                "query":                r["query"],
                "retrieved_docs_concat": docs,
                "answer":               r["answer"],
                "reference":            r["reference"],
                "label_binary":         "",   # human fills: 0 = hallucinated, 1 = grounded
                "label_grade":          "",   # human fills: 1 (bad) – 5 (good)
                "notes":                "",
            })

    # ── metrics CSV (for Task 07 correlation analysis) ────────────────────────
    metrics_path = LABEL_DIR / f"metrics_{run_id}.csv"
    metrics_fields = [
        "id", "h_score", "best_h_score", "faithfulness", "claim_coverage",
        "contradiction", "answer_relevance", "rouge_l", "retries", "drift_rejected",
    ]

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metrics_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in metrics_fields})

    print(f"\n✅ Files written:")
    print(f"   Labeling CSV : {label_path}  ({len(rows)} rows, {len(label_fields)} columns)")
    print(f"   Metrics CSV  : {metrics_path}  ({len(rows)} rows, {len(metrics_fields)} columns)")
    print(f"\nNext step → open {label_path} and fill in label_binary and label_grade for each row.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Auto-detect the most recent results file
        files = sorted(Path("eval_results").glob("results_*.csv"))
        if not files:
            print("ERROR: no results_*.csv found in eval_results/. Run evaluate.py first.")
            sys.exit(1)
        run_id = files[-1].stem.replace("results_", "")
        print(f"No RUN_ID given — using most recent: {run_id}")
    else:
        run_id = sys.argv[1]

    main(run_id)
