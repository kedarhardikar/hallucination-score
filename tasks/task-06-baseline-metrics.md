# Task 06 — Run Baseline Metrics (RAGAS + HHEM)

**Read `CONTEXT.md` first. Depends on task-05 (`labeling/labels.csv` must exist).**

## Goal

Add RAGAS faithfulness, RAGAS answer relevancy, and HHEM scores to the 50 labeled rows.

## Why

The paper claims H_score is better than RAGAS and HHEM. That claim needs numbers, not arguments. Task-07 will compute correlations against the human labels — for that to work, every labeled row needs RAGAS and HHEM scores attached.

## Steps

1. **Install RAGAS and HHEM.**
   - Add to `req.txt`:
     - `ragas>=0.2.0`
     - `sentence-transformers` (already there for embeddings — confirm)
   - HHEM is a HuggingFace model — `vectara/hallucination_evaluation_model`. Load via `transformers.AutoModelForSequenceClassification`. No extra package install needed.
   - Note that RAGAS requires an LLM backend. Configure it to use Groq via the LangChain Groq wrapper (not OpenAI). If this is too painful, use `gpt-4o-mini` with a small budget and document the cost in the run config.

2. **Write `baselines.py`.**
   - Takes two CLI args: input CSV (with `id, query, retrieved_docs_concat, answer`) and output CSV path.
   - For each row, compute:
     - `ragas_faithfulness`
     - `ragas_answer_relevancy`
     - `hhem_score` (the model returns 0–1; 1 = consistent)
   - Write a new CSV: input columns + the three new metric columns.
   - Handle failures gracefully — if RAGAS times out on one row, write `NaN` and continue.

3. **Run it.**
   ```
   python baselines.py labeling/labels.csv labeling/labels_with_baselines.csv
   ```

4. **Merge with H_score metrics.**
   - Write `scripts/merge_labels_and_metrics.py` that joins `labeling/labels_with_baselines.csv` with `labeling/metrics_<RUN_ID>.csv` on `id`.
   - Output: `labeling/labels_full.csv` with columns:
     - `id, query, answer, reference, label_binary, label_grade, notes`
     - `h_score, faithfulness, claim_coverage, contradiction, answer_relevance, rouge_l`
     - `ragas_faithfulness, ragas_answer_relevancy, hhem_score`

## Done when

- [ ] `labeling/labels_with_baselines.csv` exists with all 50 rows and three baseline columns populated.
- [ ] `labeling/labels_full.csv` exists with all 50 rows, all labels, all H_score components, and all three baseline scores.
- [ ] No more than 2 rows have NaN in any baseline column (if more fail, debug RAGAS / HHEM setup).
- [ ] You log the wall-clock time and API cost per run in the run config.

## Do not

- Do not change H_score values to match baselines. The whole point is to compare.
- Do not skip HHEM because it's "just one number." It's the fastest baseline and the most direct comparison for the faithfulness + contradiction part of H_score.
- Do not use a different LLM judge for RAGAS than what's documented. Whatever you use, log it.
