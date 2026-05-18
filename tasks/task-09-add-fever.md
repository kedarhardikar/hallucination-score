# Task 09 — Add FEVER Dataset

**Read `CONTEXT.md` first. Depends on task-01. Can run in parallel with tasks 04–07.**

## Goal

Add FEVER as a third evaluation dataset and report how H_score components behave across the three FEVER labels (SUPPORTS / REFUTES / NOT ENOUGH INFO).

## Why

FEVER is the perfect testbed for the contradiction component — every claim has a ground-truth label. The original research plan called for FEVER as the second dataset alongside HotpotQA.

For the paper, FEVER's REFUTES label gives the first direct empirical validation that the contradiction component actually fires when it should.

## Steps

1. **Add `load_fever`** to `dataset.py`.
   - Use `load_dataset("fever", "v1.0")` from HuggingFace.
   - Each FEVER sample has: `id, claim, label (SUPPORTS/REFUTES/NOT ENOUGH INFO), evidence`.
   - For RAG-style evaluation, treat the `claim` as both the query and the answer to score, and the `evidence` (Wikipedia passages) as the context.
   - For NEI samples, evidence is empty — handle this case (use top-k retrieval from the full FEVER corpus, or skip NEI for the first pass).

2. **Add `get_fever_documents(samples)`** that converts evidence into LlamaIndex Documents.

3. **Add `get_fever_queries(samples)`** that returns `(id, claim, expected_label, "fever")` tuples.

4. **Add a `fever` mode to `evaluate.py`.**
   - Add to the if/elif chain.
   - Collection name: `fever`.
   - Cache queries the same way as hotpotqa.

5. **Add FEVER-specific analysis.**
   - Write `scripts/fever_analysis.py` that takes a `results_*.csv` from a FEVER run and produces:
     - Per-label aggregated stats (mean h_score, faithfulness, coverage, contradiction, answer_relevance).
     - A bar chart `analysis/fever_components_by_label.png` showing faithfulness, coverage, and contradiction for each label.
   - Expected pattern:
     - SUPPORTS → high faithfulness, high coverage, low contradiction
     - REFUTES → low faithfulness, low coverage, **high contradiction**
     - NEI → low faithfulness, low coverage, low contradiction

6. **Run a sanity-size evaluation first.**
   ```
   python evaluate.py fever --n-samples 50
   python scripts/fever_analysis.py eval_results/results_<RUN_ID>.csv
   ```
   Verify the expected pattern shows up before scaling.

7. **Once the pattern looks right, scale up.**
   - `n=200` is fine. Multi-seed not required for FEVER (the labels are deterministic).

## Done when

- [ ] `python evaluate.py fever --n-samples 50` runs to completion.
- [ ] `analysis/fever_components_by_label.png` exists and shows the expected pattern (contradiction high on REFUTES).
- [ ] A final FEVER run with `n=200` produces a results CSV.
- [ ] If the expected pattern does NOT appear, investigate and document — this is informative either way.

## Do not

- Do not score FEVER claims using the H_score retry loop unless you also report results without retries — FEVER is about fact verification, not answer generation, so the refinement loop is conceptually misaligned. Add a note in the paper about this difference.
- Do not skip the NEI label handling. NEI is where contradiction-based metrics often fail.
