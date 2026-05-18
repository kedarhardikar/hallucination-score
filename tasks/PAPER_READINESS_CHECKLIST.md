# H_score Project — Paper Readiness Checklist

A complete inventory of the current state of the project and every change required before submission. Tick items as they are completed.

---

## 0. How to read this document

- **Part A** — what the code does well (no action needed, listed so nothing gets accidentally regressed).
- **Part B** — correctness issues and small fixes in the existing code.
- **Part C** — new work required before the paper can be written, in the order it should be done.
- **Part D** — paper-writing checklist (after Part C is complete).

Each item has a check box `[ ]`. When done, change to `[x]`.

---

## Part A — What the code already does well (do not regress)

These are working as intended. Listed here so Claude Code knows not to touch them when making other changes.

- [x] LangGraph pipeline: retrieve → generate → score → conditional refine → finalize
- [x] H_score formula combines four components: faithfulness, claim coverage, contradiction, answer relevance
- [x] Per-sentence × per-passage NLI scoring with max-entailment pooling (signal is no longer diluted by long concatenated context)
- [x] Contradiction is measured from the **best-matching** passage, not the worst across all passages
- [x] Faithfulness is computed over covered sentences only (grounding strength, not diluted by unsupported sentences)
- [x] Claim coverage is computed over all sentences (grounding breadth)
- [x] AnswerRelevance uses cosine similarity between original query and answer (deterministic, no LLM-as-judge)
- [x] AnswerRelevance always uses `original_query`, never the refined query — drift is always penalized
- [x] Query refinement prompt explicitly instructs the LLM not to change the topic
- [x] Drift guard: refined query is rejected if cosine similarity with original query falls below `DRIFT_CUTOFF = 0.5`
- [x] `should_retry` checks `best_h_score`, not the latest `h_score` — a worse retry cannot trigger acceptance
- [x] NLTK sentence tokenization (handles abbreviations correctly)
- [x] ChromaDB persistence layer with per-collection isolation (`stress`, `hotpotqa`, `both`)
- [x] Source-isolated retrieval for stress queries via `filter_source_id` metadata filter
- [x] HotpotQA query cache to avoid repeated downloads
- [x] Batch evaluation harness exports CSV, JSON summary, and LaTeX table
- [x] Ablation harness covers component ablation, weight grid search, and refinement on/off
- [x] `--no-refine` flag for single-pass ablation runs
- [x] `RESEARCH.md` correctly positions the work against RAGAS / HHEM / TruLens / NLI-only baselines

---

## Part B — Existing-code fixes (do these before any new work)

Small but real issues in the current code. None of them require new dependencies.

### B1. Correctness in the H_score components

- [ ] **Verify faithfulness edge case behavior.** When only 1 of 6 sentences is covered, faithfulness reflects only that one sentence and can be very high (e.g. 0.95) while coverage is very low (0.17). Confirm with a unit test that the **composite H_score** still reflects this as a low-quality answer (it should, because β=0.30 × low coverage drags it down — but verify).
- [ ] **Add a `drift_rejected` boolean column** to `evaluate_query`'s output dict and to the CSV header. Currently, when the drift guard fires it sets `retries = MAX_RETRIES`, which makes the `retries` column ambiguous ("did we actually try 3 times or were we cut short?"). Track them separately.
- [ ] **Decide and document the `best_h_score ≥ THRESHOLD` on first pass behavior.** Right now, if pass #1 scores ≥ 0.65, the pipeline never refines. This is probably correct, but write a comment in `should_retry` explaining the choice.
- [ ] **Check the AnswerRelevance distribution.** Run the existing stress + hotpotqa eval and look at the `answer_relevance` column in the CSV. Compute min, max, mean, std. If every value is in 0.6–0.8 regardless of quality, the δ component is doing less work than expected — BGE embeddings produce similar vectors for question + answer even when the answer is wrong. If this happens, consider switching to a question-answer-trained model like `BAAI/bge-reranker-base` or report it as a limitation.
- [ ] **Sanity-test the metric on three crafted answers** before any other work:
  - Answer A: directly copied from one retrieved passage (should score near 1.0 on faithfulness, near 1.0 on coverage, 0.0 contradiction).
  - Answer B: completely off-topic ("the sky is blue") (should score near 0 on faithfulness and coverage).
  - Answer C: answers a related but different question (should score moderate faithfulness if any sentence happens to be entailed, but low answer_relevance).
  - Add these as `tests/test_metric_sanity.py` with `pytest` assertions.

### B2. Engineering polish

- [ ] **Move `device=0` to a config flag** in `main.py`. Currently NLI runs on CPU (slow on 200+ queries). Add `NLI_DEVICE = int(os.getenv("NLI_DEVICE", -1))` and pass through to `_get_nli`. Document in README.
- [ ] **Cache NLI scores within a single H_score computation.** If the same `(passage, sentence)` pair is scored twice (rare but possible with deduped retrieval), avoid the second call. Use `functools.lru_cache` on `nli_score`.
- [ ] **Replace `print(...)` with `logging`** throughout `main.py`, `evaluate.py`, `ablation.py`. Use `logging.INFO` for progress, `logging.DEBUG` for per-sentence NLI traces. Lets you silence verbose output during large runs.
- [ ] **Add a `--verbose` / `--quiet` flag** to `evaluate.py` and `ablation.py` that maps to log level.
- [ ] **Save the run config to the output directory.** Each run should emit `config_<RUN_ID>.json` containing: weights (α, β, γ, δ), threshold, max_retries, drift_cutoff, NLI model, embedding model, LLM model, dataset, n_samples, no_refine flag, git commit hash. Reproducibility for the paper.
- [ ] **Pin all dependencies in `req.txt`** to exact versions (use `pip freeze` output), not `>=`. Reviewers will want to reproduce.
- [ ] **Add a `Makefile` or `run.sh`** with the canonical commands (`make eval-stress`, `make eval-hotpot`, `make ablation`, `make all`). Makes the "how to reproduce" section of the paper one paragraph.

### B3. Reproducibility

- [ ] **Set a random seed** in `evaluate.py` for HotpotQA sample selection. Currently `ds.select(range(n_samples))` is deterministic but if `n_samples` changes the set changes — add explicit seeding so the *same N samples* are selected every time for a given seed.
- [ ] **Log the exact HotpotQA sample IDs used** in the run config. So a reviewer can rerun on the same subset.
- [ ] **Save the model versions** (NLI model SHA from HuggingFace, embedding model SHA, Groq model string) in the run config.

---

## Part C — New work required before the paper can be written

Ordered. Do them in this sequence — each one feeds into the next.

### C1. Human-label a small ground-truth set (highest priority)

This is the single biggest gap in the project. Without this, the central claim of the paper has no evidence.

- [ ] **Run the existing pipeline on HotpotQA with `n_samples=100`** to get 100 outputs. Save the CSV.
- [ ] **Write a labeling rubric.** A markdown file `labeling/RUBRIC.md` defining:
  - Binary label: `hallucinated` (any unsupported claim) vs `grounded` (all claims supported by retrieved passages).
  - Optional 1–5 grounding score: 1 = entirely fabricated, 5 = every claim verifiable.
  - Edge cases: partial support, claims true in general knowledge but absent from context (= hallucinated), refusals ("I don't have enough info") count as grounded if context truly lacks the answer.
- [ ] **Hand-label 50 randomly sampled outputs** from the 100. Save as `labeling/labels.csv` with columns `id, label_binary, label_grade, notes`.
- [ ] **(Stretch) Get a second annotator** for 20 of the 50 and compute Cohen's κ. Even a friend doing 20 labels gives a credible inter-annotator agreement number — without it, reviewers ask "is your labeling consistent?"

### C2. Run baseline metrics on the same outputs

You compare H_score to RAGAS / HHEM in `RESEARCH.md` only conceptually. The comparison must be empirical.

- [ ] **Install RAGAS** (`pip install ragas`). Add to `req.txt`.
- [ ] **Install HHEM** (`vectara/hallucination_evaluation_model` from HuggingFace). Add to `req.txt`.
- [ ] **Write `baselines.py`** that takes a results CSV (query, answer, retrieved_docs) and adds three columns:
  - `ragas_faithfulness`
  - `ragas_answer_relevancy`
  - `hhem_score`
- [ ] **Run `baselines.py` on the labeled 50 from C1.** Save as `labeling/labels_with_baselines.csv`.

### C3. Compute the correlation table

This is the empirical core of the paper.

- [ ] **Write `correlation.py`** that takes `labels_with_baselines.csv` and outputs a table:
  | Metric              | Spearman ρ with binary | Spearman ρ with grade | Pearson r with grade |
  |---------------------|------------------------|-----------------------|----------------------|
  | H_score (full)      |                        |                       |                      |
  | H_score (faith only)|                        |                       |                      |
  | RAGAS faithfulness  |                        |                       |                      |
  | RAGAS answer_rel    |                        |                       |                      |
  | HHEM                |                        |                       |                      |
  | ROUGE-L             |                        |                       |                      |
- [ ] **Also export this as a LaTeX table** ready to paste into the paper.
- [ ] **Run a paired bootstrap test** (1000 resamples) comparing H_score's correlation against each baseline's. Report the p-value. Lets you claim "H_score correlates significantly better than RAGAS faithfulness, p < 0.05" rather than just "slightly higher number."

### C4. Scale up HotpotQA

50 samples is too small for the main result.

- [ ] **Run `python evaluate.py hotpotqa` with `n_samples=200`.** Pipeline already supports it — change the default in `evaluate.py` or pass via CLI.
- [ ] **Report mean ± std across 3 runs** with different random seeds. Single-run results on 200 samples can still vary by 5–10% in metrics — show stability.
- [ ] **Update the LaTeX table generator** to optionally include std columns.

### C5. Add FEVER dataset

Your original plan called for FEVER. Adding it gives a second independent benchmark and a direct test for the contradiction component.

- [ ] **Add `load_fever()`** to `dataset.py` using `load_dataset("fever", "v1.0")` or the more recent `fever-aug`.
- [ ] **FEVER samples have `SUPPORTS / REFUTES / NOT ENOUGH INFO` labels.** Map these to expected H_score signals:
  - SUPPORTS → expect high faithfulness, high coverage, low contradiction
  - REFUTES → expect low faithfulness, high contradiction
  - NEI → expect low faithfulness, low coverage, low contradiction
- [ ] **Add a `fever` mode** to `evaluate.py`.
- [ ] **Report H_score behavior across the three FEVER labels.** A bar chart of faithfulness / coverage / contradiction by label. This directly validates the contradiction component, which currently has no empirical justification.

### C6. Strengthen the ablation results

The ablation harness exists but the metrics it reports are weak.

- [ ] **Replace `avg_rouge_l` with `Spearman ρ vs human labels`** as the primary "is this component pulling its weight" signal in `ablation_components`. ROUGE-L is a poor target — the whole point of the paper is that lexical overlap is not hallucination.
- [ ] **Add the by-stress-type breakdown to the weight sensitivity grid.** Currently `ablation_weights` reports only overall numbers. Adding per-stress-type lets you find weights that help on `noisy` but not `irrelevant` — a more interesting finding than a single global optimum.
- [ ] **Plot the weight sensitivity results.** A heatmap of α vs β with γ = 0.15 fixed (one panel), and the same for other γ values. Currently the grid output is a JSON dump that's hard to interpret.
- [ ] **For the refinement ablation, also report**:
  - Fraction of queries where refinement was triggered.
  - Fraction where the drift guard fired (after B1 adds the `drift_rejected` column).
  - Fraction where refinement improved best_h_score vs left it unchanged vs made it worse.

### C7. Qualitative analysis

Reviewers always ask for examples.

- [ ] **Hand-pick 6 qualitative examples** from the labeled HotpotQA outputs:
  - 2 where H_score correctly flagged hallucination that RAGAS missed.
  - 2 where RAGAS flagged hallucination that H_score correctly accepted.
  - 1 clear failure case for H_score (where the metric got it wrong).
  - 1 drift-guard-fired case (where refinement was correctly rejected).
- [ ] **Write a 1-paragraph commentary** for each, ready to drop into a "Qualitative Analysis" subsection of the paper.

### C8. Tighten the metric design

Two things in the current design that reviewers will pick at.

- [ ] **Justify the 0.5 thresholds** (for coverage, contradiction, drift cutoff). Currently they're stated without analysis. Run a sweep over `coverage_threshold ∈ {0.3, 0.4, 0.5, 0.6, 0.7}` and report Spearman ρ with human labels for each. Pick the threshold that maximizes correlation. Same for contradiction threshold and drift cutoff. Add results to the ablation document.
- [ ] **Add a calibration plot.** Bin outputs by H_score (e.g. into 10 bins of 0.1 width). For each bin, plot the fraction of human-labeled-as-grounded outputs. A well-calibrated metric produces a monotonically increasing curve. This is one figure that demonstrates the metric is more than a coarse ranker.

---

## Part D — Paper writing checklist (only start after Part C is done)

When Part C is complete, you have everything needed to write. Below is the checklist for the paper itself.

### D1. Section-by-section content

- [ ] **Abstract** — problem (RAG hallucination), gap (composite metrics + drift safety missing), method (H_score), key result (Spearman ρ with human labels vs RAGAS / HHEM).
- [ ] **Introduction** — RAG hallucination problem, two failure modes (grounding + drift), why existing metrics insufficient, contributions (4 components + drift guard + corrective loop), paper outline.
- [ ] **Related work** — RAGAS, ARES, TruLens, HHEM, NLI-based metrics, Self-RAG, CRAG, ReDeEP. Use the comparison table from `RESEARCH.md`.
- [ ] **Method** — H_score formula with explanation of each component. Why best-matching passage for contradiction. Why original query for answer relevance. Drift guard. Corrective refinement loop.
- [ ] **Experimental setup** — datasets (stress, HotpotQA 200, FEVER), models (NLI, embeddings, LLM), baselines (RAGAS, HHEM, ROUGE-L), evaluation protocol (50 human labels with Cohen's κ if available).
- [ ] **Results** — correlation table from C3, FEVER-by-label breakdown from C5, ablation results from C6, calibration plot from C8.
- [ ] **Qualitative analysis** — the 6 examples from C7.
- [ ] **Discussion / limitations** — small label set, NLI model limits on numerical/multi-hop reasoning, ROUGE-L as proxy, source-isolated stress retrieval as a controlled but artificial setup, weight selection still empirical.
- [ ] **Conclusion + future work** — extending to multi-modal RAG, learning weights, integrating with retriever fine-tuning.

### D2. Submission readiness

- [ ] **All numbers in the paper come from the same `RUN_ID`** that is included in the appendix / supplementary.
- [ ] **Supplementary materials**: results CSVs, labels CSV, run config, plotting scripts.
- [ ] **Code release**: clean the repo (remove debug prints, dead code), add a top-level `README.md` with paper citation, single-command reproduction (`make reproduce-paper`).
- [ ] **Pick a venue.** Workshop track of EMNLP / NAACL / ACL for first submission is realistic given dataset size. NeurIPS / ICLR datasets-and-benchmarks track if you scale up to 1000+ HotpotQA samples and add at least one more dataset.

---

## Summary — minimum work between now and a submittable paper

Critical path, in order:

1. **Part B (existing-code fixes)** — ~1 day.
2. **C1 (label 50 outputs)** — ~half a day.
3. **C2 (run RAGAS + HHEM)** — ~half a day.
4. **C3 (correlation table)** — ~2 hours.
5. **C4 (scale HotpotQA to 200, 3 seeds)** — ~few hours of compute, mostly unattended.
6. **C5 (add FEVER)** — ~half a day.
7. **C6 (better ablation reporting)** — ~half a day.
8. **C7 (qualitative examples)** — ~2 hours.
9. **C8 (threshold sweep + calibration plot)** — ~half a day.

Total: **~4–5 working days** of focused work + compute time before writing begins.

Without this work, the paper has no empirical defense for its main claim. With this work, you have a workshop-strong (and short-paper-strong) submission.
