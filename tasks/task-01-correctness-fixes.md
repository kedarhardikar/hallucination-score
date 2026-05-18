# Task 01 — Correctness Fixes

**Read `CONTEXT.md` first.**

## Goal

Fix small correctness issues in the existing pipeline without changing the metric's design.

## Why

These don't change the H_score formula but they make downstream numbers (and ablations) cleaner and more trustworthy. Do them first because every other task runs on top of this code.

## Steps

1. **Add `drift_rejected` boolean to the pipeline state and CSV output.**
   - In `main.py` `RAGState`, add `drift_rejected: bool` (default `False`).
   - In `refine_query_node`, when the drift guard fires, set `state["drift_rejected"] = True` *in addition to* setting `retries = MAX_RETRIES`.
   - In `evaluate.py` `evaluate_query`, include `drift_rejected` in the returned dict.
   - In `evaluate.py` `export_csv`, add `drift_rejected` to the `fields` list.

2. **Document `should_retry` first-pass behavior.**
   - In `main.py`, add a comment above `should_retry` explaining: "If best_h_score ≥ THRESHOLD on the first pass, the pipeline finalizes without refining. This is intentional — a confident first-pass answer should not be perturbed."

3. **Verify the faithfulness edge case.**
   - Write a one-off Python script `scripts/check_edge_case.py` that constructs an answer with 1 strongly-entailed sentence and 5 unsupported sentences against a fixed context.
   - Compute H_score on it.
   - Print the four components and the composite.
   - Confirm that the composite stays low (β · 0.17 + γ · ~1.0 + δ · ? — expect composite around 0.4–0.5, well below threshold).
   - If composite exceeds threshold, raise an issue — the weights need re-discussion.

4. **Move NLI device to a config flag.**
   - In `main.py`: `NLI_DEVICE = int(os.getenv("NLI_DEVICE", -1))`.
   - Pass `device=NLI_DEVICE` in `_get_nli`.
   - Document in README: "Set `NLI_DEVICE=0` for GPU."

5. **Add `functools.lru_cache` to `nli_score`.**
   - Cache size 2048 is fine.
   - This avoids redundant inference when the same (passage, sentence) appears twice in one H_score computation.
   - Note: pipeline objects are not hashable, but the wrapper function takes strings — should work directly.

## Done when

- [ ] `drift_rejected` appears as a column in `eval_results/results_*.csv` after running `python evaluate.py stress`.
- [ ] Running `python scripts/check_edge_case.py` prints the four components and a composite score, with the composite < 0.65.
- [ ] `NLI_DEVICE=0 python evaluate.py stress` runs on GPU (or fails with a clear CUDA error on a CPU-only machine — that's fine).
- [ ] Re-running stress eval produces results within ±0.01 of a previous run (the `lru_cache` change should not move numbers, just speed things up).

## Do not

- Do not change weights (α, β, γ, δ).
- Do not change thresholds (0.5 coverage cutoff, 0.5 drift cutoff, 0.65 acceptance threshold).
- Do not change the NLI or embedding models.
- Do not change the H_score formula or the per-sentence × per-passage scoring approach.
