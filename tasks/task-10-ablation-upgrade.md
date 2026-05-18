# Task 10 — Upgrade Ablation Reporting

**Read `CONTEXT.md` first. Depends on task-07 (correlation table must exist).**

## Goal

Replace ROUGE-L with Spearman ρ vs human labels as the primary "is this component pulling its weight" signal in the ablation studies.

## Why

The whole point of the paper is that lexical overlap is not hallucination. Using ROUGE-L as the ablation target undermines this. Spearman ρ against human labels is the correct target.

## Steps

1. **Modify `ablation.py` to require a labels CSV.**
   - Add `--labels` CLI argument pointing to `labeling/labels_full.csv`.
   - Without it, ablation refuses to run (or falls back to ROUGE-L with a clear warning).

2. **For each ablation configuration:**
   - Run the pipeline with those weights.
   - Compute H_score for each labeled row.
   - Compute Spearman ρ between H_score and `label_grade`.
   - Report ρ alongside (not instead of) the existing ROUGE-L number.

3. **Update `ablation_components`.**
   - Per-config output now includes `spearman_rho`, `spearman_p`.
   - Identify the best-performing component-removal as evidence each component contributes.
   - If removing answer_relevance (δ=0) drops Spearman by < 0.02, the δ component is not earning its weight — re-discuss whether to keep it.

4. **Update `ablation_weights`.**
   - Per-combo output now includes `spearman_rho`.
   - Produce a heatmap `analysis/weight_heatmap_alpha_beta.png` showing Spearman ρ across the α × β grid (with γ fixed at three values: 0.0, 0.15, 0.30).
   - Identify and report the top-3 weight combinations by Spearman ρ.

5. **Update `ablation_refinement`.**
   - Add: fraction of queries where refinement was triggered, fraction where drift guard fired (uses task-01's `drift_rejected` column), fraction where refinement improved best_h_score vs left it unchanged vs made it worse.

## Done when

- [ ] `python ablation.py component --labels labeling/labels_full.csv` runs and reports Spearman ρ per config.
- [ ] `analysis/weight_heatmap_alpha_beta.png` exists.
- [ ] The component ablation clearly shows whether each of the 4 components contributes.
- [ ] If the default weights (0.30, 0.30, 0.15, 0.25) are not in the top-5 by Spearman ρ, you discuss updating them.

## Do not

- Do not delete ROUGE-L reporting entirely. Keep it as a secondary column — it's still informative.
- Do not run the full weight grid more than once per labeled set — it's expensive.
