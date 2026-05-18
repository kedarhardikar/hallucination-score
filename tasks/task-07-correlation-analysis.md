# Task 07 — Correlation Analysis

**Read `CONTEXT.md` first. Depends on task-06 (`labeling/labels_full.csv` must exist).**

## Goal

Produce the central empirical result of the paper: a table showing H_score correlates better with human judgment than RAGAS, HHEM, and ROUGE-L do.

## Why

This is the main result. Without it, the paper has no defense.

## Steps

1. **Write `correlation.py`.**
   - Takes `labeling/labels_full.csv` as input.
   - Computes for each of these metrics: H_score, faithfulness (alone), claim_coverage (alone), answer_relevance (alone), RAGAS faithfulness, RAGAS answer relevancy, HHEM, ROUGE-L:
     - Spearman ρ vs `label_binary`
     - Spearman ρ vs `label_grade`
     - Pearson r vs `label_grade`
     - 95% confidence interval for each (1000-resample bootstrap)
   - Outputs:
     - `analysis/correlation_table.csv` — machine-readable
     - `analysis/correlation_table.tex` — LaTeX table ready for paper
     - `analysis/correlation_summary.json` — for archival

2. **Add paired bootstrap test.**
   - For each baseline (RAGAS faithfulness, HHEM, ROUGE-L), run a paired bootstrap test comparing H_score's Spearman ρ against the baseline's.
   - Report p-value.
   - Add a `p_vs_h_score` column to the correlation table.

3. **Inter-annotator agreement (if `labels_annotator2.csv` exists).**
   - Compute Cohen's κ on the 20 overlapping rows for `label_binary`.
   - Compute weighted κ for `label_grade`.
   - Save to `analysis/agreement.json`.
   - If only one annotator's labels exist, skip this step and note it in the output.

4. **Print a human-readable summary at the end** showing:
   - Top 3 metrics by Spearman ρ vs binary labels.
   - Whether H_score significantly beats each baseline.
   - Cohen's κ if applicable.

## Done when

- [ ] `analysis/correlation_table.tex` exists and renders as a valid LaTeX table.
- [ ] `analysis/correlation_table.csv` shows H_score row + ablated H_score rows + baseline rows.
- [ ] `p_vs_h_score` column exists and has plausible values (0 to 1).
- [ ] The printed summary is clear enough that you immediately know whether H_score wins.

## Do not

- Do not cherry-pick a subset of rows to make H_score look better. Use all labeled rows.
- Do not report only Pearson without Spearman. Spearman is more robust for this use case.
- Do not skip the bootstrap. A single point estimate is not enough — reviewers want CIs and p-values.
- If H_score does NOT beat the baselines, do not panic. Report the result honestly. It may mean the metric needs design changes — that's a finding too. Loop back to task-01 or open a discussion.
