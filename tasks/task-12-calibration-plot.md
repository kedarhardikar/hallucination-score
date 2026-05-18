# Task 12 — Calibration Plot

**Read `CONTEXT.md` first. Depends on task-07.**

## Goal

Produce a calibration plot showing that higher H_score really does mean higher likelihood of being grounded.

## Why

Correlation tells you the ranking is right. Calibration tells you the *number* is meaningful — that an H_score of 0.7 means "70% chance of being grounded" and not just "higher than 0.6."

This is one figure that demonstrates the metric is more than a coarse ranker.

## Steps

1. **Write `scripts/calibration.py`.**
   - Takes `labeling/labels_full.csv` as input.
   - Bins H_score values into 10 bins of width 0.1 (0.0–0.1, 0.1–0.2, …, 0.9–1.0).
   - For each bin, computes the fraction of rows where `label_binary == 1`.
   - Also computes the bin count (so the reader can see which bins are sparse).

2. **Plot.**
   - `analysis/calibration.png` — x-axis = H_score bin midpoint, y-axis = fraction grounded.
   - Diagonal reference line `y = x` for perfect calibration.
   - Bar width proportional to bin count, or annotate each bin with `n=X`.

3. **Compute ECE (Expected Calibration Error).**
   - Standard formula: `ECE = sum(|p_bin - acc_bin| * n_bin / N)`.
   - Report alongside the plot.

4. **Repeat the same plot for baseline metrics** (RAGAS faithfulness, HHEM).
   - `analysis/calibration_comparison.png` with all three metrics overlaid.
   - Lets the paper claim "H_score is better calibrated than RAGAS."

## Done when

- [ ] `analysis/calibration.png` exists.
- [ ] `analysis/calibration_comparison.png` exists.
- [ ] ECE for H_score is reported.
- [ ] The plot's bins are clearly labeled with their counts.

## Do not

- Do not use more than 10 bins — with only 50 labeled rows, finer bins are mostly noise.
- Do not omit bins with zero rows — leave them empty so the reader sees the coverage.
