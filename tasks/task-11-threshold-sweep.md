# Task 11 — Justify the 0.5 Thresholds

**Read `CONTEXT.md` first. Depends on task-07.**

## Goal

Empirically justify the three magic-number thresholds in the code: coverage cutoff (0.5), contradiction cutoff (0.5), and drift cutoff (0.5).

## Why

Right now they are stated without justification. Any reviewer will ask "why 0.5?" — the answer needs to be "because the sweep showed 0.5 maximizes correlation with human labels."

## Steps

1. **Write `scripts/threshold_sweep.py`.**
   - Takes `labeling/labels_full.csv` as input.
   - For each threshold in `[0.3, 0.4, 0.5, 0.6, 0.7]`:
     - Recompute H_score on each labeled row using that threshold (you'll need to re-run the metric with different thresholds — modify `compute_h_score` to take thresholds as arguments rather than hardcoding 0.5).
     - Compute Spearman ρ vs `label_grade`.
   - Repeat for the coverage threshold, the contradiction threshold, and the drift cutoff.
   - Output: `analysis/threshold_sweep.csv` and `analysis/threshold_sweep.tex`.

2. **Modify `main.py` to accept thresholds as parameters.**
   - Add module-level constants `COVERAGE_THRESHOLD = 0.5`, `CONTRADICTION_THRESHOLD = 0.5` (DRIFT_CUTOFF already exists).
   - `compute_h_score` reads from these constants.
   - This makes the sweep cheap to run — just override the constants.

3. **Plot the results.**
   - `analysis/threshold_sweep.png` — one panel per threshold (coverage, contradiction, drift), x-axis = threshold value, y-axis = Spearman ρ.
   - Mark the chosen value with a vertical line.

4. **Pick the best thresholds.**
   - If the optimal differs from 0.5 by more than 0.05 Spearman ρ, update the defaults.
   - If 0.5 is within 0.02 ρ of the optimum, keep 0.5 and report this in the paper as "robust to threshold choice in this range."

## Done when

- [ ] `analysis/threshold_sweep.csv` exists with rows for every threshold × parameter combination.
- [ ] `analysis/threshold_sweep.png` exists.
- [ ] The chosen thresholds are documented in `main.py` with a comment citing the sweep results.

## Do not

- Do not change the thresholds without re-running the correlation analysis (task-07).
- Do not sweep more granularly than 0.1 — gives diminishing returns and obscures the result.
