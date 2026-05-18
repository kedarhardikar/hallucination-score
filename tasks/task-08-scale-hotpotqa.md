# Task 08 — Scale HotpotQA to n=200, multi-seed

**Read `CONTEXT.md` first. Depends on task-01 and task-02. Can run in parallel with tasks 03–07.**

## Goal

Run HotpotQA evaluation at `n=200` across 3 random seeds and report mean ± std.

## Why

The current default of `n=50` is too small for a credible main result. Single-run results on 200 samples can still vary by 5–10% — reporting mean ± std shows stability.

## Steps

1. **Confirm `evaluate.py` accepts `--n-samples` and `--seed`** (added in task-02). If not, add them now.

2. **Run the pipeline three times.**
   ```
   python evaluate.py hotpotqa --n-samples 200 --seed 1
   python evaluate.py hotpotqa --n-samples 200 --seed 2
   python evaluate.py hotpotqa --n-samples 200 --seed 3
   ```
   Each run will hit the cache for the embedding model but rebuild the ChromaDB collection if the sample set changes. Force fresh collections per seed (e.g. collection name = `hotpotqa_seed1`, etc.) to keep them isolated.

3. **Write `scripts/aggregate_seeds.py`.**
   - Takes 3 run IDs as CLI args.
   - Reads all three `summary_*.json` files.
   - Outputs `analysis/hotpot_200_aggregated.json` with mean and std for every metric in the `overall` block and every metric in `by_risk["hotpotqa"]`.

4. **Add std columns to the LaTeX table generator.**
   - Modify `evaluate.py` `export_latex` to optionally accept a stds dict.
   - When stds are provided, write each cell as `mean ± std`.

5. **Run a single aggregated table.**
   - `python scripts/aggregate_seeds.py <run_id_1> <run_id_2> <run_id_3>` should produce `analysis/hotpot_200_results.tex`.

## Done when

- [ ] Three `results_*.csv` files in `eval_results/` from the three seeds, each with 200 rows.
- [ ] `analysis/hotpot_200_aggregated.json` exists with mean and std for every metric.
- [ ] `analysis/hotpot_200_results.tex` exists and shows `mean ± std` in cells.
- [ ] Variance across seeds is documented (e.g. "avg_best_h_score = 0.71 ± 0.02").

## Do not

- Do not run more than 3 seeds. Diminishing returns and lots of compute.
- Do not skip seed isolation in ChromaDB collection names — otherwise seeds reuse each other's embeddings and the multi-seed result is meaningless.
- Do not change the metric weights between seeds.
