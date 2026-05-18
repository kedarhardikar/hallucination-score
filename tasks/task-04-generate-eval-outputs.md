# Task 04 — Generate Outputs for Labeling

**Read `CONTEXT.md` first. Depends on task-01 and task-02.**

## Goal

Run the pipeline on 100 HotpotQA samples and produce a CSV that's ready for the human to label.

## Why

Task 05 (human labeling) needs a clean CSV with one row per output. This task generates it. After labeling, tasks 06 and 07 add baseline metrics and compute correlations — the empirical core of the paper.

## Steps

1. **Run the pipeline.**
   ```
   python evaluate.py hotpotqa --n-samples 100 --seed 42
   ```
   (CLI flags from task-02. If `--n-samples` doesn't exist yet, add it now.)

2. **Generate a labeling-friendly CSV.**
   - Write `scripts/prepare_labeling_csv.py` that reads `eval_results/results_<RUN_ID>.csv` and produces `labeling/to_label.csv` with these columns in this order:
     - `id`
     - `query`
     - `retrieved_docs_concat` (the retrieved passages joined with `\n---\n`, max 5000 chars)
     - `answer`
     - `reference` (HotpotQA's gold answer)
     - `label_binary` (empty — for the human to fill: 0 or 1)
     - `label_grade` (empty — for the human to fill: 1–5)
     - `notes` (empty)
   - The script should take the RUN_ID as a CLI arg.
   - The script must NOT include H_score, RAGAS, or any metric columns. The human must label blind to avoid anchoring bias.

3. **Save a separate metrics file** keyed by `id`.
   - Write `labeling/metrics_<RUN_ID>.csv` containing only `id`, `h_score`, `best_h_score`, `faithfulness`, `claim_coverage`, `contradiction`, `answer_relevance`, `rouge_l`, `retries`, `drift_rejected`.
   - This is the file that task-07 will merge with the human labels.

4. **Sanity check the CSV.**
   - Verify there are exactly 100 rows.
   - Verify no row has an empty `query` or `answer`.
   - Verify `retrieved_docs_concat` has content for every row.

## Done when

- [ ] `labeling/to_label.csv` exists with 100 rows, the right columns, and empty label columns.
- [ ] `labeling/metrics_<RUN_ID>.csv` exists with all 100 rows and all metric columns populated.
- [ ] `labeling/to_label.csv` does NOT contain any metric columns (human must label blind).
- [ ] You print the path to both files at the end so the human can find them.

## Do not

- Do not put metric scores in the labeling CSV. Blind labeling is essential to avoid the human anchoring to H_score.
- Do not pre-fill any labels. The human does that in task-05.
- Do not change the seed from 42 without recording it in the file name.
