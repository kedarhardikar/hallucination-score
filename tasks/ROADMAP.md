# Roadmap — Order of Work

**Read `CONTEXT.md` first.** This file lists the work blocks and their order.

Each task below has its own file. Do them sequentially. Do not start `task-02` until `task-01` is checked off, and so on. Some tasks produce outputs that later tasks depend on.

| Order | File                              | What it covers                                                    | Who does it       | Blocks                       |
|-------|-----------------------------------|-------------------------------------------------------------------|-------------------|------------------------------|
| 01    | `task-01-correctness-fixes.md`    | Small correctness fixes to existing metric / pipeline             | Claude Code       | none                         |
| 02    | `task-02-engineering-polish.md`   | Logging, config, reproducibility, Makefile                        | Claude Code       | task-01                      |
| 03    | `task-03-sanity-tests.md`         | Pytest tests for the metric on hand-crafted inputs                | Claude Code       | task-01                      |
| 04    | `task-04-generate-eval-outputs.md`| Run pipeline on 100 HotpotQA samples → CSV ready for labeling     | Claude Code       | task-01, task-02             |
| 05    | `task-05-labeling.md`             | Hand-label 50 outputs — you do this, not Claude Code              | **You (human)**   | task-04                      |
| 06    | `task-06-baseline-metrics.md`     | Add RAGAS + HHEM scores to the labeled CSV                        | Claude Code       | task-05                      |
| 07    | `task-07-correlation-analysis.md` | Spearman / Pearson table + bootstrap significance test            | Claude Code       | task-06                      |
| 08    | `task-08-scale-hotpotqa.md`       | Run HotpotQA at n=200 across 3 seeds, report mean ± std           | Claude Code       | task-01, task-02             |
| 09    | `task-09-add-fever.md`            | Add FEVER dataset + evaluation mode                               | Claude Code       | task-01                      |
| 10    | `task-10-ablation-upgrade.md`     | Replace ROUGE-L with Spearman in ablations + plotting             | Claude Code       | task-07                      |
| 11    | `task-11-threshold-sweep.md`      | Justify all 0.5 thresholds (coverage / contradiction / drift)     | Claude Code       | task-07                      |
| 12    | `task-12-calibration-plot.md`     | Calibration plot of H_score vs human labels                       | Claude Code       | task-07                      |
| 13    | `task-13-qualitative-examples.md` | Pick 6 examples (H_score vs RAGAS disagreements + drift case)     | Claude Code + you | task-07                      |

After task-13: you are ready to start writing the paper.

## Recommended cadence

- Open one fresh Claude Code session per task.
- Always paste `CONTEXT.md` into the session along with the task file.
- After each task, manually verify the "Done when" criteria, then check the box.
- Commit to git between tasks. One task = one commit (or one PR).

## What NOT to do

- Do not hand multiple task files to one Claude Code session at once. It will conflate them and partially complete several.
- Do not skip task-01 to get to "the interesting work" — the correctness fixes change numbers in all downstream tasks.
- Do not let Claude Code run task-05 (the labeling). The metric must be validated against **human** judgment, not an LLM's judgment of an LLM. If you label with an LLM, your paper has no ground truth.
