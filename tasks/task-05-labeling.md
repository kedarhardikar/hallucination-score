# Task 05 — Human Labeling (YOU do this, not Claude Code)

**Read `CONTEXT.md` first. Depends on task-04.**

**This task is for the human researcher. Do not ask Claude Code to do it. Do not use an LLM to label.**

## Goal

Hand-label 50 of the 100 outputs from task-04 as grounded or hallucinated.

## Why

This is the ground truth that task-07 will correlate against. Without it, the paper has no empirical defense for the metric.

If an LLM labels the outputs, the paper has zero ground truth — you'd be comparing one LLM's judgment against another LLM's judgment. Reviewers will reject this.

## Labeling rubric

For each row, read the **query**, the **retrieved_docs_concat**, and the **answer**. Ignore the reference column.

### Binary label (`label_binary`)

- **0 (hallucinated)** — the answer makes at least one factual claim that cannot be verified from the retrieved passages. Includes:
  - Specific facts (dates, names, numbers) not present in the passages.
  - Strong assertions plausibly true in general but not stated in the passages.
  - Claims that contradict the passages.
- **1 (grounded)** — every factual claim in the answer is verifiable from the retrieved passages. Includes:
  - Honest refusals ("the context does not contain information about X") when the context truly lacks the answer.
  - Direct quotes or close paraphrases of the passages.

### Grade label (`label_grade`)

- **5** — every claim is directly supported by a passage. Faithful summary or quote.
- **4** — every claim supported but slightly looser paraphrase or minor reorganization.
- **3** — mix of supported and weakly supported claims, no clear fabrication.
- **2** — at least one fabricated or contradicted claim, but the answer is partially grounded.
- **1** — answer is mostly or entirely fabricated.

### Edge cases

- **Refusal when context has the answer** → label 0 (it's not hallucinated but it's wrong; grade 2).
- **Refusal when context lacks the answer** → label 1, grade 5.
- **Answer is technically supported but answers the wrong question** → label 0, note "off-topic" in `notes`.
- **Answer cites a passage that wasn't actually used** (post-rationalization) → label 0, note "post-rationalized" in `notes`.

## Procedure

1. Open `labeling/to_label.csv` in Excel / Google Sheets / a CSV viewer.
2. Randomly pick 50 rows (or use rows 1–50 if you don't want to randomize).
3. For each row: fill `label_binary` and `label_grade`. Use `notes` for any edge cases.
4. Save as `labeling/labels.csv` (do NOT overwrite `to_label.csv`).
5. Aim for ~2 minutes per row. 50 rows → ~100 minutes.

## Stretch goal — inter-annotator agreement

If possible, get a friend or colleague to label 20 of the same rows independently. Save as `labeling/labels_annotator2.csv`. Task-07 will compute Cohen's κ.

Even one extra annotator on 20 rows gives a credible κ number. Without it, reviewers ask "is your labeling consistent?"

## Done when

- [ ] `labeling/labels.csv` exists with 50 rows fully labeled.
- [ ] No row has empty `label_binary` or `label_grade`.
- [ ] You have a rough sense of the label distribution. If all 50 are grounded or all 50 are hallucinated, something is wrong — re-examine the rubric or pick different rows.

## Do not

- Do not look at the metrics file while labeling.
- Do not use ChatGPT, Claude, or any LLM to label.
- Do not label more than 50 in one sitting — accuracy drops with fatigue.
- Do not label rows where you can't understand the question or context. Skip them and pick another.
