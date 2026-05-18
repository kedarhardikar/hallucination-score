# Task 13 — Qualitative Examples

**Read `CONTEXT.md` first. Depends on task-07.**

## Goal

Pick 6 examples from the labeled set that showcase H_score's behavior. These go into a "Qualitative Analysis" subsection of the paper.

## Why

Reviewers always ask for examples. Quantitative tables show the metric works on average; qualitative examples show *how* and *when* it works (and fails).

## Steps

1. **Write `scripts/find_qualitative_examples.py`.**
   - Reads `labeling/labels_full.csv`.
   - Categorizes rows into:
     - **A — H_score correct, RAGAS wrong** (H_score < threshold and label = hallucinated, but RAGAS faithfulness > 0.5).
     - **B — RAGAS correct, H_score wrong** (RAGAS faithfulness < 0.5 and label = hallucinated, but H_score > threshold).
     - **C — Clear H_score failures** (H_score > 0.8 but label = 1 or 2).
     - **D — Drift guard fires** (`drift_rejected == True`, regardless of label).
   - For each category, prints the top 3 candidates (sorted by how cleanly they fit the category).

2. **Human picks the final 6** from the printed candidates. Roughly: 2 of category A, 2 of category B, 1 of C, 1 of D.

3. **For each chosen example, write a markdown file** in `paper_drafts/qualitative/example_N.md` containing:
   - Query.
   - Retrieved passages (full text, not truncated).
   - Answer.
   - All metric scores (H_score components, RAGAS, HHEM, ROUGE-L).
   - Human label.
   - Commentary (1 paragraph) — why this example matters, what it shows about H_score, what failure mode it illustrates.

4. **Combine into `paper_drafts/qualitative_analysis.md`** for direct inclusion in the paper.

## Done when

- [ ] `paper_drafts/qualitative/example_1.md` through `example_6.md` exist.
- [ ] `paper_drafts/qualitative_analysis.md` combines them with framing prose.
- [ ] At least one example per category A, B, C, D appears.

## Do not

- Do not invent examples or modify the actual outputs. Use only real rows from the labeled CSV.
- Do not skip category C (H_score failures). Reviewers respect honesty about failures.
- Do not let Claude Code make the final pick of which 6 to use — you do that after seeing the candidates. The narrative judgment is yours.
