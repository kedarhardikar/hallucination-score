# Project Context — H_score for RAG Hallucination Detection

**Read this first in every session. Do not modify it.**

## What this project is

A RAG pipeline that scores its own outputs for hallucination using a novel composite metric (**H_score**) and uses that score to drive a corrective query-refinement loop. The end goal is a research paper.

## H_score formula

```
H_score = 0.30 · (Faithfulness × ClaimCoverage)   (strength × breadth, coupled)
        + 0.30 · ClaimCoverage                     (fraction of sentences grounded)
        + 0.15 · (1 − Contradiction)               (fraction of sentences contradicted by best-matching passage)
        + 0.25 · AnswerRelevance                   (cosine sim between original query and answer)
```

Faithfulness (avg entailment of covered sentences) is multiplied by ClaimCoverage
before weighting so that grounding strength and breadth are inseparable — a single
well-grounded sentence cannot inflate the score when most sentences are unsupported.

**Split-premise NLI strategy (multi-hop fix):**
- Faithfulness and ClaimCoverage use the CONCATENATED retrieved passages as the NLI
  premise. This lets the model verify claims that require synthesising facts across
  multiple passages (multi-hop), which no single passage fully entails.
- Contradiction uses the BEST-MATCHING single passage as the premise, to avoid
  false positives from noisy/irrelevant passages contradicting a correct answer.

NLI: `cross-encoder/nli-deberta-v3-small`
Embeddings: `BAAI/bge-small-en-v1.5`
LLM: Groq `llama-3.3-70b-versatile`
Vector store: ChromaDB (persistent)

## Repo layout

```
main.py        — pipeline, metric, LangGraph graph
dataset.py     — stress dataset (10 hand-crafted items) + HotpotQA loader
evaluate.py    — batch eval harness (CSV/JSON/LaTeX exports)
ablation.py    — component / weight grid / refinement ablations
db.py          — ChromaDB persistence layer
req.txt        — dependencies
```

## What is paper-critical

The single biggest gap blocking the paper is **empirical validation of the metric**. Specifically:

1. No human labels of grounded vs hallucinated.
2. No empirical comparison against RAGAS or HHEM (only conceptual comparisons in `RESEARCH.md`).
3. HotpotQA evaluation runs only 50 samples (too small).
4. FEVER is in the research plan but not implemented.

Everything in `tasks/` is structured around closing these gaps.

## Critical design choices — do not change without discussion

- **Faithfulness over covered sentences only.** Grounding strength is separated from grounding breadth (coverage).
- **Contradiction from the best-matching passage**, not max contradiction across all passages. Prevents false positives from irrelevant noisy retrieval.
- **AnswerRelevance always uses `original_query`**, never the refined query. Topic drift must always be penalized.
- **Drift guard** in `refine_query_node`: refined query rejected if cosine similarity to original < 0.5.
- **`should_retry` uses `best_h_score`**, not the latest `h_score`. A worse retry cannot trigger acceptance.
- **Source-isolated retrieval** for stress queries (via `filter_source_id` metadata filter). HotpotQA queries search the full pool.

## How tasks are organized

Each file in `tasks/` is a single scoped work block. Format:

- **Goal** — what done looks like.
- **Why** — how it fits into the paper.
- **Steps** — concrete actions.
- **Done when** — checkable criteria.
- **Do not** — guardrails.

Work tasks in numerical order. **Do not start a task without finishing the previous one.** Some tasks depend on outputs of earlier ones.

## Running list of canonical commands

```bash
python evaluate.py stress                  # 10-query stress eval
python evaluate.py hotpotqa                # 50-query HotpotQA eval (default)
python evaluate.py hotpotqa --no-refine    # ablation: single-pass
python ablation.py all                     # all three ablation experiments
python db.py inspect stress                # view stored chunks
python db.py reset stress                  # force rebuild
```
