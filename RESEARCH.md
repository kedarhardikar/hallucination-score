# Research Significance — H_score for RAG Hallucination Detection

## What Problem Does This Project Solve?

Large Language Models used in RAG systems hallucinate in two distinct ways:

1. **Grounding failures** — the answer makes claims that are not supported by the retrieved documents
2. **Topic drift** — the answer is technically grounded but answers a different question than what was asked (happens when query refinement lets the LLM silently change the subject)

Existing metrics either measure only one of these, or conflate them in ways that produce misleading scores.

---

## What Already Exists and Why It Is Not Enough

### RAGAS
The most widely used RAG evaluation framework. Measures faithfulness, answer relevancy, context precision, and context recall — but as **separate, independent scores**.

**What RAGAS misses:**
- Scores are computed separately, so a single composite "is this answer trustworthy?" number does not exist
- Faithfulness is computed by an LLM-as-judge (GPT-4 or similar) — slow (~35 seconds per query), expensive, and inconsistent across runs (same input can produce different scores)
- Does not distinguish between a claim that is *unsupported* and one that is *directly contradicted* — both get the same treatment
- Answer relevance in RAGAS measures cosine similarity between a *regenerated* question and the original, not between the actual answer and the original question
- Does not guard against topic drift during query refinement

### HHEM (Hughes Hallucination Evaluation Model)
Fast (0.6 seconds), deterministic, classification-based. Better than RAGAS at detecting factual inconsistency.

**What HHEM misses:**
- Single dimension only — tells you whether the answer is consistent with context, but nothing about how much of the answer is covered, or whether it answers the right question
- No answer relevance component
- No claim coverage (breadth) — an answer that makes one correct claim and five unsupported claims scores the same as one that makes six correct claims

### TruLens
Measures groundedness, answer relevance, and context relevance as a RAG Triad.

**What TruLens misses:**
- LLM-based feedback functions — same speed/cost/inconsistency problems as RAGAS
- No contradiction distinction
- No query refinement mechanism or drift guard

### NLI-based metrics (standalone)
Fast and deterministic. Some papers use NLI entailment scores directly as faithfulness proxies.

**What standalone NLI misses:**
- No coverage component — a one-sentence answer that is perfectly entailed scores the same as a thorough answer that is fully grounded
- No answer relevance
- No contradiction signal — entailment and contradiction scores are computed but only entailment is typically used

---

## What H_score Does Differently

### 1. Four components in one score, not four separate scores

```
H_score = 0.30 · Faithfulness + 0.30 · ClaimCoverage
        + 0.15 · (1 − Contradiction) + 0.25 · AnswerRelevance
```

This gives a **single actionable number** that captures grounding strength, grounding breadth, conflict detection, and topic alignment simultaneously. RAGAS gives you four numbers that you have to manually interpret together.

### 2. Faithfulness and ClaimCoverage are deliberately separated

Most metrics treat these as the same thing. They are not:

- **Faithfulness** = *how strongly* are the grounded claims supported (avg entailment of covered sentences)
- **ClaimCoverage** = *how many* claims are grounded at all (fraction of sentences with best entailment > 0.5)

An answer can have high faithfulness but low coverage (one sentence is perfectly entailed, five are unsupported). An answer can have high coverage but low faithfulness (many sentences pass the 0.5 threshold but none are strongly entailed). H_score penalises both cases independently.

### 3. Contradiction is measured from the best-matching passage, not the worst

Existing NLI-based approaches either:
- Ignore contradiction entirely, or
- Flag a sentence as contradicted if *any* passage contradicts it

The second approach causes false positives in noisy retrieval — an irrelevant passage may accidentally score high on contradiction against a well-supported sentence. H_score uses the contradiction score from the **same passage that best entails the sentence**, which is the only passage actually relevant to that claim.

### 4. AnswerRelevance is computed against the original query

RAGAS computes answer relevance by generating a new question from the answer and comparing it to the original. This is indirect and LLM-dependent.

H_score computes cosine similarity between the **original query embedding** and the **answer embedding** directly, using the same local embedding model already running in the pipeline. This is deterministic, fast, and always references the original intent — not a derived version of it.

### 5. Semantic drift guard on query refinement

No existing metric addresses the query refinement problem. When an LLM rewrites a query to improve retrieval, it may silently drift to a different topic (one where good context exists). This produces high faithfulness scores for answers to the wrong question.

H_score includes a drift guard: before accepting any refined query, it computes cosine similarity between the original and refined query. If similarity < 0.5, the refinement is rejected and the pipeline finalises immediately. This is a novel safety mechanism not present in RAGAS, TruLens, HHEM, or standalone NLI metrics.

---

## Is This Meaningful for a Paper?

Yes — for the following reasons, backed by recent literature:

**"The Mirage of Hallucination Detection" (EMNLP 2025)** evaluated six metric families across 37 LLMs and four datasets. Its core finding: existing metrics show "weak to no correlation" with each other and frequently disagree with human judgments. The field explicitly needs better composite metrics.

**"Correctness is not Faithfulness in RAG Attributions" (SIGIR ICTIR 2025)** found that up to 57% of citations in RAG systems are post-rationalized — the model generated from parametric knowledge and then cited a document that technically matched, without actually using it. This is exactly the gap that AnswerRelevance in H_score targets: an answer can be technically faithful to retrieved text while still not answering what was asked.

**HALT-RAG (2025)** shows that three-way NLI classification (entailment / contradiction / neutral) outperforms binary approaches. H_score adopts this framing by treating contradiction as an independent signal rather than just the absence of entailment.

**ReDeEP (ICLR 2025)** argues for decoupling external context contribution from parametric knowledge contribution. H_score's separation of Faithfulness from AnswerRelevance operationalises exactly this distinction at the output level.

---

## Your Novelty — Specific Claims You Can Make

| Claim | What supports it |
|---|---|
| H_score unifies four independent hallucination signals into one interpretable score | No existing metric (RAGAS, HHEM, TruLens) does this with a single weighted formula |
| Faithfulness and ClaimCoverage are decoupled | Literature conflates them; H_score treats them as measuring different failure modes |
| Contradiction is measured from the best-matching passage | Prevents false positives from noisy retrieval — documented limitation not addressed elsewhere |
| AnswerRelevance is computed directly, not via LLM re-generation | Faster, deterministic, no API calls — distinct from RAGAS's indirect approach |
| Semantic drift guard on query refinement | Not present in any existing evaluation framework |
| H_score drives a corrective refinement loop | Shows the metric is actionable, not just diagnostic |
| Validated on adversarial stress types (irrelevant, conflicting, missing, noisy) | Structured failure-mode testing absent from RAGAS and HHEM evaluations |

---

## Honest Limitations to Acknowledge in the Paper

Research reviewers will respect honesty about limitations more than overclaiming.

- **No human evaluation baseline** — H_score correlates with ROUGE-L but is not validated against human hallucination judgments. This is the most significant limitation.
- **NLI model limitations** — `cross-encoder/nli-deberta-v3-small` struggles with complex multi-hop reasoning chains and numerical claims. Scores may be unreliable for highly technical or quantitative answers.
- **Stress dataset is small** — 10 hand-crafted items. Results are illustrative, not statistically significant at that scale. HotpotQA (50 samples) adds realism but is still a narrow slice.
- **ROUGE-L as proxy** — ROUGE-L measures lexical overlap with a reference answer, not factual correctness. It is a weak ground truth for hallucination evaluation.
- **Source-isolated retrieval for stress queries** — each stress query only searches its own context set. This tests the metric in isolation but does not reflect real RAG deployments where all documents are in one pool.
- **Weight selection** — the α=0.30, β=0.30, γ=0.15, δ=0.25 weights are justified by the ablation grid but not by theoretical derivation. Different domains may need different weights.

---

## How This Maps to Paper Sections

| Paper Section | What to write | What results support it |
|---|---|---|
| Introduction | RAG hallucination problem, why existing metrics are insufficient | Citations from RAGAS, HHEM, The Mirage paper |
| Related Work | RAGAS, TruLens, HHEM, NLI-based metrics, query refinement drift | Comparison table above |
| Method (Section 3) | H_score formula, per-sentence NLI, best-matching passage contradiction, AnswerRelevance, drift guard | `main.py` |
| Experiments (Section 4) | Stress dataset results, HotpotQA results, H_score vs ROUGE-L correlation | `eval_results/summary_*.json` |
| Ablation (Section 4.x) | Component ablation, weight sensitivity, refinement loop contribution | `ablation_results/` |
| Discussion | Failure modes still not captured, limitations, future work | Limitations section above |
