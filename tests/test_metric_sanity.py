"""
tests/test_metric_sanity.py
---------------------------
Sanity tests that pin the H_score metric's expected behavior on hand-crafted inputs.
Any future code change that breaks these tests is a regression and must be reviewed.

Rules:
  - No LLM calls (non-deterministic, would flake).
  - No Groq API access.
  - Test metric functions directly, not LangGraph wiring.
  - Assert with inequalities, not exact values (NLI scores vary across hardware).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import (
    compute_h_score, compute_answer_relevance, _cosine_sim,
    ALPHA, BETA, GAMMA, DELTA, THRESHOLD, DRIFT_CUTOFF,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def h_composite(nli: dict, answer_relevance: float) -> float:
    effective_faith = nli["faithfulness"] * nli["claim_coverage"]
    return (
        ALPHA * effective_faith +
        BETA  * nli["claim_coverage"] +
        GAMMA * (1 - nli["contradiction"]) +
        DELTA * answer_relevance
    )


# ── Test 1: directly grounded ─────────────────────────────────────────────────

def test_directly_grounded():
    """
    Answer is the exact same text as the single retrieved passage.
    When premise == hypothesis the NLI model must score entailment near 1.0,
    giving faithfulness ≈ 1.0, claim_coverage = 1.0, contradiction ≈ 0.
    We use a self-contained (passage, answer) pair rather than the shared
    rag_docs fixture to avoid the NLI model's paragraph-level scoring quirks
    when comparing a sentence against a multi-sentence passage.
    """
    passage = (
        "Faithfulness measures whether the generated answer is factually consistent "
        "with the retrieved context."
    )
    nli = compute_h_score(passage, [passage])
    assert nli["faithfulness"]   > 0.85, f"faithfulness={nli['faithfulness']}"
    assert nli["claim_coverage"] > 0.85, f"claim_coverage={nli['claim_coverage']}"
    assert nli["contradiction"]  < 0.10, f"contradiction={nli['contradiction']}"


# ── Test 2: off-topic answer ──────────────────────────────────────────────────

def test_off_topic(rag_docs, answer_off_topic):
    """Answer completely unrelated to docs — faithfulness and coverage must be low."""
    nli = compute_h_score(answer_off_topic, rag_docs)
    assert nli["faithfulness"]   < 0.20, f"faithfulness={nli['faithfulness']}"
    assert nli["claim_coverage"] < 0.20, f"claim_coverage={nli['claim_coverage']}"


# ── Test 3: partial grounding ─────────────────────────────────────────────────

def test_partial_grounding(rag_docs, answer_partial, query_dense_retrieval):
    """1 grounded + 5 fabricated sentences — coverage low, composite below threshold."""
    nli = compute_h_score(answer_partial, rag_docs)
    rel = compute_answer_relevance(query_dense_retrieval, answer_partial)

    assert nli["claim_coverage"] < 0.30, f"claim_coverage={nli['claim_coverage']}"

    composite = h_composite(nli, rel)
    assert composite < THRESHOLD, (
        f"composite={composite:.4f} should be < threshold={THRESHOLD}. "
        f"nli={nli}  rel={rel:.4f}"
    )


# ── Test 4: contradicted answer ───────────────────────────────────────────────

def test_contradicted(rag_docs, answer_contradicted):
    """Answer contradicts the retrieved passage — contradiction component must be high."""
    nli = compute_h_score(answer_contradicted, rag_docs)
    assert nli["contradiction"] > 0.30, f"contradiction={nli['contradiction']}"


# ── Test 5: answer relevance — off-topic answer ───────────────────────────────

def test_answer_relevance_off_topic():
    """
    Query about dense retrieval, answer about baking bread.
    Completely different domain — cosine similarity must be low.
    This validates that the δ component catches post-rationalization
    (an answer that sounds fine but answers a different question entirely).
    Note: two ML-domain texts (e.g. dense retrieval vs CNNs) share enough
    embedding space to produce sim > 0.5, so we use a truly unrelated domain.
    """
    query  = "What is dense retrieval?"
    answer = (
        "To bake sourdough bread, combine flour, water, salt and a starter culture. "
        "Allow the dough to ferment overnight before shaping and baking at high heat."
    )
    rel = compute_answer_relevance(query, answer)
    assert rel < 0.50, f"answer_relevance={rel:.4f} — expected < 0.50 for a completely off-topic answer"


# ── Test 6: drift guard triggers ─────────────────────────────────────────────

def test_drift_guard_triggers():
    """
    _cosine_sim between an original RAG query and a completely unrelated
    refined query must fall below DRIFT_CUTOFF, confirming the guard fires.
    """
    original = "What is dense retrieval and how does it work in RAG systems?"
    drifted  = "What are the best recipes for Italian pasta carbonara?"

    sim = _cosine_sim(original, drifted)
    assert sim < DRIFT_CUTOFF, (
        f"sim={sim:.4f} should be < DRIFT_CUTOFF={DRIFT_CUTOFF}. "
        "The drift guard would not have fired on this pair."
    )
