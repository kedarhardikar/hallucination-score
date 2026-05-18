"""
hscore_langgraph.py
-------------------
LangGraph RAG pipeline with H_score hallucination metric.
Uses Groq API for generation + LlamaIndex for retrieval.

Dependencies:
    pip install langgraph llama-index llama-index-llms-groq \
                llama-index-embeddings-huggingface transformers torch nltk
"""

import os
import math
from typing import TypedDict, List, Optional
from functools import partial

import nltk
from langgraph.graph import StateGraph, END
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import pipeline as hf_pipeline
from dotenv import load_dotenv

# Download sentence tokeniser data once (silent if already present)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")

# ── Groq + Embedding config ───────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"

Settings.llm         = Groq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ── H_score weights & thresholds ─────────────────────────────────────────────
ALPHA     = 0.30   # Faithfulness weight       (grounding strength)
BETA      = 0.30   # Claim Coverage weight     (grounding breadth)
GAMMA     = 0.15   # (1 - Contradiction) weight (conflict penalty)
DELTA     = 0.25   # Answer Relevance weight   (topic drift penalty)
THRESHOLD    = 0.65
MAX_RETRIES  = 3
DRIFT_CUTOFF = 0.5  # Min cosine sim between original query and refined query

# ── NLI model — lazy-loaded on first use ──────────────────────────────────────
_nli_pipeline = None


def _get_nli():
    global _nli_pipeline
    if _nli_pipeline is None:
        _nli_pipeline = hf_pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=-1,   # set to 0 for GPU
        )
    return _nli_pipeline


# ── Graph State ───────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    query:            str
    original_query:   str
    retrieved_docs:   List[str]
    answer:           str
    h_score:          float
    faithfulness:     float
    claim_coverage:   float
    contradiction:    float
    answer_relevance: float
    retries:          int
    best_answer:      Optional[str]
    best_h_score:     float
    final_answer:     Optional[str]
    filter_source_id: Optional[str]   # if set, retrieval is restricted to this source_id


# ── Helpers ───────────────────────────────────────────────────────────────────
def split_sentences(text: str) -> List[str]:
    """
    Sentence tokeniser using NLTK — handles abbreviations (Dr., e.g., U.S.A.)
    and edge cases that naive regex splitting breaks on.
    """
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def _cosine_sim(text_a: str, text_b: str) -> float:
    """
    Cosine similarity between two texts using the loaded embedding model.
    Shared helper used by both compute_answer_relevance and the drift guard.
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0
    emb_a  = Settings.embed_model.get_text_embedding(text_a)
    emb_b  = Settings.embed_model.get_text_embedding(text_b)
    dot    = sum(x * y for x, y in zip(emb_a, emb_b))
    norm_a = math.sqrt(sum(x * x for x in emb_a))
    norm_b = math.sqrt(sum(y * y for y in emb_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_answer_relevance(original_query: str, answer: str) -> float:
    """
    Cosine similarity between the original query and the generated answer.
    Measures whether the answer addresses what was actually asked.
    Uses original_query (never the refined query) so topic drift is always detected.
    """
    return _cosine_sim(original_query, answer)


def nli_score(premise: str, hypothesis: str) -> dict:
    """Run NLI inference on a premise/hypothesis pair using the cross-encoder."""
    nli     = _get_nli()
    results = nli({"text": premise, "text_pair": hypothesis}, top_k=None)
    return {r["label"].lower(): r["score"] for r in results}


def compute_h_score(answer: str, docs: List[str]) -> dict:
    """
    Returns NLI-based components only (faithfulness, claim_coverage, contradiction).
    The final H_score is assembled in hallucination_metric_node after adding
    the answer_relevance (DELTA) component.

    Per-sentence × per-passage NLI scoring:
      For each answer sentence, find the BEST-MATCHING passage (max entailment).

      Faithfulness  = avg entailment of COVERED sentences only
                      (grounding STRENGTH — how confidently are grounded claims supported)
      ClaimCoverage = fraction of sentences with best entailment > 0.5
                      (grounding BREADTH — how many claims are grounded at all)
      Contradiction = fraction of sentences where the best-matching passage
                      ALSO contradicts them (intrinsic hallucination signal)

    Key design: contradiction uses the SAME best-matching passage, not the worst
    passage across all. Prevents false positives from irrelevant noisy passages.
    """
    sentences = split_sentences(answer)

    if not sentences or not docs:
        return {"faithfulness": 0.0, "claim_coverage": 0.0, "contradiction": 0.0}

    covered_entailment_scores = []
    covered      = 0
    contradicted = 0

    print(f"[H_score] {len(sentences)} answer sentences × {len(docs)} passages")

    for s_idx, sent in enumerate(sentences):
        ent_per_passage = []
        con_per_passage = []

        for passage in docs:
            scores = nli_score(premise=passage, hypothesis=sent)
            ent_per_passage.append(scores.get("entailment",    0.0))
            con_per_passage.append(scores.get("contradiction", 0.0))

        best_idx           = ent_per_passage.index(max(ent_per_passage))
        best_entailment    = ent_per_passage[best_idx]
        best_contradiction = con_per_passage[best_idx]  # from same best-matching passage

        is_covered      = best_entailment    > 0.5
        is_contradicted = best_contradiction > 0.5

        if is_covered:
            covered += 1
            covered_entailment_scores.append(best_entailment)
        if is_contradicted:
            contradicted += 1

        print(f"  sent[{s_idx}] best_entail={best_entailment:.3f} "
              f"contra_from_best={best_contradiction:.3f} -> "
              f"{'covered' if is_covered else 'uncovered'}"
              f"{' + CONTRADICTED' if is_contradicted else ''}")

    faithfulness   = (
        sum(covered_entailment_scores) / len(covered_entailment_scores)
        if covered_entailment_scores else 0.0
    )
    claim_coverage = covered      / len(sentences)
    contradiction  = contradicted / len(sentences)

    return {
        "faithfulness":   round(faithfulness,   4),
        "claim_coverage": round(claim_coverage, 4),
        "contradiction":  round(contradiction,  4),
    }


# ── LangGraph Nodes ───────────────────────────────────────────────────────────
def retrieve_node(state: RAGState, index: VectorStoreIndex) -> RAGState:
    """
    Retrieves top-5 passages for the current query.

    If filter_source_id is set (stress queries), retrieval is restricted to passages
    from that specific item only — so each stress query only searches its own designated
    context set and cannot accidentally pull in relevant passages from other items.

    For HotpotQA and free-form queries, filter_source_id is None and the full collection
    is searched.
    """
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

    kwargs: dict = {"similarity_top_k": 5}
    if state.get("filter_source_id"):
        kwargs["filters"] = MetadataFilters(filters=[
            MetadataFilter(key="source_id", value=state["filter_source_id"])
        ])

    retriever = index.as_retriever(**kwargs)
    nodes = retriever.retrieve(state["query"])
    state["retrieved_docs"] = [n.get_content() for n in nodes]
    return state


def generate_node(state: RAGState) -> RAGState:
    """Generate answer from already-retrieved docs so H_score evaluates the same context."""
    context = "\n\n".join(state["retrieved_docs"])
    prompt  = (
        f"Answer the following question using only the provided context. "
        f"If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\n\n"
        f"Answer:"
    )
    response       = Settings.llm.complete(prompt)
    state["answer"] = str(response).strip()
    return state


def hallucination_metric_node(state: RAGState) -> RAGState:
    # Step 1: NLI components (faithfulness, coverage, contradiction)
    nli_metrics = compute_h_score(state["answer"], state["retrieved_docs"])

    # Step 2: Answer relevance — original query ↔ generated answer
    # Always uses original_query so drift from refinement is always detected
    answer_relevance = compute_answer_relevance(state["original_query"], state["answer"])

    # Step 3: Assemble full H_score
    h_score = (
        ALPHA * nli_metrics["faithfulness"] +
        BETA  * nli_metrics["claim_coverage"] +
        GAMMA * (1 - nli_metrics["contradiction"]) +
        DELTA * answer_relevance
    )

    state.update({
        **nli_metrics,
        "h_score":          round(h_score,         4),
        "answer_relevance": round(answer_relevance, 4),
    })

    if state["h_score"] > state["best_h_score"]:
        state["best_h_score"] = state["h_score"]
        state["best_answer"]  = state["answer"]

    print(f"\n[H_score Node] Retry #{state['retries']}")
    print(f"  H_score          : {state['h_score']}")
    print(f"  Faithfulness     : {state['faithfulness']}")
    print(f"  Claim Cov.       : {state['claim_coverage']}")
    print(f"  Contradiction    : {state['contradiction']}")
    print(f"  Answer Relevance : {state['answer_relevance']}")
    print(f"  Best so far      : {state['best_h_score']}")
    return state


def refine_query_node(state: RAGState) -> RAGState:
    """
    Rewrites the query to improve retrieval of the ORIGINAL topic.
    The prompt explicitly anchors to the original question so the LLM
    cannot change the subject to match available context.

    Drift guard: if the refined query drifts too far from the original
    (cosine sim < DRIFT_CUTOFF), the refinement is rejected and retries
    are maxed out to force immediate finalization.
    """
    state["retries"] += 1

    prompt = (
        f"The question below was answered with low factual grounding.\n\n"
        f"Original question : {state['original_query']}\n"
        f"Current question  : {state['query']}\n"
        f"Current answer    : {state['answer']}\n\n"
        f"Rewrite the CURRENT question to be more specific and detailed "
        f"so the retriever can find better supporting passages.\n"
        f"STRICT RULES:\n"
        f"  1. Do NOT change the topic or subject of the original question.\n"
        f"  2. Do NOT ask about something different just because it appears in the context.\n"
        f"  3. Only add specificity — expand keywords, add context clues.\n"
        f"Return only the rewritten question, nothing else."
    )
    refined_query = str(Settings.llm.complete(prompt)).strip()

    # Drift guard: reject if refined query has strayed too far from original
    drift_sim = _cosine_sim(state["original_query"], refined_query)
    if drift_sim < DRIFT_CUTOFF:
        print(f"\n[Refine Node] ⚠ Drift detected (sim={drift_sim:.3f} < {DRIFT_CUTOFF}) "
              f"— rejecting refinement, forcing finalize")
        state["retries"] = MAX_RETRIES   # triggers finalize on next should_retry check
    else:
        state["query"] = refined_query
        print(f"\n[Refine Node] Accepted (sim={drift_sim:.3f}): {state['query']}")

    return state


def finalize_node(state: RAGState) -> RAGState:
    state["final_answer"] = state["best_answer"]
    print(f"\n✅ Finalized. Best H_score = {state['best_h_score']}")
    return state


# ── Conditional Edge ──────────────────────────────────────────────────────────
def should_retry(state: RAGState) -> str:
    if state["best_h_score"] >= THRESHOLD or state["retries"] >= MAX_RETRIES:
        return "finalize"
    return "refine"


# ── Graph Builder ─────────────────────────────────────────────────────────────
def build_rag_graph(index: VectorStoreIndex, no_refine: bool = False):
    """
    no_refine=False  → full pipeline with query refinement loop (default)
    no_refine=True   → single-pass pipeline, used for ablation study
    """
    g = StateGraph(RAGState)

    g.add_node("retrieve",             partial(retrieve_node, index=index))
    g.add_node("generate",             generate_node)
    g.add_node("hallucination_metric", hallucination_metric_node)
    g.add_node("refine",               refine_query_node)
    g.add_node("finalize",             finalize_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve",             "generate")
    g.add_edge("generate",             "hallucination_metric")
    g.add_edge("refine",               "retrieve")
    g.add_edge("finalize",             END)

    if no_refine:
        g.add_edge("hallucination_metric", "finalize")
    else:
        g.add_conditional_edges(
            "hallucination_metric",
            should_retry,
            {"refine": "refine", "finalize": "finalize"},
        )

    return g.compile()


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dataset import get_contexts_as_documents
    from db import build_or_load_index, collection_exists

    docs  = None if collection_exists(collection_name="stress") \
            else get_contexts_as_documents()
    index = build_or_load_index(docs, collection_name="stress")
    graph = build_rag_graph(index)

    query = "What are the main causes of hallucination in RAG systems?"
    initial: RAGState = {
        "query":            query,
        "original_query":   query,
        "retrieved_docs":   [],
        "answer":           "",
        "h_score":          0.0,
        "faithfulness":     0.0,
        "claim_coverage":   0.0,
        "contradiction":    0.0,
        "answer_relevance": 0.0,
        "retries":          0,
        "best_answer":      None,
        "best_h_score":     -1.0,
        "final_answer":     None,
        "filter_source_id": None,   # None = search full collection
    }

    result = graph.invoke(initial)
    print("\n── Final Answer ──────────────────────────────────")
    print(result["final_answer"])
