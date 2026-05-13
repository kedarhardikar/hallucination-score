"""
hscore_langgraph.py
-------------------
LangGraph RAG pipeline with H_score hallucination metric.
Uses Groq API for generation + LlamaIndex for retrieval.

Dependencies:
    pip install langgraph llama-index llama-index-llms-groq \
                llama-index-embeddings-huggingface transformers torch
"""

import os
import re
from typing import TypedDict, List, Optional
from functools import partial

from langgraph.graph import StateGraph, END
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import pipeline as hf_pipeline
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")

# ── Groq + Embedding config ───────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"

Settings.llm         = Groq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ── H_score weights & thresholds ─────────────────────────────────────────────
ALPHA     = 0.4   # Faithfulness weight
BETA      = 0.4   # Claim Coverage weight
GAMMA     = 0.2   # (1 - Contradiction) weight
THRESHOLD = 0.65
MAX_RETRIES = 3

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
    query:          str
    original_query: str
    retrieved_docs: List[str]
    answer:         str
    h_score:        float
    faithfulness:   float
    claim_coverage: float
    contradiction:  float
    retries:        int
    best_answer:    Optional[str]
    best_h_score:   float
    final_answer:   Optional[str]


# ── Helpers ───────────────────────────────────────────────────────────────────
def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def nli_score(premise: str, hypothesis: str) -> dict:
    """Run NLI inference on a premise/hypothesis pair using the cross-encoder."""
    nli = _get_nli()
    # return_all_scores gives a score for every label, not just the top one
    results = nli({"text": premise, "text_pair": hypothesis}, top_k=None)
    return {r["label"].lower(): r["score"] for r in results}


def compute_h_score(answer: str, docs: List[str]) -> dict:
    """
    H_score = α·Faithfulness + β·ClaimCoverage + γ·(1 − ContradictionRate)
    """
    context   = " ".join(docs)
    sentences = split_sentences(answer)

    if not sentences:
        return {"h_score": 0.0, "faithfulness": 0.0,
                "claim_coverage": 0.0, "contradiction": 0.0}

    entailment_scores, contradiction_scores, covered = [], [], 0
    print(f"Context retrieved from db: {context}")
    print(f"Sentences (answer from llm after splitting:) {sentences}")

    for sent in sentences:
        scores = nli_score(premise=context, hypothesis=sent)
        print(f"scores after nli: {scores}")
        entailment_scores.append(scores.get("entailment", 0))
        contradiction_scores.append(scores.get("contradiction", 0))
        if scores.get("entailment", 0) > 0.5:
            covered += 1

    faithfulness   = sum(entailment_scores) / len(entailment_scores)
    claim_coverage = covered / len(sentences)
    contradiction  = sum(contradiction_scores) / len(contradiction_scores)
    h_score = ALPHA * faithfulness + BETA * claim_coverage + GAMMA * (1 - contradiction)

    return {
        "h_score":        round(h_score, 4),
        "faithfulness":   round(faithfulness, 4),
        "claim_coverage": round(claim_coverage, 4),
        "contradiction":  round(contradiction, 4),
    }


# ── LangGraph Nodes ───────────────────────────────────────────────────────────
def retrieve_node(state: RAGState, index: VectorStoreIndex) -> RAGState:
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(state["query"])
    state["retrieved_docs"] = [n.get_content() for n in nodes]
    return state


def generate_node(state: RAGState) -> RAGState:
    """Generate answer from the already-retrieved docs so H_score evaluates the same context."""
    context = "\n\n".join(state["retrieved_docs"])
    prompt = (
        f"Answer the following question using only the provided context. "
        f"If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\n\n"
        f"Answer:"
    )
    response = Settings.llm.complete(prompt)
    state["answer"] = str(response).strip()
    return state


def hallucination_metric_node(state: RAGState) -> RAGState:
    metrics = compute_h_score(state["answer"], state["retrieved_docs"])
    state.update(metrics)

    if state["h_score"] > state["best_h_score"]:
        state["best_h_score"] = state["h_score"]
        state["best_answer"]  = state["answer"]

    print(f"\n[H_score Node] Retry #{state['retries']}")
    print(f"  H_score      : {state['h_score']}")
    print(f"  Faithfulness : {state['faithfulness']}")
    print(f"  Claim Cov.   : {state['claim_coverage']}")
    print(f"  Contradiction: {state['contradiction']}")
    print(f"  Best so far  : {state['best_h_score']}")
    return state


def refine_query_node(state: RAGState) -> RAGState:
    """Use Groq to generate a better query when H_score is too low."""
    state["retries"] += 1
    prompt = (
        f"The following question was answered with low factual grounding:\n"
        f"Question: {state['query']}\n"
        f"Answer: {state['answer']}\n\n"
        f"Rewrite the question to be more specific so the answer "
        f"stays grounded in retrieved context. Return only the rewritten question."
    )
    refined = Settings.llm.complete(prompt)
    state["query"] = str(refined).strip()
    print(f"\n[Refine Node] New query: {state['query']}")
    return state


def finalize_node(state: RAGState) -> RAGState:
    state["final_answer"] = state["best_answer"]
    print(f"\n✅ Finalized. Best H_score = {state['best_h_score']}")
    return state


# ── Conditional Edge ──────────────────────────────────────────────────────────
def should_retry(state: RAGState) -> str:
    if state["h_score"] >= THRESHOLD or state["retries"] >= MAX_RETRIES:
        return "finalize"
    return "refine"


# ── Graph Builder ─────────────────────────────────────────────────────────────
def build_rag_graph(index: VectorStoreIndex):
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

    g.add_conditional_edges(
        "hallucination_metric",
        should_retry,
        {"refine": "refine", "finalize": "finalize"},
    )

    return g.compile()


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dataset import get_contexts_as_documents
    from llama_index.core import VectorStoreIndex

    docs  = get_contexts_as_documents()
    index = VectorStoreIndex.from_documents(docs)
    graph = build_rag_graph(index)

    query = "What are the main causes of hallucination in RAG systems?"
    initial: RAGState = {
        "query":          query,
        "original_query": query,
        "retrieved_docs": [],
        "answer":         "",
        "h_score":        0.0,
        "faithfulness":   0.0,
        "claim_coverage": 0.0,
        "contradiction":  0.0,
        "retries":        0,
        "best_answer":    None,
        "best_h_score":   -1.0,
        "final_answer":   None,
    }

    result = graph.invoke(initial)
    print("\n── Final Answer ──────────────────────────────────")
    print(result["final_answer"])
