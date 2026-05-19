"""
tests/conftest.py
-----------------
Shared fixtures for H_score metric sanity tests.
All fixtures use fixed strings — no LLM calls, no network access.
"""

import pytest


# ── Retrieved passage fixtures ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rag_docs():
    """A small set of fixed passages that stand in for retrieved documents."""
    return [
        # passage 0 — about dense retrieval in RAG (used for grounded / off-topic tests)
        "Dense retrieval in RAG systems encodes queries and documents into dense vector "
        "embeddings using a bi-encoder model. Approximate nearest neighbour search is then "
        "used to find the most relevant passages from a large corpus at query time.",

        # passage 1 — about faithfulness evaluation
        "Faithfulness measures whether the generated answer is factually consistent with "
        "the retrieved context. An answer is considered faithful if all its claims can be "
        "verified against the source documents provided to the model.",

        # passage 2 — used to create a direct contradiction
        "The Eiffel Tower was completed in 1889 and stands 330 metres tall. "
        "It was designed by Gustave Eiffel and served as the entrance arch for the 1889 World's Fair.",
    ]


# ── Answer fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def answer_grounded(rag_docs):
    """
    First sentence of passage 0, verbatim.
    Single-sentence answer guarantees coverage is 0 or 1 — no partial-coverage edge cases.
    The NLI model can fail to entail a sentence > 0.5 even from its own passage when
    the passage contains multiple sentences (paragraph-level premise vs. isolated hypothesis).
    Using a single sentence makes the grounded/not-grounded determination unambiguous.
    """
    from main import split_sentences
    return split_sentences(rag_docs[0])[0]


@pytest.fixture(scope="session")
def answer_off_topic():
    """Answer completely unrelated to the RAG passages."""
    return "The sky is blue. Grass is green."


@pytest.fixture(scope="session")
def answer_partial():
    """1 grounded sentence + 5 fabricated sentences against the RAG docs."""
    return (
        "Dense retrieval encodes queries and documents into dense vector embeddings. "
        "The retriever was invented in 1650 by Sir Isaac Newton. "
        "All RAG systems require at least 100 GB of GPU memory to operate. "
        "Documents are always retrieved from a single centralised server in Antarctica. "
        "Bi-encoders were banned by the EU in 2022 due to privacy concerns. "
        "The standard chunk size for RAG is exactly 4,096 tokens with no exceptions."
    )


@pytest.fixture(scope="session")
def answer_contradicted(rag_docs):
    """
    Uses explicit logical negations of passage 1 (faithfulness definition).
    Direct negations are reliably scored as CONTRADICTION by NLI models,
    unlike factual errors (wrong dates/numbers) which are scored as NEUTRAL.
    """
    return (
        "Faithfulness does not measure factual consistency with the retrieved context. "
        "An answer can be considered faithful even if none of its claims appear in the source documents."
    )


@pytest.fixture(scope="session")
def query_dense_retrieval():
    return "What is dense retrieval and how does it work in RAG systems?"


@pytest.fixture(scope="session")
def query_eiffel():
    return "When was the Eiffel Tower completed and how tall is it?"
