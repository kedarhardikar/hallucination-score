"""
dataset.py
----------
Two dataset sources for RAG hallucination research:

  1. STRESS_DATASET  — adversarial stress-test (irrelevant, conflicting, missing, noisy)
  2. HotpotQA        — real multi-hop QA; context is embedded in each sample

Public API
----------
  get_hotpotqa_documents(samples)  → List[Document]
  get_hotpotqa_queries(samples)    → List[(id, question, answer, "hotpotqa")]
  get_stress_contexts_as_documents() → List[Document]
  get_stress_queries()             → List[(id, query, reference, stress_type)]
  get_contexts_as_documents()      → alias for stress version
  get_queries()                    → alias for stress version
"""

# ── Adversarial stress-test dataset ──────────────────────────────────────────

STRESS_DATASET = [

    # ══ IRRELEVANT RETRIEVAL ══════════════════════════════════════════════════
    {
        "id": "s001",
        "stress_type": "irrelevant",
        "query": "How does dense retrieval work in RAG?",
        "contexts": [
            "Convolutional neural networks (CNNs) are primarily used for image recognition tasks. "
            "They apply filters across spatial dimensions to detect edges, textures, and objects.",

            "Gradient descent is an optimization algorithm used to minimize a loss function "
            "by iteratively adjusting model parameters in the direction of steepest descent.",

            "Transformer architectures rely on self-attention mechanisms to model dependencies "
            "between tokens regardless of their distance in the sequence.",
        ],
        "correct_context_idx": -1,
        "reference": (
            "Dense retrieval encodes queries and documents into vector embeddings "
            "and uses approximate nearest neighbour search to find relevant passages."
        ),
        "note": "No relevant context retrieved. Model must rely on parametric memory → hallucination expected.",
    },
    {
        "id": "s002",
        "stress_type": "irrelevant",
        "query": "What is the purpose of re-ranking in a RAG pipeline?",
        "contexts": [
            "Data augmentation techniques such as random cropping, flipping, and color jitter "
            "are commonly used in computer vision to improve model generalization.",

            "Batch normalization normalizes activations across a mini-batch to stabilize "
            "and accelerate training of deep neural networks.",

            "Dropout randomly sets a fraction of neuron activations to zero during training "
            "to prevent overfitting in neural networks.",
        ],
        "correct_context_idx": -1,
        "reference": (
            "Re-ranking reorders retrieved documents using a cross-encoder to surface "
            "the most relevant passages before generation, reducing hallucination."
        ),
        "note": "All retrieved docs are about vision/training — completely off-topic.",
    },

    # ══ CONFLICTING CONTEXT ═══════════════════════════════════════════════════
    {
        "id": "s003",
        "stress_type": "conflicting",
        "query": "Was RAG introduced before or after GPT-3?",
        "contexts": [
            "Retrieval-Augmented Generation (RAG) was introduced by Lewis et al. in 2020, "
            "the same year GPT-3 was released by OpenAI. Both papers appeared in 2020, "
            "with GPT-3 released in May and the RAG paper in May as well.",

            "RAG was a successor to GPT-3, introduced in 2022 after researchers observed "
            "the limitations of purely parametric models. It was designed specifically to "
            "address hallucination problems that appeared in GPT-3 deployments.",

            "OpenAI released GPT-3 in 2020, a large language model with 175 billion parameters "
            "trained on a diverse corpus of internet text.",
        ],
        "correct_context_idx": 0,
        "reference": "RAG and GPT-3 were both introduced in 2020.",
        "note": "Passage 1 directly contradicts passage 0. Model must resolve conflict.",
    },
    {
        "id": "s004",
        "stress_type": "conflicting",
        "query": "What does faithfulness measure in RAG evaluation?",
        "contexts": [
            "Faithfulness measures whether the generated answer is factually consistent "
            "with the retrieved context. An answer is faithful if all its claims can be "
            "verified against the source documents.",

            "Faithfulness in RAG evaluation refers to whether the answer matches the user's "
            "original intent and question, regardless of what the retrieved documents say. "
            "A faithful answer directly addresses what the user asked.",

            "Answer relevancy measures how well the generated answer addresses the original "
            "query, while context recall measures how much of the reference answer is covered.",
        ],
        "correct_context_idx": 0,
        "reference": (
            "Faithfulness measures factual consistency between the generated answer "
            "and the retrieved context."
        ),
        "note": "Passage 1 redefines faithfulness incorrectly.",
    },
    {
        "id": "s005",
        "stress_type": "conflicting",
        "query": "Can hallucination be completely eliminated in RAG systems?",
        "contexts": [
            "Current research suggests hallucination cannot be completely eliminated in RAG. "
            "While retrieval grounding reduces hallucination, issues such as context ignorance, "
            "retrieval errors, and model bias persist.",

            "Recent advances in grounding techniques have effectively eliminated hallucination "
            "in RAG systems. By enforcing strict context attribution, modern RAG pipelines "
            "achieve near-zero hallucination rates in production deployments.",
        ],
        "correct_context_idx": 0,
        "reference": (
            "Hallucination cannot be completely eliminated; retrieval grounding helps "
            "but context ignorance, retrieval errors, and model bias remain."
        ),
        "note": "Passage 1 makes a false strong claim.",
    },

    # ══ MISSING CONTEXT ═══════════════════════════════════════════════════════
    {
        "id": "s006",
        "stress_type": "missing",
        "query": "What is Self-RAG and how does it differ from standard RAG?",
        "contexts": [
            "Standard RAG pipelines always retrieve documents for every query, regardless "
            "of whether retrieval is actually needed. This can introduce noise when the "
            "model's parametric knowledge is sufficient.",

            "Adaptive retrieval methods attempt to decide when retrieval is beneficial. "
            "These approaches condition retrieval on query complexity or model confidence.",

            "Reflection mechanisms allow language models to evaluate and revise their own "
            "outputs. This is related to chain-of-thought and self-consistency techniques.",
        ],
        "correct_context_idx": -1,
        "reference": (
            "Self-RAG retrieves on demand and critiques its own outputs using reflection tokens, "
            "unlike standard RAG which retrieves unconditionally."
        ),
        "note": "Self-RAG specific context is absent. Near-misses may mislead the generator.",
    },
    {
        "id": "s007",
        "stress_type": "missing",
        "query": "What is CRAG and how does it handle low-quality retrievals?",
        "contexts": [
            "RAGAS provides four metrics for evaluating RAG pipelines: faithfulness, "
            "answer relevancy, context precision, and context recall.",

            "Query rewriting is a technique where the original user query is paraphrased "
            "to improve retrieval recall by matching more document vocabulary.",

            "Retrieval quality can be assessed using NDCG, MRR, and Recall@k metrics, "
            "which measure how well the retriever ranks relevant documents.",
        ],
        "correct_context_idx": -1,
        "reference": (
            "CRAG (Corrective RAG) evaluates retrieval quality and falls back to "
            "web search when retrieved documents are deemed low quality."
        ),
        "note": "CRAG context is entirely missing. Tests whether the metric catches fabrication.",
    },

    # ══ NOISY CONTEXT ════════════════════════════════════════════════════════
    {
        "id": "s008",
        "stress_type": "noisy",
        "query": "What is the role of the retriever in a RAG pipeline?",
        "contexts": [
            "Reinforcement learning from human feedback (RLHF) is used to align language "
            "models with human preferences by training a reward model on human comparisons.",

            "Tokenization converts raw text into subword units using algorithms like BPE "
            "or WordPiece before feeding text into a transformer model.",

            "In a RAG pipeline, the retriever selects the most relevant documents from "
            "a large corpus given a user query. Dense retrievers encode queries and documents "
            "into vector embeddings and use approximate nearest neighbour search.",

            "Positional encodings are added to token embeddings in transformers to give "
            "the model information about the order of tokens in the sequence.",

            "Fine-tuning adapts a pretrained language model to a specific downstream task "
            "using a smaller task-specific dataset with supervised learning.",
        ],
        "correct_context_idx": 2,
        "reference": (
            "The retriever selects relevant documents from a corpus using dense or sparse methods. "
            "Dense retrievers use vector embeddings and ANN search."
        ),
        "note": "Correct context is at index 2, buried in 4 irrelevant passages.",
    },
    {
        "id": "s009",
        "stress_type": "noisy",
        "query": "What is chunk size and why does it matter in RAG?",
        "contexts": [
            "Attention mechanisms compute a weighted sum of value vectors, where weights "
            "are determined by the similarity between query and key vectors.",

            "Layer normalization applies normalization across the feature dimension rather "
            "than the batch dimension, making it suitable for sequence models.",

            "Knowledge distillation trains a smaller student model to mimic the output "
            "distribution of a larger teacher model.",

            "Chunk size determines how documents are split before indexing in RAG. "
            "Smaller chunks improve retrieval precision but may lose surrounding context. "
            "Larger chunks preserve context but reduce retrieval relevance by adding noise.",

            "Zero-shot learning enables models to handle tasks or categories not seen "
            "during training by leveraging semantic relationships.",

            "In-context learning allows LLMs to perform tasks by conditioning on a few "
            "examples provided directly in the prompt without parameter updates.",
        ],
        "correct_context_idx": 3,
        "reference": (
            "Chunk size controls how documents are split before indexing. "
            "Smaller chunks improve precision; larger chunks preserve context but add noise."
        ),
        "note": "Correct context is at index 3 among 5 noisy passages.",
    },
    {
        "id": "s010",
        "stress_type": "noisy",
        "query": "What is intrinsic hallucination in RAG?",
        "contexts": [
            "Extrinsic hallucination refers to generated content that cannot be verified "
            "from the source documents — neither supported nor contradicted.",

            "Model calibration refers to how well a model's confidence scores reflect "
            "the true probability of correctness.",

            "Semantic similarity between sentences can be measured using cosine similarity "
            "of their embedding vectors.",

            "Intrinsic hallucination occurs when the generated output directly contradicts "
            "the source documents provided as context. This is distinct from extrinsic "
            "hallucination where the output is merely unverifiable.",

            "Perplexity measures how well a language model predicts a sample of text. "
            "Lower perplexity indicates better predictive performance.",
        ],
        "correct_context_idx": 3,
        "reference": (
            "Intrinsic hallucination occurs when the generated output contradicts "
            "the retrieved source documents."
        ),
        "note": "Correct context at index 3. Passage 0 about extrinsic hallucination may confuse the model.",
    },
]


# ── Stress dataset helpers ────────────────────────────────────────────────────

def get_stress_contexts_as_documents():
    """Returns all stress-test context passages as LlamaIndex Documents."""
    from llama_index.core import Document
    docs = []
    for item in STRESS_DATASET:
        for i, ctx in enumerate(item["contexts"]):
            docs.append(Document(
                text=ctx,
                metadata={
                    "source_id":   item["id"],
                    "stress_type": item["stress_type"],
                    "is_correct":  (i == item["correct_context_idx"]),
                }
            ))
    return docs


def get_stress_queries():
    """Returns (id, query, reference, stress_type) tuples for stress dataset."""
    return [
        (item["id"], item["query"], item["reference"], item["stress_type"])
        for item in STRESS_DATASET
    ]


# ── HotpotQA helpers ──────────────────────────────────────────────────────────

def load_hotpotqa(split: str = "validation", n_samples: int = 50, seed: int = 42):
    """
    Loads HotpotQA distractor split from HuggingFace datasets.
    Shuffles with seed before selecting so the same seed always yields the same samples.
    Returns a list of raw samples (dicts with question, answer, context, id).
    """
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    return list(ds.shuffle(seed=seed).select(range(n_samples)))


def get_hotpotqa_documents(samples: list):
    """
    Converts HotpotQA samples into LlamaIndex Documents.
    Each article (title + sentences) in sample["context"] becomes one Document.
    """
    from llama_index.core import Document
    docs = []
    seen = set()
    for sample in samples:
        titles    = sample["context"]["title"]
        sentences = sample["context"]["sentences"]
        for title, sent_list in zip(titles, sentences):
            text = " ".join(sent_list).strip()
            key  = (title, text)   # full text prevents false-positive dedup on shared prefixes
            if key in seen:
                continue
            seen.add(key)
            docs.append(Document(
                text=text,
                metadata={
                    "title":      title,
                    "source_id":  sample["id"],
                    "dataset":    "hotpotqa",
                }
            ))
    return docs


def get_hotpotqa_queries(samples: list):
    """Returns (id, question, answer, 'hotpotqa') tuples."""
    return [
        (sample["id"], sample["question"], sample["answer"], "hotpotqa")
        for sample in samples
    ]


# ── Backward-compat aliases ───────────────────────────────────────────────────

def get_contexts_as_documents():
    return get_stress_contexts_as_documents()


def get_queries():
    return get_stress_queries()
