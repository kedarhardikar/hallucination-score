# RAG Hallucination Detection — Project Overview

## What This Project Does

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** with a custom hallucination detection metric called **H_score**. The pipeline retrieves relevant documents from a vector database, generates an answer using the Groq LLM, scores the answer for hallucination, and — if the score is too low — automatically rewrites the query and tries again.

The goal is to measure and reduce hallucination in RAG systems across different failure scenarios, and to validate H_score as a better alternative to existing metrics like RAGAS.

---

## System Architecture

```
User Query
    │
    ▼
┌─────────────┐
│   Retrieve  │  ← Searches ChromaDB for top-5 relevant passages
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Generate  │  ← Groq LLM (llama-3.3-70b) answers using ONLY retrieved passages
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│  Hallucination Metric │  ← Computes H_score using local NLI model
└──────────┬───────────┘
           │
    ┌──────▼──────┐
    │ H_score ≥   │ YES ──→ Finalize (return best answer)
    │  0.65 or    │
    │ retries ≥ 3 │ NO  ──→ Refine Query (Groq rewrites the query)
    └─────────────┘              │
                                 └──→ back to Retrieve
```

**Every node in the graph is a LangGraph node. The state (`RAGState`) is passed between nodes and updated at each step.**

---

## H_score Formula

```
H_score = α · Faithfulness + β · ClaimCoverage + γ · (1 − ContradictionRate) + δ · AnswerRelevance
        = 0.30 · Faithfulness + 0.30 · ClaimCoverage + 0.15 · (1 − ContradictionRate) + 0.25 · AnswerRelevance
```

For each sentence in the LLM's answer, the pipeline runs NLI against **each retrieved passage individually** and picks the **best-matching passage** (highest entailment score).

| Component | Weight | What it measures | How it is computed |
|---|---|---|---|
| **Faithfulness** | α = 0.30 | Grounding *strength* — how confidently are grounded claims supported | Avg entailment score of covered sentences only |
| **ClaimCoverage** | β = 0.30 | Grounding *breadth* — what fraction of claims are grounded at all | Fraction of sentences where best entailment > 0.5 |
| **Contradiction** | γ = 0.15 | Conflict detection — does the best-matching passage also contradict the sentence | Fraction of sentences where best-matching passage contradiction > 0.5 |
| **AnswerRelevance** | δ = 0.25 | Topic alignment — does the answer actually address the original question | Cosine similarity between original query embedding and answer embedding |

**Key design choices:**

1. **Contradiction uses the same best-matching passage** — not the worst passage across all docs. This prevents false positives in noisy retrieval: an irrelevant passage accidentally scoring high on contradiction should not penalise a well-supported sentence.

2. **AnswerRelevance always compares against `original_query`** — never the refined query. This ensures that topic drift introduced by query refinement is always detected and penalised.

3. **Semantic drift guard in query refinement** — before accepting a rewritten query, the pipeline computes cosine similarity between the original query and the refined query. If the similarity falls below `DRIFT_CUTOFF = 0.5`, the refinement is rejected and the pipeline immediately finalises on the best answer seen so far. This prevents the LLM from drifting to a different topic just because related context happens to be available.

---

## Models Used

| Role | Model | Where it runs |
|---|---|---|
| LLM (generation + query refinement) | `llama-3.3-70b-versatile` via Groq API | Remote (Groq) |
| Embeddings (document + query encoding) | `BAAI/bge-small-en-v1.5` | Local (HuggingFace) |
| NLI (H_score computation) | `cross-encoder/nli-deberta-v3-small` | Local (HuggingFace) |

**No OpenAI. Only Groq API key is needed.**

---

## File Structure

```
hallucination/
├── main.py          ← RAG pipeline, H_score metric, LangGraph graph
├── dataset.py       ← Stress dataset + HotpotQA data loaders
├── evaluate.py      ← Batch evaluation harness, exports CSV/JSON/LaTeX
├── ablation.py      ← Ablation studies (components, weights, refinement)
├── db.py            ← ChromaDB persistence layer
├── req.txt          ← Dependencies
├── .env             ← GROQ_API_KEY=your_key (not committed)
├── chroma_db/       ← Created on first run, persisted vector database
│   ├── chroma.sqlite3
│   ├── stress_queries.json
│   └── hotpotqa_queries.json
├── eval_results/    ← Created on first run, evaluation outputs
│   ├── results_<RUN_ID>.csv
│   ├── summary_<RUN_ID>.json
│   └── paper_table_<RUN_ID>.tex
└── ablation_results/← Created on first ablation run
    ├── component_ablation_<RUN_ID>.json
    ├── weight_sensitivity_<RUN_ID>.json
    └── refinement_ablation_<RUN_ID>.json
```

---

## Datasets

### 1. Stress Dataset (`dataset.py` → `STRESS_DATASET`)
A hand-crafted adversarial dataset of 10 queries designed to test specific hallucination failure modes.

| Stress Type | What it tests | Items |
|---|---|---|
| `irrelevant` | No relevant context exists — model must hallucinate or refuse | s001, s002 |
| `conflicting` | Two contradictory passages exist — model must resolve conflict | s003, s004, s005 |
| `missing` | Correct context is absent, near-misses may mislead | s006, s007 |
| `noisy` | Correct passage is buried among many irrelevant ones | s008, s009, s010 |

**Important — source-isolated retrieval for stress queries:**
All stress contexts are indexed into a single ChromaDB collection, but each query's retrieval is filtered to its own `source_id`. This prevents cross-contamination: without filtering, a query designed to have *no relevant context* (e.g., s001) would accidentally retrieve relevant passages from other items (e.g., s008's dense retrieval context), destroying the stress effect. Each stress query therefore only searches its own designated passages — exactly the set it was designed to test against.

HotpotQA queries do **not** use this filter — multi-hop retrieval requires searching the full document pool.

### 2. HotpotQA (`dataset.py` → `load_hotpotqa`)
A real-world multi-hop QA benchmark from HuggingFace. Each question requires reasoning across **2 gold Wikipedia articles** to answer, with **8 distractor articles** mixed in to challenge the retriever.

- Default: 50 validation samples → ~500 documents indexed
- Each question tests multi-hop retrieval — neither gold article alone is sufficient

---

## Running `evaluate.py`

### Basic usage

```bash
python evaluate.py [mode] [--no-refine]
```

### Arguments

#### `mode` (positional, default: `stress`)

Controls which dataset is used for evaluation.

| Value | Dataset used | ChromaDB collection | When to use |
|---|---|---|---|
| `stress` | Hand-crafted adversarial dataset (10 queries) | `stress` | Fast testing, specific failure mode analysis |
| `hotpotqa` | HotpotQA distractor split (50 queries, ~500 docs) | `hotpotqa` | Realistic multi-hop evaluation |
| `both` | Stress + HotpotQA combined | `both` | Full evaluation for paper results |

**Why it matters:** Each mode indexes a different set of documents into ChromaDB. Running `stress` and `hotpotqa` separately keeps the collections isolated — the stress retriever only searches stress passages, the HotpotQA retriever only searches Wikipedia articles. Mixing them (via `both`) creates a larger, harder retrieval task.

#### `--no-refine` (flag, optional)

Disables the query refinement loop. The pipeline runs a **single pass only** — retrieve → generate → score → finalize, with no retries regardless of H_score.

**Why it matters:** This is the **ablation baseline**. Running the same queries with and without `--no-refine` lets you measure exactly how much the refinement loop improves H_score and ROUGE-L. Without this comparison, the refinement contribution cannot be claimed in the paper.

### Examples

```bash
# Standard evaluation on stress dataset (with refinement)
python evaluate.py stress

# Standard evaluation on HotpotQA (with refinement)
python evaluate.py hotpotqa

# Full evaluation combining both datasets
python evaluate.py both

# Ablation: stress dataset WITHOUT query refinement
python evaluate.py stress --no-refine

# Ablation: HotpotQA WITHOUT query refinement
python evaluate.py hotpotqa --no-refine
```

### First run vs subsequent runs

| Condition | What happens |
|---|---|
| **First run** (no ChromaDB) | Downloads data (HotpotQA only) → embeds documents → saves to `chroma_db/` → runs evaluation |
| **Subsequent runs** | Loads index directly from `chroma_db/` → loads queries from `.json` cache → skips all embedding — much faster |

To force a rebuild (e.g. after changing the dataset):
```bash
python db.py reset stress
python db.py reset hotpotqa
```

---

## Running `ablation.py`

```bash
python ablation.py [experiment]
```

| Value | What it runs |
|---|---|
| `component` | Removes each H_score component one at a time — proves each of the four components contributes |
| `weights` | Grid search over α+β+γ+δ=1 combinations — justifies the chosen weight distribution |
| `refinement` | Runs with and without query refinement — proves the loop improves results |
| `all` (default) | Runs all three experiments sequentially |

```bash
python ablation.py all
```

---

## Running `db.py` (database inspection)

```bash
# Inspect what is stored in a collection
python db.py inspect stress
python db.py inspect hotpotqa

# Delete a collection (forces rebuild on next evaluate run)
python db.py reset stress
```

You can also open `chroma_db/chroma.sqlite3` directly in **DB Browser for SQLite** (free download: https://sqlitebrowser.org/) to browse the raw embeddings, metadata, and document text.

---

## Output Files

Every `evaluate.py` run creates three files in `eval_results/` stamped with a timestamp (`RUN_ID`):

| File | Contents | Use in paper |
|---|---|---|
| `results_<RUN_ID>.csv` | One row per query: H_score, best_H_score, faithfulness, claim_coverage, contradiction, ROUGE-L, retries, latency, accepted | Raw data for results table |
| `summary_<RUN_ID>.json` | Aggregated stats grouped by stress type + overall | Section 4 summary statistics |
| `paper_table_<RUN_ID>.tex` | Ready-to-paste LaTeX table | Direct paste into paper |

### Key CSV columns explained

| Column | Meaning |
|---|---|
| `h_score` | H_score of the **last** iteration |
| `best_h_score` | H_score of the **best** iteration across all retries — use this for paper results |
| `faithfulness` | Avg entailment of covered sentences (last iteration) |
| `claim_coverage` | Fraction of sentences grounded (last iteration) |
| `contradiction` | Fraction of sentences contradicted by best-matching passage (last iteration) |
| `retries` | How many query refinement cycles fired (0 = accepted on first pass) |
| `rouge_l` | ROUGE-L F1 against reference answer — secondary factual accuracy metric |
| `accepted` | `True` if `best_h_score ≥ 0.65` |

---

## Setup

### 1. Install dependencies
```bash
pip install -r req.txt
```

### 2. Create `.env` file
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at https://console.groq.com

### 3. Run
```bash
python evaluate.py stress
```

The first run will download the NLI and embedding models (~500MB total, one-time).

---

## Environment

- Python 3.10+
- GROQ_API_KEY in `.env`
- GPU optional (NLI and embedding models run on CPU by default; set `device=0` in `main.py` to use GPU)
