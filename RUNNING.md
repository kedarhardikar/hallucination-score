# Running the Pipeline — Arguments, Outputs, and How to Read Results

## Quick Start

```bash
# Install dependencies (one time)
pip install -r req.txt

# Add your Groq API key
echo GROQ_API_KEY=your_key_here > .env

# Run on stress dataset
python evaluate.py stress
```

First run downloads the NLI model and embedding model (~500MB total, one-time). Subsequent runs load from cache and are much faster.

---

## evaluate.py

The main evaluation script. Runs queries through the full RAG pipeline and saves results.

```bash
python evaluate.py [mode] [--no-refine]
```

### Argument: `mode`

Controls which dataset is used. Default is `stress`.

---

#### `stress`
```bash
python evaluate.py stress
```

**What it does:** Runs the 10 hand-crafted adversarial queries against their designated context sets.

**Why use it:** Fast (10 queries), no internet download required, tests specific failure modes. Good for verifying the pipeline works and the metric behaves correctly across different hallucination types.

**What to expect:**
- `irrelevant` queries (s001, s002): Low H_score, low ROUGE-L. The model has no relevant context and must hallucinate or refuse.
- `conflicting` queries (s003–s005): Medium H_score, high contradiction component. The model sees contradictory passages.
- `missing` queries (s006, s007): Low H_score. The specific fact asked about is absent.
- `noisy` queries (s008–s010): Higher H_score. The correct answer exists but is buried.

**ChromaDB collection used:** `stress`

---

#### `hotpotqa`
```bash
python evaluate.py hotpotqa
```

**What it does:** Downloads 50 validation questions from the HotpotQA dataset, indexes ~500 Wikipedia articles, runs full evaluation.

**Why use it:** Real-world multi-hop questions where the answer requires reasoning across two Wikipedia articles, with 8 distractor articles mixed in. This is the realistic benchmark for the paper — not hand-crafted, not cherry-picked.

**First run:** Downloads from HuggingFace (~50MB), embeds all documents into ChromaDB, caches queries to `chroma_db/hotpotqa_queries.json`. Takes several minutes.

**Subsequent runs:** Loads index and queries from cache. Fast.

**ChromaDB collection used:** `hotpotqa`

---

#### `both`
```bash
python evaluate.py both
```

**What it does:** Combines the stress dataset contexts and HotpotQA documents into a single larger collection, runs all queries together.

**Why use it:** Full evaluation for the paper. Shows the metric works across both adversarial and real-world distributions in the same run.

**ChromaDB collection used:** `both`

---

### Argument: `--no-refine`

```bash
python evaluate.py stress --no-refine
python evaluate.py hotpotqa --no-refine
```

**What it does:** Disables the query refinement loop. The pipeline runs exactly once per query — retrieve → generate → score → finalize. No retries, no query rewriting.

**Why it matters for the paper:** This is the ablation baseline. Running the same queries with and without this flag gives you the numbers for the refinement contribution claim:

```
WITH refinement:    avg_best_h_score = X,  avg_rouge_l = Y
WITHOUT refinement: avg_best_h_score = X', avg_rouge_l = Y'
```

The difference (X − X') is what you report as the refinement loop's contribution. Without this comparison, the claim that "query refinement improves answer quality" cannot be made.

---

### All examples

```bash
# Standard stress evaluation (with refinement)
python evaluate.py stress

# Standard HotpotQA evaluation (with refinement)
python evaluate.py hotpotqa

# Full combined evaluation
python evaluate.py both

# Ablation: no refinement on stress
python evaluate.py stress --no-refine

# Ablation: no refinement on HotpotQA
python evaluate.py hotpotqa --no-refine
```

---

## ablation.py

Runs structured experiments to justify the H_score design choices. Used for paper Section 4 (ablation studies).

```bash
python ablation.py [experiment]
```

### Argument: `experiment`

---

#### `component`
```bash
python ablation.py component
```

**What it does:** Runs the full stress dataset 8 times, each time with a different weight configuration:
- Full model (all four components at default weights)
- No faithfulness (α=0)
- No claim coverage (β=0)
- No contradiction (γ=0)
- No answer relevance (δ=0)
- Faithfulness only
- Coverage only
- Relevance only

**Why it matters:** Proves that each component of H_score contributes independently. If removing any one component causes avg_rouge_l to drop, that component is doing real work. This is the standard ablation study for any proposed metric.

**Output:** `ablation_results/component_ablation_<RUN_ID>.json`

---

#### `weights`
```bash
python ablation.py weights
```

**What it does:** Grid search over all combinations of α, β, γ, δ that sum to 1.0 (at steps of 0.25). Runs the full stress dataset for each combination and records avg_best_h_score, avg_rouge_l, and acceptance_rate.

**Why it matters:** Justifies the chosen weights (0.30, 0.30, 0.15, 0.25). You can show that these weights sit at or near the Pareto-optimal point on the H_score vs ROUGE-L correlation. Without this, a reviewer can ask "why those weights and not others?"

**Output:** `ablation_results/weight_sensitivity_<RUN_ID>.json`

---

#### `refinement`
```bash
python ablation.py refinement
```

**What it does:** Runs the stress dataset twice at default weights — once with refinement enabled, once without. Identical to running `evaluate.py stress` and `evaluate.py stress --no-refine` back to back, but keeps the comparison in a single output file.

**Why it matters:** Isolates the contribution of the query refinement loop and the drift guard. The by_stress_type breakdown shows whether refinement helps more on some failure types than others (expected: helps on `noisy`, less on `irrelevant` and `missing`).

**Output:** `ablation_results/refinement_ablation_<RUN_ID>.json`

---

#### `all` (default)
```bash
python ablation.py all
python ablation.py         # same thing
```

Runs all three experiments sequentially. Takes the longest but produces everything needed for the ablation section of the paper.

---

## db.py

Database inspection and management. Use this to check what is stored, or to force a rebuild.

```bash
python db.py inspect stress       # print all stored chunks for the stress collection
python db.py inspect hotpotqa     # print all stored chunks for the HotpotQA collection
python db.py reset stress         # delete the stress collection (forces rebuild on next run)
python db.py reset hotpotqa       # delete the HotpotQA collection
```

You can also open `chroma_db/chroma.sqlite3` in DB Browser for SQLite (https://sqlitebrowser.org) to browse raw embeddings, metadata, and document text.

---

## Output Files Explained

### eval_results/results_\<RUN_ID\>.csv

One row per query. This is the raw data for your paper's results table.

| Column | What it means | Use in paper |
|---|---|---|
| `id` | Query ID (s001–s010 for stress, HotpotQA hash for hotpotqa) | Row identifier |
| `risk` | Stress type or "hotpotqa" | Group rows by this |
| `h_score` | H_score from the **last** iteration | Do not use this for paper results |
| `best_h_score` | H_score from the **best** iteration across all retries | **Use this for paper results** |
| `faithfulness` | Avg entailment score of covered sentences (last iteration) | Component analysis |
| `claim_coverage` | Fraction of answer sentences that are grounded (last iteration) | Component analysis |
| `contradiction` | Fraction of sentences contradicted by their best-matching passage (last iteration) | Component analysis |
| `answer_relevance` | Cosine similarity between original query and answer (last iteration) | Component analysis |
| `retries` | How many query refinement cycles fired (0 = accepted on first pass) | Refinement contribution |
| `rouge_l` | ROUGE-L F1 against the reference answer | Secondary factual accuracy metric |
| `latency_s` | Total wall-clock time for this query in seconds | Efficiency analysis |
| `accepted` | True if best_h_score ≥ 0.65 | Acceptance rate |
| `query` | The original query text | Qualitative analysis |
| `answer` | The final answer returned by the pipeline | Qualitative analysis |
| `reference` | The ground-truth reference answer | Qualitative analysis |

**Why `h_score` and `best_h_score` are different:**

The pipeline may run up to 3 iterations per query. `h_score` is whatever score the last iteration produced — which may be lower than an earlier iteration if the refined query performed worse. `best_h_score` is the highest score seen across all iterations and is always the score of the answer returned in `final_answer`. Always use `best_h_score` for reporting.

---

### eval_results/summary_\<RUN_ID\>.json

Aggregated statistics grouped by stress type plus an overall summary. Paste the `overall` block directly into your paper.

```json
{
  "overall": {
    "avg_best_h_score":     0.71,    ← main metric for paper
    "avg_h_score":          0.68,    ← last-iteration average (lower bound)
    "avg_answer_relevance": 0.62,    ← topic alignment across all queries
    "avg_rouge_l":          0.41,    ← factual accuracy proxy
    "avg_retries":          1.2,     ← how often refinement was needed
    "acceptance_rate":      0.70,    ← fraction of queries that passed threshold
    "avg_latency_s":        12.4     ← average seconds per query
  },
  "by_risk": {
    "irrelevant": { "avg_best_h_score": ..., "avg_rouge_l": ..., "acceptance_rate": ... },
    "conflicting": { ... },
    "missing":     { ... },
    "noisy":       { ... }
  }
}
```

**What to look for:**
- `avg_best_h_score` should be clearly higher than `avg_h_score` — shows refinement improved results
- `irrelevant` and `missing` should have the lowest acceptance rates — shows the metric correctly rejects hallucinated answers
- `noisy` should have higher acceptance rates than `irrelevant` — shows the metric correctly rewards grounded answers even in noisy retrieval

---

### eval_results/paper_table_\<RUN_ID\>.tex

A complete LaTeX table ready to paste into your paper. Opens with `\begin{table}`, closes with `\end{table}`, includes per-query rows and per-stress-type averages at the bottom.

Columns: ID, Risk, Best H_score, Faithfulness, Coverage, Contradiction, Answer Relevance, ROUGE-L, Retries.

---

### ablation_results/component_ablation_\<RUN_ID\>.json

For each of the 8 weight configurations, records avg_best_h_score, avg_rouge_l, acceptance_rate, and by_stress_type breakdown.

**What to look for:** The "full" configuration should have the highest avg_rouge_l. Any single-component-only configuration should perform worse. This is your evidence that all four components are necessary.

---

### ablation_results/weight_sensitivity_\<RUN_ID\>.json

One record per weight combination (α, β, γ, δ). Each record has avg_best_h_score, avg_rouge_l, acceptance_rate.

**What to look for:** Sort by avg_rouge_l. The default weights (0.30, 0.30, 0.15, 0.25) should appear near the top. If they don't, update the weights to whatever the grid search finds optimal.

---

### ablation_results/refinement_ablation_\<RUN_ID\>.json

Two keys: `with_refinement` and `without_refinement`. Each has avg_best_h_score, avg_rouge_l, avg_retries, acceptance_rate, and by_stress_type.

**What to look for:**
- `with_refinement.avg_best_h_score` > `without_refinement.avg_best_h_score` — refinement helps overall
- `with_refinement.avg_rouge_l` > `without_refinement.avg_rouge_l` — refinement produces more accurate answers
- The improvement should be larger for `noisy` type than for `irrelevant` type — consistent with the expectation that refinement can surface buried correct context but cannot create context that does not exist

---

## How to Force a Full Rebuild

If you change the dataset or suspect stale embeddings:

```bash
python db.py reset stress
python db.py reset hotpotqa
python db.py reset both
python evaluate.py stress     # rebuilds from scratch
```

---

## Environment Notes

- Python 3.10+
- GROQ_API_KEY in `.env`
- NLI and embedding models run on CPU by default
- To use GPU: set `device=0` in the `_get_nli()` call in `main.py`
- HotpotQA requires internet access on first run only
