# Task 03 — Sanity Tests

**Read `CONTEXT.md` first. Depends on task-01.**

## Goal

Pytest tests that lock in the metric's expected behavior on hand-crafted inputs. Any future code change that breaks one of these tests is a regression and must be reviewed.

## Why

Without tests, every refactor risks silently changing what the metric measures. The paper's claims rest on the metric's behavior — that behavior must be pinned.

## Steps

1. **Create `tests/` directory and `tests/conftest.py`.**
   - Fixtures: a small `docs` list and a few hand-crafted `answer` strings.

2. **Create `tests/test_metric_sanity.py`** with these tests:

   - **`test_directly_grounded`**: Answer is copied from one of the retrieved passages.
     - Assert `faithfulness > 0.85`.
     - Assert `claim_coverage > 0.85`.
     - Assert `contradiction < 0.1`.

   - **`test_off_topic`**: Answer is "The sky is blue. Grass is green." with docs about RAG.
     - Assert `faithfulness < 0.2`.
     - Assert `claim_coverage < 0.2`.

   - **`test_partial_grounding`**: 1 grounded sentence + 5 fabricated sentences against fixed context.
     - Assert `claim_coverage < 0.3`.
     - Assert composite H_score (after adding the answer_relevance δ component) is < 0.65 — i.e. below threshold.

   - **`test_contradicted`**: Answer makes a claim that is directly contradicted by one of the retrieved passages, plus some neutral filler.
     - Assert `contradiction > 0.3`.

   - **`test_answer_relevance_off_topic`**: Original query "What is dense retrieval?", answer about CNN image recognition (entailed from a passage but topically wrong).
     - Assert `answer_relevance < 0.5`.
     - (This validates that the δ component catches post-rationalization.)

   - **`test_drift_guard_triggers`**: Construct a refine scenario where the LLM produces a refined query unrelated to the original.
     - This requires mocking or constructing the scenario directly.
     - Either (a) call `_cosine_sim` on a known-drifted query pair and assert < 0.5, or (b) monkeypatch the LLM response to a drifted query and run one full refine cycle, asserting `drift_rejected == True`.

3. **Add `pytest` to `req.txt`.**

4. **Add `make test`** to the Makefile from task-02.

## Done when

- [ ] `make test` runs and all 6 tests pass.
- [ ] Test file is under 200 lines.
- [ ] Tests do not require Groq API access (use small fixed answers, not generated ones).

## Do not

- Do not assert exact numeric values (NLI scores vary slightly across hardware). Use inequalities.
- Do not use LLM calls in tests. Generation is non-deterministic and tests would flake.
- Do not test the LangGraph wiring — test the metric functions directly.
