# Task 02 — Engineering Polish

**Read `CONTEXT.md` first. Depends on task-01.**

## Goal

Make the project reproducible and easier to run at scale.

## Why

Reviewers will ask "how do I reproduce this?" — the answer needs to be one command. Logging is needed because the per-sentence NLI traces are unreadable on a 200-query run.

## Steps

1. **Replace `print(...)` with `logging`** in `main.py`, `evaluate.py`, `ablation.py`.
   - `logging.INFO` for progress (per-query lines, summary outputs).
   - `logging.DEBUG` for per-sentence NLI traces in `compute_h_score`.
   - Configure at module top: `logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))`.

2. **Add `--verbose` / `--quiet` to `evaluate.py` and `ablation.py`.**
   - `--verbose` → `LOG_LEVEL=DEBUG`.
   - `--quiet` → `LOG_LEVEL=WARNING`.

3. **Save run config to outputs.**
   - Each `evaluate.py` and `ablation.py` run emits `config_<RUN_ID>.json` next to the results.
   - Contents: weights (α, β, γ, δ), `THRESHOLD`, `MAX_RETRIES`, `DRIFT_CUTOFF`, NLI model name, embedding model name, Groq model name, dataset/mode, n_samples, no_refine flag, random_seed, git commit hash (use `subprocess.check_output(["git", "rev-parse", "HEAD"])` — handle the case where git is not available).

4. **Pin dependencies.**
   - Run `pip freeze > req-pinned.txt`.
   - Keep `req.txt` with `>=` for installation flexibility, but check in `req-pinned.txt` as the exact reproducible set.

5. **Seed HotpotQA sample selection.**
   - In `dataset.py` `load_hotpotqa`, add a `seed: int = 42` argument.
   - Use `ds.shuffle(seed=seed).select(range(n_samples))` instead of `ds.select(range(n_samples))`.
   - Plumb `seed` through `evaluate.py` and `ablation.py` so it's settable from CLI.
   - Log the seed in the run config.

6. **Add a `Makefile`** with these targets:
   - `make install` → `pip install -r req-pinned.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"`
   - `make eval-stress` → `python evaluate.py stress`
   - `make eval-hotpot` → `python evaluate.py hotpotqa`
   - `make eval-hotpot-large` → `python evaluate.py hotpotqa --n-samples 200`
   - `make ablation` → `python ablation.py all`
   - `make reset-db` → `python db.py reset stress && python db.py reset hotpotqa`
   - `make clean` → `rm -rf eval_results/ ablation_results/`

## Done when

- [ ] `python evaluate.py stress --quiet` runs without per-sentence NLI traces.
- [ ] `python evaluate.py stress --verbose` shows them.
- [ ] `eval_results/config_*.json` exists after a run and contains every field listed above.
- [ ] `req-pinned.txt` exists.
- [ ] Running `python evaluate.py hotpotqa --seed 42` twice produces the same sample IDs.
- [ ] `make eval-stress` works.

## Do not

- Do not change pipeline behavior or metric values.
- Do not refactor the LangGraph structure.
