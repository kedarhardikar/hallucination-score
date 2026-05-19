.PHONY: install eval-stress eval-hotpot eval-hotpot-large ablation reset-db clean

install:
	pip install -r req-pinned.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

eval-stress:
	python evaluate.py stress

eval-hotpot:
	python evaluate.py hotpotqa

eval-hotpot-large:
	python evaluate.py hotpotqa --n-samples 200

ablation:
	python ablation.py all

reset-db:
	python db.py reset stress && python db.py reset hotpotqa

clean:
	rm -rf eval_results/ ablation_results/
