.PHONY: help dataset splits dev test bias leakage all up pytest clean

help:
	@echo "dataset  - regenerate data/labels.csv (deterministic)"
	@echo "splits   - regenerate data/splits/{train,dev,test}.csv"
	@echo "dev      - evaluate on the dev split"
	@echo "test     - evaluate on the frozen test split (reproduce the headline numbers)"
	@echo "bias     - run position + verbosity probes -> results/bias.csv"
	@echo "leakage  - run the induced-leakage experiment -> results/induced_failure.json"
	@echo "all      - dataset + splits + dev + test + bias + leakage + pytest"
	@echo "up       - docker compose up (frozen test run)"
	@echo "pytest   - unit tests"

dataset:
	python scripts/generate_dataset.py --output data/labels.csv

splits:
	python src/split_data.py --input data/labels.csv --output-dir data/splits/

dev:
	python evaluate.py --split dev

test:
	python evaluate.py --split test

bias:
	python src/bias_probes.py --output results/bias.csv

leakage:
	python scripts/induce_leakage.py

all: dataset splits dev test bias leakage pytest

up:
	docker compose up --build

pytest:
	python -m pytest -q

clean:
	rm -rf results/predictions_*.csv
