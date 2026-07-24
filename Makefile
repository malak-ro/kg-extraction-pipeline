.PHONY: install test lint format run

install:
	pip install -e .
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest --cov=src --cov=config

lint:
	ruff check .
	mypy src config

format:
	black .
	ruff check --fix .

run:
	python -m src.main
