.PHONY: venv lint clean

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -e . && \
	pip install -r requirements-dev.txt

lint:
	ruff check .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
