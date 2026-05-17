.PHONY: venv lint test clean
 
venv:
	python -m venv .venv

install:
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -e . && \
	pip install -r requirements-dev.txt

lint:
	ruff check .
	vulture src tests --min-confidence 80

test:
	coverage run -m unittest discover -s tests -p "test*.py"
	coverage report

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info .coverage .pytest_cache .mypy_cache .ruff_cache
