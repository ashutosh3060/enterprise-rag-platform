.PHONY: install dev test lint fmt typecheck run clean

install:
	python -m pip install -e ".[dev]"

dev: install
	cp -n .env.example .env || true
	@echo "Edit .env, then run: make run"

test:
	pytest -q

lint:
	ruff check .

fmt:
	ruff format .

typecheck:
	mypy src

run:
	@echo "See README section 'Quickstart' for this project's run command."

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
