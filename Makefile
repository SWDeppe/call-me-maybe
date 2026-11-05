.PHONY: install run debug clean lint lint-strict

install:
	uv sync --group dev

run:
	uv run call-me-maybe

debug:
	uv run python -c 'import pdb; import call_me_maybe; pdb.run("call_me_maybe.main()")'

clean:
	find . -type d \( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache \) -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	 rm -rf .coverage build dist

lint:
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
