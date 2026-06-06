.PHONY: lint format format-check fix typecheck test all

lint:
	uv run ruff check src/fws_bench/

format:
	uv run ruff format src/fws_bench/

format-check:
	uv run ruff format --check src/fws_bench/

fix:
	uv run ruff check --fix src/fws_bench/

typecheck:
	uv run pyright src/fws_bench/

test:
	uv run pytest tests/ -v || [ $$? -eq 5 ]

all: format-check lint typecheck test
