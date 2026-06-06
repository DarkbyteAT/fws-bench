# AGENTS.md

## Commands

```bash
uv sync --group dev
uv run ruff check src/fws_bench/
uv run pyright src/fws_bench/
uv run pytest tests/
make all
```

## Critical Rules

- Python 3.11+
- Plain `def test_*` functions, Given-When-Then structure
- House style: Google-style docstrings, `X | Y` union syntax, `list[T]` / `dict[K, V]` generics
- Tool configs live in dedicated files (`ruff.toml`, `pytest.ini`, `pyrightconfig.json`), not `pyproject.toml`
- Library scope is **machinery**, not instantiations — concrete datasets, task configs, and runner scripts go in `examples/` or downstream, not into `src/fws_bench/`
