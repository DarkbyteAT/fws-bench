# Contributing to fws-bench

## Development Setup

```bash
git clone https://github.com/DarkbyteAT/fws-bench.git
cd fws-bench
uv sync --group dev
```

The sibling dependencies (`jacobian-spec`, `landscape-archaeology`, `ondes`, `loom`) are not yet on PyPI. For local co-development, enable the `[tool.uv.sources]` block in `pyproject.toml` pointing at sibling checkouts under `../`.

## Code Conventions

- **Python 3.11+** — `X | Y` union syntax, `list[T]` / `dict[K, V]` generics
- **Google-style docstrings** with LaTeX math support (`$...$` inline, `$$...$$` display)

## Quality Gates

```bash
make all    # format-check + lint + typecheck + test
make fix    # auto-fix lint violations
```

Tool configs live in dedicated files (`ruff.toml`, `pytest.ini`, `pyrightconfig.json`).

## Testing

- Plain `def test_*` functions — no classes
- Given-When-Then structure
- `tests/` mirrors `src/fws_bench/` layout
