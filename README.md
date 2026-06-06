# fws-bench

Experimental orchestration for the FWS (Fractal Weight Spaces) programme. Houses the z-space vs W-space paired training harness, the regime switcher (joint / alternating / meta), the `optax.multi_transform` G/z partition setup, and the activation + renderer + per-group-optimiser ablation runners.

Not a renderer. Not a library of primitives. fws-bench is the *experimental layer* — it consumes sibling FWS libraries ([`loom`](https://github.com/DarkbyteAT/loom), [`ondes`](https://github.com/DarkbyteAT/ondes), [`jacobian-spec`](https://github.com/DarkbyteAT/jacobian-spec), [`landscape-archaeology`](https://github.com/DarkbyteAT/landscape-archaeology)) and orchestrates the experiments that compare W-direct training to W = render(G, z) reparameterised training under matched compute.

See [`docs/PHILOSOPHY.md`](docs/PHILOSOPHY.md) for the design rationale, the paired-arm contract, and the regime taxonomy.

## What it looks like

```python
import fws_bench

results = fws_bench.paired_train(
    render_fn,
    task_loss_fn,
    regime=fws_bench.Regime.JOINT,
    num_outer_steps=1000,
)
results["fws_arm"]   # curves + diagnostics for W = render(G, z) arm
results["w_arm"]     # curves + diagnostics for direct W arm
```

## Status

**v0.0.0 — scaffold only.** The public surface above is a placeholder that raises `NotImplementedError`. Sibling dependencies (`jacobian-spec`, `landscape-archaeology`, `ondes`, `loom`) are unpublished, so `uv sync` will fail until either the siblings ship or `tool.uv.sources` is wired up locally — see `pyproject.toml` for the co-development pattern.

The first real cell — paired training across a single `(regime, optimiser, mainnet-activation)` point — lands as the Trello board's *First z-space vs W-space paired-training prototype* card.

## Install

Not on PyPI. Once siblings publish, install from git:

```bash
uv pip install git+https://github.com/DarkbyteAT/fws-bench
```

For local co-development against sibling checkouts:

```bash
git clone https://github.com/DarkbyteAT/fws-bench
cd fws-bench
# Edit pyproject.toml to enable the [tool.uv.sources] block pointing at ../loom etc.
uv sync --group dev
```

## License

MIT
