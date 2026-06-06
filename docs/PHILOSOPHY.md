# fws-bench — Philosophy

## The question

What is fws-bench for?

## The verdict

> **fws-bench is the experimental orchestration layer for the FWS programme.**

It runs the harness. It owns the *paired-arm contract* (FWS-arm vs W-arm trained on the same task under matched compute), the *regime taxonomy* (joint / alternating / meta), the `optax.multi_transform` G/z partition setup, and the ablation runners across activation choice, renderer choice, and per-group optimiser. Nothing else.

Everything upstream of orchestration — INR primitives, the re-parameterisation substrate, spectral diagnostics, loss-landscape probes — lives in sibling libraries. fws-bench *composes* them; it does not re-implement them.

## The dependency stack

```
fws-bench  (this repo)
  ├── loom                  — render(P, f, params) substrate
  ├── ondes                 — INR primitives (SIREN, WIRE, Fourier encodings)
  ├── jacobian-spec         — Jacobian-based spectral diagnostics
  └── landscape-archaeology — loss-landscape probes (sharpness, basin shape)
```

Each sibling owns one thing. fws-bench owns *the experiment that uses all four*.

## The paired-arm contract

The FWS programme's core empirical question is:

> Does training `z` through `W = render(G, z)` (FWS-arm) reach the same task loss as training `W` directly (W-arm) under matched compute?

Every cell run by fws-bench answers this question for one point in the design space. The contract:

1. **Same task.** Both arms see the same data, the same task loss, the same evaluation protocol.
2. **Same compute budget.** Outer-step count, batch size, and gradient accumulation are matched. Wall-clock differences from the renderer's forward pass are reported, not hidden.
3. **Same initialisation distribution for W.** W-arm starts from `W ~ p_W`. FWS-arm starts from `(G_0, z_0)` such that `render(G_0, z_0) ~ p_W` in distribution. This is a non-trivial calibration step and is part of the harness.
4. **Same diagnostics.** Both arms feed the same Jacobian-spec and landscape-archaeology probes at the same checkpoints.

A result is reported as a pair: `(loss_fws_arm, loss_w_arm)`. A single arm without its match is not a result.

## The regime taxonomy

The FWS-arm has three flavours of optimisation regime, encoded as `Regime.JOINT`, `Regime.ALTERNATING`, `Regime.META`:

### JOINT

```
for t in range(num_outer_steps):
    grad_G, grad_z = ∇(G, z) L(render(G, z), batch_t)
    G ← opt_G.step(G, grad_G)
    z ← opt_z.step(z, grad_z)
```

Single forward pass, both groups updated per outer step. `optax.multi_transform` partitions the optimiser state between G and z so the two groups can carry different learning rates, schedules, and momentum.

### ALTERNATING

```
for outer in range(num_outer_steps // (N_G + N_z)):
    for _ in range(N_z):                       # fix G, train z
        z ← opt_z.step(z, ∇_z L(render(G, z), batch))
    for _ in range(N_G):                       # fix z, train G
        G ← opt_G.step(G, ∇_G L(render(G, z), batch))
```

Block-coordinate descent over the two groups. Useful when the two groups have very different conditioning or when the renderer's Jacobian w.r.t. G is much more expensive than w.r.t. z.

### META

```
for outer in range(num_outer_steps):
    # Inner loop: adapt z from a task-conditional prior
    z_T = inner_adapt(G, z_prior, batch, K)
    grad_G = ∇_G L(render(G, z_T), val_batch)
    G ← opt_G.step(G, grad_G)
```

Outer loop optimises G; inner loop adapts z per task batch from a prior. This is the iMAML / Reptile flavour — G becomes a *meta-prior over weight spaces*, z is the per-task adaptation. The inner-loop scan composes with `loom.render` per the recipe in `loom/docs/PHILOSOPHY.md`.

## What fws-bench owns

- The `Regime` enum and its dispatch
- `paired_train(...)` — the harness
- The `optax.multi_transform` setup for the G/z partition
- The matched-compute accounting
- The initialisation calibration (FWS-arm's `(G_0, z_0)` to match W-arm's `W ~ p_W`)
- The activation / renderer / per-group-optimiser ablation runners
- The per-checkpoint diagnostic invocation (calling out to jacobian-spec and landscape-archaeology)
- Run-artefact persistence (the curves + diagnostics dict that `paired_train` returns)

## What fws-bench does NOT own

- The render substrate — `loom`
- INR bodies, Fourier encodings, coord grids — `ondes`
- Jacobian / spectral diagnostics — `jacobian-spec`
- Loss-landscape probes — `landscape-archaeology`
- Concrete task instances (CIFAR-10, ImageNet patches, etc.) — `examples/` or downstream
- Plotting and figure generation — downstream notebooks
- Sweep registries beyond the built-in ablation axes — downstream scripts
- Experiment tracking — `xptrack`

## Non-goals (explicit)

- **Owning the renderer.** `loom` does. fws-bench takes a `render_fn` callable.
- **Owning the INR.** `ondes` does. fws-bench accepts whatever `render_fn` returns.
- **Owning the diagnostic.** `jacobian-spec` and `landscape-archaeology` do.
- **Re-implementing optax.** The G/z partition is one call to `optax.multi_transform`; fws-bench wires it, not re-invents it.
- **Owning task data.** Datasets live in `examples/` or downstream. The harness sees a `task_loss_fn`.

## Status

v0.0.0 is a scaffold. The public surface raises `NotImplementedError`. The first real cell — paired training across a single `(regime, optimiser, mainnet-activation)` point — lands as the Trello board's *First z-space vs W-space paired-training prototype* card.
