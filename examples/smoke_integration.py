"""Phase 1 integration smoke for the FWS programme.

Demonstrates the four-library integration (jax + optax + landscape-archaeology
+ fws-bench) end-to-end on a synthetic task with a placeholder renderer.

This is not yet a real FWS experiment: the renderer is a smooth non-identity
stand-in for the ondes SIREN, and the mainnet is a linear regressor. The point
is to prove the pipes connect — both arms train cleanly, the
``singular_spectrum`` probe binds against ``render_fn`` without library
plumbing, and a research-log markdown comes out the other side.

Run:
    uv run python examples/smoke_integration.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float
from landscape_archaeology import singular_spectrum

import fws_bench


RESEARCH_DIR = Path("/Users/ammar/Documents/Research/fractal-weight-spaces/research")
RESEARCH_FILE = RESEARCH_DIR / "2026-06-06-integration-smoke.md"


def make_synthetic_regression(
    *,
    n_in: int = 8,
    n_out: int = 4,
    n_train: int = 100,
    n_test: int = 50,
    seed: int = 0,
):
    """Build a synthetic linear-regression task.

    ``W_target`` is drawn once; ``(x, y)`` pairs satisfy ``y = W_target @ x``
    exactly (noise-free) so MSE has a well-defined zero. Returns the loss
    function plus the train/test splits packaged as ``{"x", "y"}`` dicts.
    """
    key = jax.random.key(seed)
    key_w, key_x_train, key_x_test = jax.random.split(key, 3)
    W_target = jax.random.normal(key_w, (n_out, n_in))
    x_train = jax.random.normal(key_x_train, (n_train, n_in))
    y_train = x_train @ W_target.T
    x_test = jax.random.normal(key_x_test, (n_test, n_in))
    y_test = x_test @ W_target.T

    def task_loss_fn(W: Float[Array, "n_out n_in"], batch: dict) -> Float[Array, ""]:
        preds = batch["x"] @ W.T
        return jnp.mean((preds - batch["y"]) ** 2)

    train_batch = {"x": x_train, "y": y_train}
    test_batch = {"x": x_test, "y": y_test}
    return task_loss_fn, train_batch, test_batch, W_target


def placeholder_render(
    G_params: Float[Array, "32 16"],
    z_params: Float[Array, " 16"],
) -> Float[Array, "4 8"]:
    """Smooth non-identity reparameterisation z -> W.

    Computes ``tanh(G @ z)`` and reshapes the resulting 32-vector into a
    ``(4, 8)`` weight matrix. Not SIREN, not Householder — just a smooth
    nonlinearity that gives the spectrum probe something non-trivial to
    measure without pulling in ondes/loom.
    """
    return jnp.tanh(G_params @ z_params).reshape(4, 8)


def make_inits(
    *,
    n_out: int = 4,
    n_in: int = 8,
    z_dim: int = 16,
    seed: int = 1,
    scale: float = 0.1,
):
    """Initial pytrees for both arms."""
    key = jax.random.key(seed)
    key_g, key_z, key_w = jax.random.split(key, 3)
    init_G = jax.random.normal(key_g, (n_out * n_in, z_dim)) * scale
    init_z = jax.random.normal(key_z, (z_dim,)) * scale
    init_W = jax.random.normal(key_w, (n_out, n_in)) * scale
    return init_G, init_z, init_W


def spectrum_at(
    G_params: Float[Array, "32 16"],
    z_params: Float[Array, " 16"],
    *,
    k: int = 10,
    num_iterations: int = 80,
    seed: int = 7,
) -> Float[Array, " k"]:
    """Top-k singular values of d render_fn / d z at the given (G, z)."""
    operator = lambda z_var: placeholder_render(G_params, z_var)  # noqa: E731
    return singular_spectrum(
        operator,
        z_params,
        k=k,
        num_iterations=num_iterations,
        key=jax.random.key(seed),
    )


def format_trajectory_row(step: int, fws_loss, fws_gn, w_loss, w_gn) -> str:
    return f"| {step:>4d} | {float(fws_loss):.6g} | {float(fws_gn):.6g} | {float(w_loss):.6g} | {float(w_gn):.6g} |"


def main() -> None:
    # --- Setup --------------------------------------------------------------
    task_loss_fn, train_batch, test_batch, W_target = make_synthetic_regression()
    init_G, init_z, init_W = make_inits()
    num_outer_steps = 500

    # --- Train both arms ----------------------------------------------------
    result = fws_bench.paired_train(
        init_G_params=init_G,
        init_z_params=init_z,
        render_fn=placeholder_render,
        init_W_params=init_W,
        task_loss_fn=task_loss_fn,
        task_batch=train_batch,
        regime=fws_bench.Regime.JOINT,
        G_optimiser=optax.adam(1e-3),
        z_optimiser=optax.adam(1e-3),
        W_optimiser=optax.adam(1e-3),
        num_outer_steps=num_outer_steps,
    )

    # --- Measure renderer-Jacobian spectrum at init and final ---------------
    sigmas_init = spectrum_at(init_G, init_z)
    final_G = result.fws_arm.final_params["G"]
    final_z = result.fws_arm.final_params["z"]
    sigmas_final = spectrum_at(final_G, final_z)

    # --- "Did it work?" structural checks -----------------------------------
    fws_loss = result.fws_arm.loss_trajectory
    w_loss = result.w_arm.loss_trajectory
    fws_gn = result.fws_arm.grad_norm_trajectory
    w_gn = result.w_arm.grad_norm_trajectory

    both_finite = bool(
        jnp.all(jnp.isfinite(fws_loss))
        and jnp.all(jnp.isfinite(w_loss))
        and jnp.all(jnp.isfinite(fws_gn))
        and jnp.all(jnp.isfinite(w_gn))
        and jnp.all(jnp.isfinite(sigmas_init))
        and jnp.all(jnp.isfinite(sigmas_final))
    )
    fws_early = float(jnp.mean(fws_loss[:10]))
    fws_late = float(jnp.mean(fws_loss[-10:]))
    w_early = float(jnp.mean(w_loss[:10]))
    w_late = float(jnp.mean(w_loss[-10:]))
    fws_decreased = fws_late < fws_early
    w_decreased = w_late < w_early

    spectrum_shape_ok = sigmas_init.shape == (10,) and sigmas_final.shape == (10,)
    spectrum_descending = bool(
        jnp.all(sigmas_init[:-1] >= sigmas_init[1:]) and jnp.all(sigmas_final[:-1] >= sigmas_final[1:])
    )

    # Did the geometry shift? Compare component-wise; use float eps to qualify
    # "changed" without inventing a magnitude threshold.
    eps = float(jnp.finfo(sigmas_init.dtype).eps) * sigmas_init.shape[0]
    spectra_changed = bool(jnp.any(jnp.abs(sigmas_final - sigmas_init) > eps))

    cond_init = float(sigmas_init[0] / jnp.maximum(sigmas_init[-1], jnp.finfo(sigmas_init.dtype).tiny))
    cond_final = float(sigmas_final[0] / jnp.maximum(sigmas_final[-1], jnp.finfo(sigmas_final.dtype).tiny))

    test_loss_fws = float(task_loss_fn(placeholder_render(final_G, final_z), test_batch))
    test_loss_w = float(task_loss_fn(result.w_arm.final_params, test_batch))

    # --- Stdout summary -----------------------------------------------------
    print("=" * 72)
    print("Phase 1 integration smoke — fws-bench + landscape-archaeology")
    print("=" * 72)
    print(f"W_target shape:                       {tuple(W_target.shape)}")
    print(f"train / test batch sizes:             {train_batch['x'].shape[0]} / {test_batch['x'].shape[0]}")
    print(f"num_outer_steps:                      {num_outer_steps}")
    print(f"FWS arm loss (mean first 10):         {fws_early:.6g}")
    print(f"FWS arm loss (mean last 10):          {fws_late:.6g}")
    print(f"W   arm loss (mean first 10):         {w_early:.6g}")
    print(f"W   arm loss (mean last 10):          {w_late:.6g}")
    print(f"FWS arm test loss:                    {test_loss_fws:.6g}")
    print(f"W   arm test loss:                    {test_loss_w:.6g}")
    print(f"sigmas at z_init:                     {[float(s) for s in sigmas_init]}")
    print(f"sigmas at z_final:                    {[float(s) for s in sigmas_final]}")
    print(f"cond(J) at z_init  = σ_1 / σ_min:     {cond_init:.6g}")
    print(f"cond(J) at z_final = σ_1 / σ_min:     {cond_final:.6g}")
    print()
    print("Did it work?")
    print(f"  - finite trajectories + spectra:    {'yes' if both_finite else 'no'}")
    print(f"  - FWS arm loss decreased:           {'yes' if fws_decreased else 'no'}")
    print(f"  - W   arm loss decreased:           {'yes' if w_decreased else 'no'}")
    print(f"  - spectrum sorted desc, length 10:  {'yes' if (spectrum_shape_ok and spectrum_descending) else 'no'}")
    print(f"  - spectrum changed init -> final:   {'yes' if spectra_changed else 'no'}")

    # --- Write research log -------------------------------------------------
    checkpoint_steps = [0, 100, 250, num_outer_steps - 1]
    trajectory_rows = "\n".join(
        format_trajectory_row(
            s,
            fws_loss[s],
            fws_gn[s],
            w_loss[s],
            w_gn[s],
        )
        for s in checkpoint_steps
    )

    md = f"""# Integration smoke — 2026-06-06

End-to-end verification that `fws-bench.paired_train` + `landscape-archaeology.singular_spectrum` + a placeholder renderer + a synthetic task connect cleanly.

## Setup

- **Task**: synthetic linear regression. `W_target ∈ R^(4×8)`, `{{(x, y)}}` pairs with `x ~ N(0, I_8)` and `y = W_target @ x` (noise-free). 100 train + 50 test points; MSE loss; full-batch (no minibatching) for the smoke
- **Placeholder renderer**: `render_fn(G_params, z) = tanh(G_params @ z).reshape(4, 8)` with `G_params ∈ R^(32×16)`, `z ∈ R^16`. Smooth, non-identity, no library dependencies
- **Mainnet**: linear regressor `y = W @ x`; `W ∈ R^(4×8)` for both arms
- **Training**: `paired_train(Regime.JOINT, 500 outer steps, optax.adam(1e-3))` on all three optimisers
- **Measurement**: top-10 singular values of `∂ render_fn / ∂ z` at `(init_G, init_z)` and `(final_G, final_z)` via `singular_spectrum` (80 power-iteration steps)

## Results

### Training trajectories (sampled checkpoints)

| step | FWS arm loss | FWS arm grad-norm | W arm loss | W arm grad-norm |
|------|--------------|---------------------|------------|--------------------|
{trajectory_rows}

Mean loss over first 10 / last 10 steps:
- FWS arm: {fws_early:.6g} → {fws_late:.6g}
- W   arm: {w_early:.6g} → {w_late:.6g}

Held-out test MSE at end of training:
- FWS arm: {test_loss_fws:.6g}
- W   arm: {test_loss_w:.6g}

### Renderer-Jacobian spectrum

|            | σ_1 | σ_2 | σ_3 | σ_4 | σ_5 | σ_6 | σ_7 | σ_8 | σ_9 | σ_10 |
|------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| at z_init  | {" | ".join(f"{float(s):.4g}" for s in sigmas_init)} |
| at z_final | {" | ".join(f"{float(s):.4g}" for s in sigmas_final)} |

Condition number `cond(J) = σ_1 / σ_min`:
- at z_init: {cond_init:.6g}
- at z_final: {cond_final:.6g}

## Did it work?

- Both arms produced finite-valued trajectories: {"yes" if both_finite else "no"}
- FWS arm loss decreased over training: {"yes" if fws_decreased else "no"} (mean first 10 = {fws_early:.6g}, mean last 10 = {fws_late:.6g})
- W arm loss decreased over training: {"yes" if w_decreased else "no"} (mean first 10 = {w_early:.6g}, mean last 10 = {w_late:.6g})
- `singular_spectrum` returned a sorted-descending array of length 10: {"yes" if (spectrum_shape_ok and spectrum_descending) else "no"}
- Spectrum changed between init and final: {"yes" if spectra_changed else "no"} (component-wise `|Δσ|` exceeded `N · eps` for at least one index)

## What this proves

This phase 1 smoke demonstrates the four-library integration (`jax` + `optax` + `landscape-archaeology` + `fws-bench`) compiles and runs end-to-end. It does **not** yet demonstrate that the FWS reparameterisation reshapes the loss landscape — that's phase 2, when the ondes SIREN replaces the placeholder renderer.

## Honest caveats

- The placeholder renderer is `tanh(G @ z)`, not SIREN; it's a smooth non-identity stand-in chosen to avoid an `ondes` import in phase 1
- The mainnet is a linear regressor, not an MLP — chosen to minimise the integration surface
- 500 outer steps is a smoke budget, not a convergence study; we report directional decrease, not absolute fit quality
- Two-checkpoint spectrum measurement (init + final); the full per-step trajectory of `∂ render_fn / ∂ z` is phase 2 work
- The task is noise-free, so MSE has a trivially-zero optimum — directional decrease here proves the optimisers move, not that the FWS path is competitive with W-direct
"""

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_FILE.write_text(md)
    print()
    print(f"Wrote research log to {RESEARCH_FILE}")


if __name__ == "__main__":
    main()
