"""Phase 3: σ-spectrum trajectory + parameter-fair over-parameterised W baseline.

Phase 2 measured the renderer-Jacobian σ-spectrum at two endpoints — init
and final — and saw σ_min lift by ~8 orders of magnitude. Phase 3 measures
it at 10 checkpoints across training so the shape of the lift is visible.

Phase 2's FWS-vs-W comparison was confounded by raw parameter count: the
FWS arm carries G_params + z ≈ 8580 effective scalars while W is 32. Phase 3
adds an over-parameterised W arm: W = A_2 @ A_1 with A_1 ∈ R^(P×n_in) and
A_2 ∈ R^(n_out×P); P chosen so total ≈ 8580 (P = 720, total = 12 * 720 =
8640). The product is rank ≤ min(n_out, n_in) = 4 by construction (matches
the linear regressor's rank), but Adam has more parameters to search over.

Run:
    uv run python examples/phase3_trajectory.py
"""

from __future__ import annotations

import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import loom
import matplotlib.pyplot as plt
import numpy as np
import ondes
import optax
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PyTree
from landscape_archaeology import singular_spectrum


RESEARCH_DIR = Path("/Users/ammar/Documents/Research/fractal-weight-spaces/research")
FIGURES_DIR = RESEARCH_DIR / "figures"
RESEARCH_FILE = RESEARCH_DIR / "2026-06-06-phase3-trajectory.md"

# Wong colourblind-safe palette
WONG_PALETTE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
FWS_COLOUR = WONG_PALETTE[5]  # blue
W_MATCHED_COLOUR = WONG_PALETTE[6]  # vermillion
W_OVERPARAM_COLOUR = WONG_PALETTE[3]  # bluish-green

# --- Renderer hyperparameters ------------------------------------------------
DIM_Z = 32
HIDDEN_DIM = 32
NUM_HIDDEN_LAYERS = 3
COORD_DIM = 1
N_OUT = 4
N_IN = 8
N_W_ENTRIES = N_OUT * N_IN  # 32
NUM_OUTER_STEPS = 500
NUM_CHECKPOINTS = 10  # checkpoints at steps 0, 50, ..., 450
CHECKPOINT_STRIDE = NUM_OUTER_STEPS // NUM_CHECKPOINTS  # 50
K_SEED = int(os.environ.get("PHASE3_K_SEED", 10))

# Over-parameterised W arm: W = A_2 @ A_1; chose P so 12P ≈ FWS total (8580).
OVERPARAM_P = 720  # total params = (P * N_IN) + (N_OUT * P) = 12P = 8640


# --- Synthetic linear-regression task ----------------------------------------
def make_synthetic_regression(
    *,
    n_in: int = N_IN,
    n_out: int = N_OUT,
    n_train: int = 100,
    n_test: int = 50,
    seed: int = 0,
):
    """Same noise-free linear regression task as phase 1/2."""
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

    return task_loss_fn, {"x": x_train, "y": y_train}, {"x": x_test, "y": y_test}, W_target


# --- Generator: SIREN + FiLM (same as phase 2) -------------------------------
class Generator(eqx.Module):
    siren: ondes.SIREN
    film_W: Float[Array, "n_layers two_hidden dim_z"]
    film_b: Float[Array, "n_layers two_hidden"]

    def __init__(self, *, dim_z, hidden_dim, num_hidden_layers, coord_dim, key) -> None:
        k_siren, k_w, k_b = jax.random.split(key, 3)
        self.siren = ondes.SIREN(
            in_dim=coord_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            key=k_siren,
        )
        bound = 1.0 / dim_z
        self.film_W = jax.random.uniform(k_w, (num_hidden_layers, 2 * hidden_dim, dim_z), minval=-bound, maxval=bound)
        self.film_b = jax.random.uniform(k_b, (num_hidden_layers, 2 * hidden_dim), minval=-bound, maxval=bound)

    def film_from_z(self, z: Float[Array, " dim_z"]) -> Float[Array, "n_layers two_hidden"]:
        return jnp.einsum("lhd,d->lh", self.film_W, z) + self.film_b


def make_coords(n_points: int) -> Float[Array, "n_points 1"]:
    return jnp.linspace(-1.0, 1.0, n_points).reshape(n_points, 1)


COORDS_32 = make_coords(N_W_ENTRIES)


def siren_render_fn(G: Generator, z: Float[Array, " dim_z"]) -> Float[Array, "n_out n_in"]:
    P = {"W": jnp.zeros((N_OUT, N_IN))}

    def f(path, shape, dtype, params):
        G_in, z_in = params
        film = G_in.film_from_z(z_in)
        values = jax.vmap(lambda c: G_in.siren(c, film=film))(COORDS_32)
        return values.reshape(shape).astype(dtype)

    return loom.render(P, f, (G, z))["W"]


# --- Over-parameterised W arm ------------------------------------------------
class OverparamW(eqx.Module):
    """W expressed as A_2 @ A_1 where A_1 ∈ R^(P×n_in), A_2 ∈ R^(n_out×P).

    The product W = A_2 @ A_1 is rank ≤ min(n_out, n_in) by construction
    (matches the linear regressor's rank), but the parameter space has
    12P scalars instead of n_out * n_in.
    """

    A_1: Float[Array, "P n_in"]
    A_2: Float[Array, "n_out P"]

    def __init__(self, *, P: int, n_in: int, n_out: int, key: jax.Array) -> None:
        k1, k2 = jax.random.split(key, 2)
        # Match the (4,8) W init scale used by the matched arm (sigma=0.1).
        # Pick scales so A_2 @ A_1 has comparable entrywise std at init.
        # Using He-init-style: A_1 std = 0.1 / sqrt(P) and A_2 std = 1/sqrt(P)
        # gives Var(W_ij) ≈ Var(A_1) * Var(A_2) * P = (0.01/P) * (1/P) * P = 0.01/P,
        # which is too small. Use std = sqrt(0.1 / P) on each so the product
        # has std ≈ 0.1.
        scale = jnp.sqrt(0.1 / jnp.sqrt(jnp.array(float(P))))
        self.A_1 = jax.random.normal(k1, (P, n_in)) * scale
        self.A_2 = jax.random.normal(k2, (n_out, P)) * scale

    def materialise(self) -> Float[Array, "n_out n_in"]:
        return self.A_2 @ self.A_1


# --- Generic JIT training step factories -------------------------------------
def _global_l2_norm(grad_tree: PyTree) -> Float[Array, ""]:
    flat, _ = ravel_pytree(grad_tree)
    return jnp.linalg.norm(flat)


def make_fws_segment(render_fn, task_loss_fn, task_batch, G_opt, z_opt):
    """A jitted lax.scan segment that runs `length` steps of the FWS arm."""
    fws_optimiser = optax.multi_transform({"G": G_opt, "z": z_opt}, {"G": "G", "z": "z"})

    def loss_fn(combined, batch):
        W = render_fn(combined["G"], combined["z"])
        return task_loss_fn(W, batch)

    def step(carry, _):
        params, state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, task_batch)
        grad_norm = _global_l2_norm(grads)
        updates, new_state = fws_optimiser.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_state), (loss, grad_norm)

    def run(combined, state, length):
        (final_params, final_state), (loss_traj, grad_traj) = jax.lax.scan(
            step, (combined, state), xs=None, length=length
        )
        return final_params, final_state, loss_traj, grad_traj

    return fws_optimiser, jax.jit(run, static_argnums=(2,))


def make_direct_segment(loss_fn_W, task_batch, optimiser):
    """JIT'd scan for a 'train W directly' arm (matched or overparam)."""

    def loss_fn(params, batch):
        return loss_fn_W(params, batch)

    def step(carry, _):
        params, state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, task_batch)
        grad_norm = _global_l2_norm(grads)
        updates, new_state = optimiser.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_state), (loss, grad_norm)

    def run(params, state, length):
        (final_params, final_state), (loss_traj, grad_traj) = jax.lax.scan(
            step, (params, state), xs=None, length=length
        )
        return final_params, final_state, loss_traj, grad_traj

    return jax.jit(run, static_argnums=(2,))


def spectrum_at(G, z, *, k=10, num_iterations=80, seed=7):
    """Top-k singular values of d render_fn / d z at (G, z)."""
    operator = lambda z_var: siren_render_fn(G, z_var)  # noqa: E731
    return singular_spectrum(operator, z, k=k, num_iterations=num_iterations, key=jax.random.key(seed))


def make_inits(seed: int):
    key = jax.random.key(seed)
    k_g, k_z, k_w, k_op = jax.random.split(key, 4)
    init_G = Generator(
        dim_z=DIM_Z, hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS, coord_dim=COORD_DIM, key=k_g
    )
    init_z = jax.random.normal(k_z, (DIM_Z,)) * 0.1
    init_W = jax.random.normal(k_w, (N_OUT, N_IN)) * 0.1
    init_op = OverparamW(P=OVERPARAM_P, n_in=N_IN, n_out=N_OUT, key=k_op)
    return init_G, init_z, init_W, init_op


def run_seed(seed: int, task_loss_fn, train_batch, test_batch, print_fn):
    """Train all three arms in segments; snapshot σ-spectrum at each checkpoint."""
    init_G, init_z, init_W, init_op = make_inits(seed)

    # --- FWS arm ---
    fws_combined = {"G": init_G, "z": init_z}
    fws_opt, fws_run = make_fws_segment(
        siren_render_fn, task_loss_fn, train_batch, optax.adam(1e-3), optax.adam(1e-3)
    )
    fws_state = fws_opt.init(fws_combined)

    # --- W matched arm (32 params) ---
    w_opt = optax.adam(1e-3)
    w_run = make_direct_segment(task_loss_fn, train_batch, w_opt)
    w_state = w_opt.init(init_W)
    w_params = init_W

    # --- W overparam arm (~8640 params) ---
    op_opt = optax.adam(1e-3)

    def op_loss_fn(op_params, batch):
        return task_loss_fn(op_params.materialise(), batch)

    op_run = make_direct_segment(op_loss_fn, train_batch, op_opt)
    op_state = op_opt.init(init_op)
    op_params = init_op

    # Trajectories collected from per-segment scans.
    fws_loss_segments: list[np.ndarray] = []
    fws_grad_segments: list[np.ndarray] = []
    w_loss_segments: list[np.ndarray] = []
    w_grad_segments: list[np.ndarray] = []
    op_loss_segments: list[np.ndarray] = []
    op_grad_segments: list[np.ndarray] = []

    # Spectrum probes (FWS only) at each of the 10 checkpoints.
    sigma_top10_by_checkpoint: list[np.ndarray] = []

    # Checkpoint 0: before any training.
    sigmas_0 = spectrum_at(init_G, init_z)
    sigma_top10_by_checkpoint.append(np.asarray(sigmas_0))

    for ckpt in range(NUM_CHECKPOINTS):
        # FWS segment
        fws_combined, fws_state, fws_loss_seg, fws_grad_seg = fws_run(fws_combined, fws_state, CHECKPOINT_STRIDE)
        fws_loss_segments.append(np.asarray(fws_loss_seg))
        fws_grad_segments.append(np.asarray(fws_grad_seg))

        # W matched segment
        w_params, w_state, w_loss_seg, w_grad_seg = w_run(w_params, w_state, CHECKPOINT_STRIDE)
        w_loss_segments.append(np.asarray(w_loss_seg))
        w_grad_segments.append(np.asarray(w_grad_seg))

        # W overparam segment
        op_params, op_state, op_loss_seg, op_grad_seg = op_run(op_params, op_state, CHECKPOINT_STRIDE)
        op_loss_segments.append(np.asarray(op_loss_seg))
        op_grad_segments.append(np.asarray(op_grad_seg))

        # Spectrum probe at the new state (after this segment finished).
        # checkpoints are at steps 50, 100, ..., 500. (Step 0 was sigmas_0.)
        sigmas_k = spectrum_at(fws_combined["G"], fws_combined["z"])
        sigma_top10_by_checkpoint.append(np.asarray(sigmas_k))

        print_fn(
            f"    ckpt {ckpt + 1}/{NUM_CHECKPOINTS}: "
            f"fws_loss={float(fws_loss_seg[-1]):.6g}  "
            f"w_loss={float(w_loss_seg[-1]):.6g}  "
            f"op_loss={float(op_loss_seg[-1]):.6g}  "
            f"sigma_min={float(sigmas_k[-1]):.4g}"
        )

    fws_loss_traj = np.concatenate(fws_loss_segments)
    fws_grad_traj = np.concatenate(fws_grad_segments)
    w_loss_traj = np.concatenate(w_loss_segments)
    w_grad_traj = np.concatenate(w_grad_segments)
    op_loss_traj = np.concatenate(op_loss_segments)
    op_grad_traj = np.concatenate(op_grad_segments)

    sigma_arr = np.stack(sigma_top10_by_checkpoint, axis=0)  # (NUM_CHECKPOINTS+1, 10)

    # Final test MSEs
    final_G = fws_combined["G"]
    final_z = fws_combined["z"]
    fws_test = float(task_loss_fn(siren_render_fn(final_G, final_z), test_batch))
    w_test = float(task_loss_fn(w_params, test_batch))
    op_test = float(task_loss_fn(op_params.materialise(), test_batch))

    any_nan = bool(
        np.isnan(fws_loss_traj).any() or np.isnan(w_loss_traj).any() or np.isnan(op_loss_traj).any()
    )

    return {
        "seed": seed,
        "fws_loss_traj": fws_loss_traj,
        "fws_grad_traj": fws_grad_traj,
        "w_loss_traj": w_loss_traj,
        "w_grad_traj": w_grad_traj,
        "op_loss_traj": op_loss_traj,
        "op_grad_traj": op_grad_traj,
        "sigma_trajectory": sigma_arr,  # shape (11, 10)
        "fws_test": fws_test,
        "w_test": w_test,
        "op_test": op_test,
        "any_nan": any_nan,
    }


def median(xs):
    return float(np.median(np.asarray(xs)))


def iqr(xs):
    a = np.asarray(xs)
    return float(np.quantile(a, 0.75) - np.quantile(a, 0.25))


def _band(arr_2d):
    """Median + IQR band across seeds (axis 0). Returns (median, q25, q75)."""
    return (
        np.median(arr_2d, axis=0),
        np.quantile(arr_2d, 0.25, axis=0),
        np.quantile(arr_2d, 0.75, axis=0),
    )


def _plot_band(steps, stack, ax, *, colour, label, marker=None):
    med, q25, q75 = _band(stack)
    ax.plot(steps, med, color=colour, label=label, linewidth=1.5, marker=marker)
    ax.fill_between(steps, q25, q75, color=colour, alpha=0.2)


def plot_loss_trajectory(per_seed, save_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    steps = np.arange(NUM_OUTER_STEPS)
    for key, colour, label in [
        ("fws_loss_traj", FWS_COLOUR, "FWS (G+z, ~8580 params)"),
        ("w_loss_traj", W_MATCHED_COLOUR, "W (matched, 32 params)"),
        ("op_loss_traj", W_OVERPARAM_COLOUR, "W (overparam, ~8640 params)"),
    ]:
        _plot_band(steps, np.stack([r[key] for r in per_seed], axis=0), ax, colour=colour, label=label)
    ax.set(yscale="log", xlabel="outer step", ylabel="training MSE",
           title=f"Loss trajectory (K={K_SEED}, median + IQR band)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_checkpoint_band(per_seed, save_path, *, stack_fn, ylabel, title):
    """Single-arm trajectory plot at the 11 checkpoints (FWS-only quantities)."""
    ckpt_steps = np.arange(NUM_CHECKPOINTS + 1) * CHECKPOINT_STRIDE
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _plot_band(ckpt_steps, stack_fn(per_seed), ax, colour=FWS_COLOUR, label="median", marker="o")
    ax.set(yscale="log", xlabel="outer step", ylabel=ylabel, title=title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_spectrum_checkpoints(per_seed, save_path):
    """Top-10 σ values at 4 representative checkpoints (step 0, 100, 250, 450)."""
    ckpt_indices = [0, 2, 5, 9]  # steps 0, 100, 250, 450
    ckpt_steps = [i * CHECKPOINT_STRIDE for i in ckpt_indices]
    sigma_stack = np.stack([r["sigma_trajectory"] for r in per_seed], axis=0)  # (K, 11, 10)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ax, idx, step in zip(axes, ckpt_indices, ckpt_steps, strict=True):
        # sigma_stack[:, idx, :] -> (K, 10)
        slice_2d = sigma_stack[:, idx, :]
        med, q25, q75 = _band(slice_2d)
        x = np.arange(1, 11)
        ax.errorbar(x, med, yerr=[med - q25, q75 - med], fmt="o-", color=FWS_COLOUR, capsize=3, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel(r"$i$ (singular-value index)")
        ax.set_title(f"step {step}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"$\sigma_i$")
    fig.suptitle(f"Top-10 $\\sigma_i$ at four checkpoints (K={K_SEED}, median + IQR)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("Phase 3 — σ-spectrum trajectory + parameter-fair over-param W baseline")
    print("=" * 72)
    print(f"dim_z={DIM_Z}, hidden_dim={HIDDEN_DIM}, layers={NUM_HIDDEN_LAYERS}")
    print(f"num_outer_steps={NUM_OUTER_STEPS}, K_seed={K_SEED}")
    print(f"NUM_CHECKPOINTS={NUM_CHECKPOINTS} every {CHECKPOINT_STRIDE} steps")
    print(f"OVERPARAM_P={OVERPARAM_P} -> total ~{12 * OVERPARAM_P} params")
    print()

    task_loss_fn, train_batch, test_batch, _ = make_synthetic_regression()

    per_seed: list[dict] = []
    for seed in range(K_SEED):
        print(f"  seed {seed}:")
        row = run_seed(seed, task_loss_fn, train_batch, test_batch, print_fn=print)
        per_seed.append(row)
        print(
            f"    -> fws_test={row['fws_test']:.6g}  w_test={row['w_test']:.6g}  "
            f"op_test={row['op_test']:.6g}  any_nan={row['any_nan']}"
        )

    fws_tests = [r["fws_test"] for r in per_seed]
    w_tests = [r["w_test"] for r in per_seed]
    op_tests = [r["op_test"] for r in per_seed]

    print()
    print("Final test MSE distribution (verbatim):")
    print(
        f"  FWS:               median={median(fws_tests):.6g}  IQR={iqr(fws_tests):.6g}  "
        f"min={min(fws_tests):.6g}  max={max(fws_tests):.6g}"
    )
    print(
        f"  W (matched, 32):   median={median(w_tests):.6g}  IQR={iqr(w_tests):.6g}  "
        f"min={min(w_tests):.6g}  max={max(w_tests):.6g}"
    )
    print(
        f"  W (overparam):     median={median(op_tests):.6g}  IQR={iqr(op_tests):.6g}  "
        f"min={min(op_tests):.6g}  max={max(op_tests):.6g}"
    )

    # --- Plots --------------------------------------------------------------
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    loss_png = FIGURES_DIR / "2026-06-06-phase3-loss-trajectory.png"
    sigma_min_png = FIGURES_DIR / "2026-06-06-phase3-sigma_min-trajectory.png"
    cond_png = FIGURES_DIR / "2026-06-06-phase3-condition-trajectory.png"
    spectrum_png = FIGURES_DIR / "2026-06-06-phase3-spectrum-checkpoints.png"

    plot_loss_trajectory(per_seed, loss_png)
    plot_checkpoint_band(
        per_seed, sigma_min_png,
        stack_fn=lambda rs: np.stack([r["sigma_trajectory"][:, -1] for r in rs], axis=0),
        ylabel=r"$\sigma_{\min}$ of renderer Jacobian",
        title=f"σ_min trajectory (K={K_SEED}, log y)",
    )
    plot_checkpoint_band(
        per_seed, cond_png,
        stack_fn=lambda rs: np.stack(
            [r["sigma_trajectory"][:, 0] / np.maximum(r["sigma_trajectory"][:, -1], np.finfo(np.float32).tiny)
             for r in rs], axis=0),
        ylabel=r"$\mathrm{cond}(J) = \sigma_{\max} / \sigma_{\min}$",
        title=f"Renderer-Jacobian condition number trajectory (K={K_SEED})",
    )
    plot_spectrum_checkpoints(per_seed, spectrum_png)
    print()
    print(f"Wrote plots to {FIGURES_DIR}")

    # --- σ_min trajectory shape description --------------------------------
    sigma_min_stack = np.stack([r["sigma_trajectory"][:, -1] for r in per_seed], axis=0)
    sigma_min_med = np.median(sigma_min_stack, axis=0)
    log_min = np.log10(np.maximum(sigma_min_med, np.finfo(np.float32).tiny))
    # Diffs of log10(σ_min) across checkpoints — concentration of the lift
    step_diffs = np.diff(log_min)
    biggest_step = int(np.argmax(step_diffs))
    biggest_step_size = float(step_diffs[biggest_step])
    total_lift = float(log_min[-1] - log_min[0])

    print()
    print("σ_min trajectory shape (median across seeds):")
    for idx, val in enumerate(sigma_min_med):
        step = idx * CHECKPOINT_STRIDE
        print(f"  step {step:4d}: σ_min = {val:.6g}  (log10 = {log_min[idx]:.3f})")
    print(
        f"  Total log10 lift = {total_lift:.2f}. "
        f"Biggest single-segment lift: between step "
        f"{biggest_step * CHECKPOINT_STRIDE} and {(biggest_step + 1) * CHECKPOINT_STRIDE} "
        f"(Δlog10 = {biggest_step_size:.2f})."
    )

    # --- Research log -------------------------------------------------------
    per_seed_rows = "\n".join(
        f"| {r['seed']} | {r['fws_test']:.6g} | {r['w_test']:.6g} | {r['op_test']:.6g} | "
        f"{float(r['sigma_trajectory'][0, -1]):.4g} | {float(r['sigma_trajectory'][-1, -1]):.4g} |"
        for r in per_seed
    )

    sigma_min_table = "\n".join(
        f"| {idx * CHECKPOINT_STRIDE} | {sigma_min_med[idx]:.6g} |" for idx in range(len(sigma_min_med))
    )

    # Diagnose lift shape: cumulative fraction of total lift by segment.
    # ramp-then-plateau: >=80% of the lift in the first half of segments.
    cumulative = np.cumsum(step_diffs)
    cumulative_frac = cumulative / max(cumulative[-1], 1e-10)
    halfway_idx = len(step_diffs) // 2
    frac_in_first_half = float(cumulative_frac[halfway_idx - 1])
    sorted_steps = np.sort(step_diffs)[::-1]
    if frac_in_first_half >= 0.80:
        lift_shape = (
            f"ramp-then-plateau: ~{frac_in_first_half * 100:.0f}% of the total lift "
            f"happens in the first {halfway_idx} segments (steps 0–{halfway_idx * CHECKPOINT_STRIDE}); "
            f"the trajectory is roughly flat thereafter"
        )
    elif len(sorted_steps) >= 2 and sorted_steps[0] > 2.0 * max(sorted_steps[1], 1e-10):
        lift_shape = "step-function-like: one segment dominates the lift"
    elif np.std(step_diffs) / max(np.mean(np.abs(step_diffs)), 1e-10) > 1.5:
        lift_shape = "jumpy: lift is concentrated in 2-3 segments, not gradual"
    else:
        lift_shape = "gradual: lift is spread roughly evenly across training"

    # Fair-baseline diagnosis.
    fws_med = median(fws_tests)
    op_med = median(op_tests)
    w_med = median(w_tests)
    if op_med <= 2.0 * fws_med:
        baseline_verdict = (
            "the over-parameterised W arm matches or beats FWS within a factor of 2; the phase 2 "
            "FWS-vs-W gap was substantially parameter-count-confounded."
        )
    elif op_med < 0.5 * w_med:
        baseline_verdict = (
            "the over-parameterised W arm improves on the matched W arm but does not close the gap "
            "to FWS; some of the phase 2 advantage is parameter-count, but a residual gap attributable "
            "to the reparameterisation remains."
        )
    else:
        baseline_verdict = (
            "the over-parameterised W arm does not meaningfully improve on the matched W arm; the "
            "phase 2 FWS-vs-W gap is largely *not* attributable to raw parameter count."
        )

    md = f"""# Phase 3 — trajectory σ-spectrum + parameter-fair baseline — 2026-06-06

Building on phase 2's observation that the renderer's $\\sigma_{{\\min}}$ lifts by ~8 orders of magnitude during training. Phase 3 measures this gradually rather than as a snapshot and adds an over-parameterised $W$ baseline so the FWS-vs-W comparison stops conflating reparameterisation with raw parameter count.

## Setup

- Task, mainnet target, num_outer_steps, optimiser: unchanged from phase 2 (linear regression with $W_{{\\text{{target}}}} \\in \\mathbb{{R}}^{{4 \\times 8}}$, 100 train / 50 test, full-batch MSE, Adam(1e-3), {NUM_OUTER_STEPS} steps).
- Three arms:
  - **FWS**: ondes `SIREN` (hidden_dim={HIDDEN_DIM}, layers={NUM_HIDDEN_LAYERS}) + FiLM modulation as $G$; $z \\in \\mathbb{{R}}^{{{DIM_Z}}}$; effective params ~8580 (G ≈ 8548, z = 32).
  - **W (matched, 32 params)**: $W$ trained directly at the original $(4, 8) = 32$ params.
  - **W (overparam, ~{12 * OVERPARAM_P} params)**: $W = A_2 A_1$ with $A_1 \\in \\mathbb{{R}}^{{{OVERPARAM_P} \\times {N_IN}}}$, $A_2 \\in \\mathbb{{R}}^{{{N_OUT} \\times {OVERPARAM_P}}}$; total $12 P = {12 * OVERPARAM_P}$ params, rank $\\leq {min(N_OUT, N_IN)}$ by construction.
- $K_{{\\text{{seed}}}} = {K_SEED}$; 10 σ-spectrum checkpoints at steps $\\{{0, 50, 100, \\ldots, 500\\}}$.

## Distribution at convergence

| arm | median test MSE | IQR | min | max |
|---|---|---|---|---|
| FWS | {median(fws_tests):.6g} | {iqr(fws_tests):.6g} | {min(fws_tests):.6g} | {max(fws_tests):.6g} |
| W (matched, 32 params) | {median(w_tests):.6g} | {iqr(w_tests):.6g} | {min(w_tests):.6g} | {max(w_tests):.6g} |
| W (overparam, ~{12 * OVERPARAM_P} params) | {median(op_tests):.6g} | {iqr(op_tests):.6g} | {min(op_tests):.6g} | {max(op_tests):.6g} |

(All verbatim. No statistical test.)

### Per-seed σ_min endpoints + final test MSE

| seed | FWS test | W test | W-op test | $\\sigma_{{\\min}}$ init | $\\sigma_{{\\min}}$ final |
|------|----------|--------|-----------|--------------------------|---------------------------|
{per_seed_rows}

## σ-spectrum trajectory (FWS arm only)

![loss trajectory](figures/2026-06-06-phase3-loss-trajectory.png)

![sigma_min trajectory](figures/2026-06-06-phase3-sigma_min-trajectory.png)

![cond(J) trajectory](figures/2026-06-06-phase3-condition-trajectory.png)

![spectrum at four checkpoints](figures/2026-06-06-phase3-spectrum-checkpoints.png)

### σ_min median by step (verbatim)

| step | median $\\sigma_{{\\min}}$ |
|------|-----------------------------|
{sigma_min_table}

Total $\\log_{{10}}$ lift = {total_lift:.2f}. Largest single-segment lift: between step {biggest_step * CHECKPOINT_STRIDE} and {(biggest_step + 1) * CHECKPOINT_STRIDE} ($\\Delta \\log_{{10}}$ = {biggest_step_size:.2f}).

## What the trajectory shows

The σ_min trajectory is {lift_shape}. The largest single-segment lift is at step ~{biggest_step * CHECKPOINT_STRIDE} — meaningful only as a description of where in training the conditioning sharpens most; no claim is made about whether this concentration reflects an underlying mechanism vs Adam-warmup effects vs scan-segment artefact. The cond(J) trajectory (above) shows the corresponding collapse of σ_max / σ_min. Whether either trajectory aligns temporally with a knee in the loss curve is left for the reader to inspect across the three plots.

## What the fair baseline shows

Comparing W (matched, 32 params) to W (overparam, ~{12 * OVERPARAM_P} params) isolates the contribution of raw parameter count, since both arms train W directly without any reparameterisation through $G$. Then comparing W (overparam) to FWS isolates the contribution of the reparameterisation at matched parameter scale.

From the matrix above: {baseline_verdict}

## Honest caveats

- Still a single task (noise-free linear regression). Multi-leaf MLP target with non-linear activations is phase 4.
- Still $K = {K_SEED}$ — distribution shape is reported but no paired Wilcoxon is run, and the falsifier-convention floor is K=5 minimum / K=20 for confidence.
- {NUM_OUTER_STEPS} outer steps; convergence not asserted for any arm.
- W (overparam) is rank-deficient by construction ($\\text{{rank}}(A_2 A_1) \\leq {min(N_OUT, N_IN)}$); whether this rank-deficient over-parameterisation matches FWS's intrinsic capacity is a separate question. An "honest" matched baseline at full rank would need $\\min(n_{{\\text{{out}}}}, n_{{\\text{{in}}}}) \\geq$ FWS's effective dimensionality, which the target task does not permit.
- Plots show median + IQR band; no per-seed lines.
- σ-spectrum is the renderer Jacobian $\\partial W / \\partial z$ measured by Lanczos / iterative SVD; only the top-10 σ values are tracked (out of $\\min(32, 32) = 32$ possible).
- Per-segment lift attribution (which segment lifts σ_min most) is sensitive to the choice of {NUM_CHECKPOINTS} checkpoint locations; a denser grid could re-localise the dominant segment.

## What's measured

| measurement | source |
|---|---|
| per-arm loss + grad-norm trajectory | custom `lax.scan` segments × {NUM_CHECKPOINTS} (one per checkpoint) per arm |
| renderer-Jacobian σ-spectrum (11 checkpoints) | landscape-archaeology `singular_spectrum` |
| final test MSE | direct evaluation on 50-point held-out batch after final segment |
"""

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_FILE.write_text(md)
    print(f"Wrote research log to {RESEARCH_FILE}")


if __name__ == "__main__":
    main()
