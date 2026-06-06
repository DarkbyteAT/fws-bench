"""Phase 5: hyperparameter sweep — is FWS's α=9.8 fixable?

Phase 4 found that the FWS arm on a 2-layer SiLU teacher-student task produced
rendered W1 leaves with HT-SR power-law exponent α ≈ 9.81 — far steeper than
the [1.4, 1.7] range characteristic of trained-network weight spectra. Phase 4
also showed FWS underperformed both the matched-parameter (58 scalar) and
over-parameterised (~8700 scalar) W arms by 50× and 4 OoM respectively.

Phase 5 sweeps three FWS-side hyperparameters and tracks α throughout training
to distinguish "FWS is hyperparameter-limited on this task" from "FWS as
currently structured cannot match trained-network spectra here":

  - SIREN ω₀ ∈ {10, 30, 100}    (applied to both omega_first and omega_hidden;
                                  ondes defaults are 6.0 / 1.0, so all three
                                  swept values push the renderer toward higher
                                  initial frequency than phase 4)
  - num_outer_steps ∈ {1000, 5000}
  - per-group LR scheme ∈ {"matched" (G=1e-3, z=1e-3),
                           "split"   (G=1e-4, z=1e-3)}

3 × 2 × 2 = 12 cells, K_seed = 3 per cell (36 runs, no statistical test).
Per cell, at 10 evenly-spaced checkpoints: HT-SR α + R² on rendered W1 leaf,
test MSE on the 50-point held-out set, σ_{min,max} + cond(J) of the renderer
Jacobian. Hessian top eigenvalue at the final step only.

Plots: α trajectory grid (12 subplots), α-by-axis marginals (3 subplots),
final test-MSE heatmap, σ_min vs α scatter across all cells × seeds ×
checkpoints. Reference band at α ∈ [1.4, 1.7] on α plots — display only, not
a gate. No magic-number thresholds; verbatim numbers throughout.

The W matched / W overparam arms are NOT under test here and are dropped —
they're parameter-count baselines, not part of the FWS hyperparameter
question. Phase 4 already reports them.

Run:
    PHASE5_K_SEED=1 uv run python examples/phase5_hyperparam_sweep.py  # pilot
    uv run python examples/phase5_hyperparam_sweep.py                  # K=3
"""

from __future__ import annotations

import os
import time
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
RESEARCH_FILE = RESEARCH_DIR / "2026-06-06-phase5-hyperparam-sweep.md"

# Wong colourblind-safe palette
WONG_PALETTE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# --- Student / teacher topology (matches phase 4) ----------------------------
IN_DIM = 4
HIDDEN_DIM_STUDENT = 8
OUT_DIM = 2
LEAF_ORDER: tuple[str, ...] = ("W1", "b1", "W2", "b2")
N_LEAVES = len(LEAF_ORDER)

LEAF_SHAPES: dict[str, tuple[int, ...]] = {
    "W1": (HIDDEN_DIM_STUDENT, IN_DIM),
    "b1": (HIDDEN_DIM_STUDENT,),
    "W2": (OUT_DIM, HIDDEN_DIM_STUDENT),
    "b2": (OUT_DIM,),
}
LEAF_SIZES: dict[str, int] = {k: int(np.prod(s)) for k, s in LEAF_SHAPES.items()}

# --- FWS renderer fixed hyperparameters (matches phase 4) -------------------
DIM_Z = 32
HIDDEN_DIM_SIREN = 32
NUM_HIDDEN_LAYERS = 3
COORD_DIM = 1 + N_LEAVES  # 5

# --- Sweep grid -------------------------------------------------------------
OMEGA_VALUES = (10.0, 30.0, 100.0)
NUM_OUTER_STEPS_VALUES = (1000, 5000)
LR_SCHEMES: dict[str, tuple[float, float]] = {
    "matched": (1e-3, 1e-3),  # (G_lr, z_lr)
    "split": (1e-4, 1e-3),
}
K_SEED = int(os.environ.get("PHASE5_K_SEED", 3))

NUM_CHECKPOINTS = 10  # 10 evenly-spaced

# Sigma / Hessian probe parameters
SIGMA_TOP_K = 10
SIGMA_POWER_ITERS = 80
HESSIAN_POWER_ITERS = 60

# Phase 4 fixed cell for reference highlighting on plots
PHASE4_CELL = ("omega=30", "steps=1000", "lr=matched")


# --- Teacher-student data (identical to phase 4) ----------------------------
def make_teacher(key: Array) -> dict[str, Array]:
    k1, k2 = jax.random.split(key)
    return {
        "W1": jax.random.normal(k1, LEAF_SHAPES["W1"]) / jnp.sqrt(IN_DIM),
        "b1": jnp.zeros(LEAF_SHAPES["b1"]),
        "W2": jax.random.normal(k2, LEAF_SHAPES["W2"]) / jnp.sqrt(HIDDEN_DIM_STUDENT),
        "b2": jnp.zeros(LEAF_SHAPES["b2"]),
    }


def student_forward(params: dict[str, Array], x: Float[Array, " in_dim"]) -> Float[Array, " out_dim"]:
    h = jax.nn.silu(params["W1"] @ x + params["b1"])
    return params["W2"] @ h + params["b2"]


def make_synthetic_task(seed: int, *, n_train: int = 200, n_test: int = 50):
    key = jax.random.key(seed)
    k_teacher, k_train, k_test = jax.random.split(key, 3)
    teacher = make_teacher(jax.random.fold_in(k_teacher, 9999))

    def gen(k, n):
        xs = jax.random.normal(k, (n, IN_DIM))
        ys = jax.vmap(lambda x: student_forward(teacher, x))(xs)
        return {"x": xs, "y": ys}

    train_batch = gen(k_train, n_train)
    test_batch = gen(k_test, n_test)

    def task_loss_fn(params: dict[str, Array], batch: dict) -> Float[Array, ""]:
        preds = jax.vmap(lambda x: student_forward(params, x))(batch["x"])
        return jnp.mean((preds - batch["y"]) ** 2)

    return task_loss_fn, train_batch, test_batch


# --- FWS generator (ω₀ now configurable) ------------------------------------
class Generator(eqx.Module):
    siren: ondes.SIREN
    film_W: Float[Array, "n_layers two_hidden dim_z"]
    film_b: Float[Array, "n_layers two_hidden"]

    def __init__(
        self,
        *,
        dim_z: int,
        hidden_dim: int,
        num_hidden_layers: int,
        coord_dim: int,
        omega: float,
        key: Array,
    ) -> None:
        k_siren, k_w, k_b = jax.random.split(key, 3)
        self.siren = ondes.SIREN(
            in_dim=coord_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            key=k_siren,
            omega_first=omega,
            omega_hidden=omega,
        )
        bound = 1.0 / dim_z
        self.film_W = jax.random.uniform(
            k_w, (num_hidden_layers, 2 * hidden_dim, dim_z), minval=-bound, maxval=bound
        )
        self.film_b = jax.random.uniform(
            k_b, (num_hidden_layers, 2 * hidden_dim), minval=-bound, maxval=bound
        )

    def film_from_z(self, z: Float[Array, " dim_z"]) -> Float[Array, "n_layers two_hidden"]:
        return jnp.einsum("lhd,d->lh", self.film_W, z) + self.film_b


# --- Coordinate stack ------------------------------------------------------
def _within_leaf_coord(size: int) -> Float[Array, " size"]:
    if size == 1:
        return jnp.zeros((1,))
    return jnp.linspace(-1.0, 1.0, size)


def build_global_coords() -> tuple[Float[Array, "total coord_dim"], dict[str, slice]]:
    coord_chunks: list[Array] = []
    slices: dict[str, slice] = {}
    offset = 0
    for leaf_id, leaf_name in enumerate(LEAF_ORDER):
        size = LEAF_SIZES[leaf_name]
        within = _within_leaf_coord(size).reshape(size, 1)
        one_hot = jax.nn.one_hot(jnp.array(leaf_id), N_LEAVES)
        one_hot_block = jnp.broadcast_to(one_hot, (size, N_LEAVES))
        coord_chunks.append(jnp.concatenate([within, one_hot_block], axis=1))
        slices[leaf_name] = slice(offset, offset + size)
        offset += size
    coords = jnp.concatenate(coord_chunks, axis=0)
    return coords, slices


GLOBAL_COORDS, LEAF_SLICES = build_global_coords()


def siren_render_fn(G: Generator, z: Float[Array, " dim_z"]) -> dict[str, Array]:
    P = {name: jnp.zeros(shape) for name, shape in LEAF_SHAPES.items()}

    def f(path, shape, dtype, params):
        G_in, z_in = params
        film = G_in.film_from_z(z_in)
        leaf_name = path[0].key
        sl = LEAF_SLICES[leaf_name]
        coords = GLOBAL_COORDS[sl]
        values = jax.vmap(lambda c: G_in.siren(c, film=film))(coords)
        return values.reshape(shape).astype(dtype)

    return loom.render(P, f, (G, z))


# --- Training scan ---------------------------------------------------------
def _global_l2_norm(grad_tree: PyTree) -> Float[Array, ""]:
    flat, _ = ravel_pytree(grad_tree)
    return jnp.linalg.norm(flat)


def make_fws_segment(task_loss_fn, task_batch, G_lr: float, z_lr: float):
    fws_optimiser = optax.multi_transform(
        {"G": optax.adam(G_lr), "z": optax.adam(z_lr)},
        {"G": "G", "z": "z"},
    )

    def loss_fn(combined, batch):
        W_tree = siren_render_fn(combined["G"], combined["z"])
        return task_loss_fn(W_tree, batch)

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


# --- Spectral probes -------------------------------------------------------
def sigma_at(G: Generator, z: Array, *, seed: int = 7) -> Array:
    """Top-k singular values of d render / dz (full pytree output)."""
    operator = lambda z_var: siren_render_fn(G, z_var)  # noqa: E731
    return singular_spectrum(operator, z, k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS, key=jax.random.key(seed))


def hessian_top_at(params: PyTree, task_loss_fn, batch: dict, *, seed: int = 11) -> Array:
    grad_fn = jax.grad(lambda p: task_loss_fn(p, batch))
    return singular_spectrum(grad_fn, params, k=SIGMA_TOP_K, num_iterations=HESSIAN_POWER_ITERS, key=jax.random.key(seed))


# --- HT-SR α ---------------------------------------------------------------
def ht_sr_alpha(W: Array) -> tuple[float, float]:
    """Power-law fit p_k ∝ k^{-α} to descending eigenvalues of W^T W."""
    M = W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T
    eigs = jnp.sort(jnp.linalg.eigvalsh(M))[::-1]
    eps = jnp.finfo(eigs.dtype).eps * eigs.shape[0] * jnp.maximum(eigs[0], jnp.array(1.0))
    mask = eigs > eps
    eigs_pos = eigs[mask]
    n = eigs_pos.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    ranks = jnp.arange(1, n + 1)
    log_r = jnp.log(ranks)
    log_p = jnp.log(eigs_pos)
    slope, intercept = jnp.polyfit(log_r, log_p, 1)
    alpha = float(-slope)
    log_p_hat = slope * log_r + intercept
    ss_res = float(jnp.sum((log_p - log_p_hat) ** 2))
    ss_tot = float(jnp.sum((log_p - jnp.mean(log_p)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return alpha, r2


# --- Per-cell, per-seed run ------------------------------------------------
def make_init_G(seed: int, omega: float) -> tuple[Generator, Array]:
    key = jax.random.key(seed)
    k_g, k_z = jax.random.split(key, 2)
    init_G = Generator(
        dim_z=DIM_Z,
        hidden_dim=HIDDEN_DIM_SIREN,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        coord_dim=COORD_DIM,
        omega=omega,
        key=k_g,
    )
    init_z = jax.random.normal(k_z, (DIM_Z,)) * 0.1
    return init_G, init_z


def run_cell_seed(
    seed: int,
    omega: float,
    num_outer_steps: int,
    G_lr: float,
    z_lr: float,
    task_loss_fn,
    train_batch,
    test_batch,
):
    """Run one (cell, seed) combination; collect per-checkpoint diagnostics."""
    init_G, init_z = make_init_G(seed, omega)
    fws_combined = {"G": init_G, "z": init_z}
    fws_opt, fws_run = make_fws_segment(task_loss_fn, train_batch, G_lr, z_lr)
    fws_state = fws_opt.init(fws_combined)

    checkpoint_stride = num_outer_steps // NUM_CHECKPOINTS

    # Pre-allocate per-checkpoint arrays (NUM_CHECKPOINTS + 1 entries: pre-train + 10 post).
    alphas = np.full(NUM_CHECKPOINTS + 1, np.nan)
    r2s = np.full(NUM_CHECKPOINTS + 1, np.nan)
    test_mses = np.full(NUM_CHECKPOINTS + 1, np.nan)
    sigma_min = np.full(NUM_CHECKPOINTS + 1, np.nan)
    sigma_max = np.full(NUM_CHECKPOINTS + 1, np.nan)
    cond_J = np.full(NUM_CHECKPOINTS + 1, np.nan)
    loss_segments: list[np.ndarray] = []

    def measure(ckpt_idx: int):
        G_now = fws_combined["G"]
        z_now = fws_combined["z"]
        W_tree = siren_render_fn(G_now, z_now)
        a, r = ht_sr_alpha(W_tree["W1"])
        alphas[ckpt_idx] = a
        r2s[ckpt_idx] = r
        test_mses[ckpt_idx] = float(task_loss_fn(W_tree, test_batch))
        sigma = np.asarray(sigma_at(G_now, z_now))
        sigma_min[ckpt_idx] = float(sigma[-1])
        sigma_max[ckpt_idx] = float(sigma[0])
        cond_J[ckpt_idx] = float(sigma[0] / max(sigma[-1], np.finfo(sigma.dtype).tiny))

    # Pre-training checkpoint
    measure(0)

    for ckpt in range(NUM_CHECKPOINTS):
        fws_combined, fws_state, loss_seg, _grad_seg = fws_run(fws_combined, fws_state, checkpoint_stride)
        loss_segments.append(np.asarray(loss_seg))
        measure(ckpt + 1)

    loss_traj = np.concatenate(loss_segments)

    # Hessian at final step only
    final_G = fws_combined["G"]
    final_z = fws_combined["z"]
    final_W_tree = siren_render_fn(final_G, final_z)
    hess = np.asarray(hessian_top_at(final_W_tree, task_loss_fn, train_batch))

    any_nan = bool(np.isnan(loss_traj).any())

    return {
        "seed": seed,
        "alphas": alphas,
        "r2s": r2s,
        "test_mses": test_mses,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "cond_J": cond_J,
        "loss_traj": loss_traj,
        "hess_top": float(hess[0]),
        "any_nan": any_nan,
        "checkpoint_steps": np.arange(NUM_CHECKPOINTS + 1) * checkpoint_stride,
    }


# --- Sweep orchestration ---------------------------------------------------
def cell_id(omega: float, steps: int, lr_name: str) -> str:
    return f"omega={int(omega)}, steps={steps}, lr={lr_name}"


def run_sweep():
    cells: list[dict] = []
    total_cells = len(OMEGA_VALUES) * len(NUM_OUTER_STEPS_VALUES) * len(LR_SCHEMES)
    cell_idx = 0
    t0 = time.time()
    nan_or_crash: list[str] = []

    for omega in OMEGA_VALUES:
        for num_outer_steps in NUM_OUTER_STEPS_VALUES:
            for lr_name, (G_lr, z_lr) in LR_SCHEMES.items():
                cell_idx += 1
                cid = cell_id(omega, num_outer_steps, lr_name)
                print(f"[{cell_idx}/{total_cells}] cell: {cid}  (elapsed {time.time() - t0:.1f}s)")
                seeds_data: list[dict] = []
                cell_crashed = False
                for seed in range(K_SEED):
                    try:
                        row = run_cell_seed(
                            seed, omega, num_outer_steps, G_lr, z_lr,
                            *make_synthetic_task(seed),
                        )
                        seeds_data.append(row)
                        if row["any_nan"]:
                            nan_or_crash.append(f"{cid} seed={seed}: NaN in loss traj")
                        print(
                            f"    seed {seed}: final_alpha={row['alphas'][-1]:.4g}  "
                            f"final_test_mse={row['test_mses'][-1]:.4g}  "
                            f"sigma_min_final={row['sigma_min'][-1]:.4g}  "
                            f"hess_top={row['hess_top']:.4g}  any_nan={row['any_nan']}"
                        )
                    except Exception as e:  # noqa: BLE001
                        cell_crashed = True
                        nan_or_crash.append(f"{cid} seed={seed}: CRASH — {type(e).__name__}: {e}")
                        print(f"    seed {seed}: CRASH — {type(e).__name__}: {e}")
                cells.append({
                    "omega": omega,
                    "num_outer_steps": num_outer_steps,
                    "lr_name": lr_name,
                    "G_lr": G_lr,
                    "z_lr": z_lr,
                    "cell_id": cid,
                    "seeds": seeds_data,
                    "crashed": cell_crashed,
                })
                # Progress ping after first cell completes
                if cell_idx == 1:
                    print(f"  --- first cell complete ({time.time() - t0:.1f}s wall) ---")
    print(f"Sweep complete: {time.time() - t0:.1f}s total wall.")
    return cells, nan_or_crash


# --- Reductions ------------------------------------------------------------
def cell_median_final_alpha(cell: dict) -> float:
    vals = [s["alphas"][-1] for s in cell["seeds"] if not np.isnan(s["alphas"][-1])]
    return float(np.median(vals)) if vals else float("nan")


def cell_median_final_test_mse(cell: dict) -> float:
    vals = [s["test_mses"][-1] for s in cell["seeds"]]
    return float(np.median(vals)) if vals else float("nan")


def cell_min_alpha_over_training(cell: dict) -> float:
    """Lowest α observed in this cell across any seed × any checkpoint."""
    arr = np.concatenate([s["alphas"] for s in cell["seeds"]]) if cell["seeds"] else np.array([np.nan])
    arr = arr[np.isfinite(arr)]
    return float(arr.min()) if arr.size else float("nan")


def any_alpha_in_band(cell: dict, lo: float = 1.4, hi: float = 1.7) -> bool:
    for s in cell["seeds"]:
        a = s["alphas"]
        a = a[np.isfinite(a)]
        if a.size and ((a >= lo) & (a <= hi)).any():
            return True
    return False


# --- Plotting --------------------------------------------------------------
def plot_alpha_trajectory_grid(cells: list[dict], save_path: Path):
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharey=True)
    axes = axes.flatten()
    for i, cell in enumerate(cells):
        ax = axes[i]
        for s in cell["seeds"]:
            ax.plot(s["checkpoint_steps"], s["alphas"], color=WONG_PALETTE[5], alpha=0.6, linewidth=1)
        ax.axhspan(1.4, 1.7, color=WONG_PALETTE[3], alpha=0.18, label="trained-net α band")
        ax.set_yscale("log")
        ax.set_title(cell["cell_id"], fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
        if i % 4 == 0:
            ax.set_ylabel("HT-SR α (W1)")
        if i >= 8:
            ax.set_xlabel("outer step")
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle(f"Phase 5: α trajectory per cell (K={K_SEED}, all seeds overplotted)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _alpha_band(cells: list[dict], filter_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median / 25th / 75th percentile of α across all (seed, step) pairs in matching cells."""
    matching = [c for c in cells if filter_fn(c)]
    if not matching:
        return np.array([]), np.array([]), np.array([])
    # Step axis differs by num_outer_steps; align by fractional progress (10 evenly-spaced + start).
    stack = []
    for c in matching:
        for s in c["seeds"]:
            stack.append(s["alphas"])
    arr = np.stack(stack, axis=0)  # (S, n_ckpt)
    return (
        np.nanmedian(arr, axis=0),
        np.nanquantile(arr, 0.25, axis=0),
        np.nanquantile(arr, 0.75, axis=0),
    )


def plot_alpha_by_axis(cells: list[dict], save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    x = np.arange(NUM_CHECKPOINTS + 1) / NUM_CHECKPOINTS  # fractional training progress

    # (a) by omega
    ax = axes[0]
    for i, w in enumerate(OMEGA_VALUES):
        med, q25, q75 = _alpha_band(cells, lambda c, w=w: c["omega"] == w)
        col = WONG_PALETTE[1 + i]
        ax.plot(x, med, color=col, label=f"ω₀={int(w)}", linewidth=2)
        ax.fill_between(x, q25, q75, color=col, alpha=0.18)
    ax.axhspan(1.4, 1.7, color=WONG_PALETTE[3], alpha=0.18)
    ax.set(xlabel="fractional training progress", ylabel="HT-SR α (W1)", title="by ω₀")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    # (b) by num_outer_steps
    ax = axes[1]
    for i, ns in enumerate(NUM_OUTER_STEPS_VALUES):
        med, q25, q75 = _alpha_band(cells, lambda c, ns=ns: c["num_outer_steps"] == ns)
        col = WONG_PALETTE[5 + i]
        ax.plot(x, med, color=col, label=f"steps={ns}", linewidth=2)
        ax.fill_between(x, q25, q75, color=col, alpha=0.18)
    ax.axhspan(1.4, 1.7, color=WONG_PALETTE[3], alpha=0.18)
    ax.set(xlabel="fractional training progress", title="by num_outer_steps")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    # (c) by LR scheme
    ax = axes[2]
    for i, lr_name in enumerate(LR_SCHEMES):
        med, q25, q75 = _alpha_band(cells, lambda c, lr_name=lr_name: c["lr_name"] == lr_name)
        col = WONG_PALETTE[2 + i * 2]
        ax.plot(x, med, color=col, label=f"lr={lr_name}", linewidth=2)
        ax.fill_between(x, q25, q75, color=col, alpha=0.18)
    ax.axhspan(1.4, 1.7, color=WONG_PALETTE[3], alpha=0.18)
    ax.set(xlabel="fractional training progress", title="by LR scheme")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(f"Phase 5: α marginal trajectories (K={K_SEED}, median + IQR over matching cells)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_test_mse_heatmap(cells: list[dict], save_path: Path):
    # rows = omega, cols = (steps, lr_scheme) — 3 rows × 4 cols
    n_rows = len(OMEGA_VALUES)
    n_cols = len(NUM_OUTER_STEPS_VALUES) * len(LR_SCHEMES)
    grid = np.full((n_rows, n_cols), np.nan)
    col_labels: list[str] = []
    for j, ns in enumerate(NUM_OUTER_STEPS_VALUES):
        for k, lr_name in enumerate(LR_SCHEMES):
            col_labels.append(f"steps={ns}\nlr={lr_name}")
            col_index = j * len(LR_SCHEMES) + k
            for i, w in enumerate(OMEGA_VALUES):
                for c in cells:
                    if c["omega"] == w and c["num_outer_steps"] == ns and c["lr_name"] == lr_name:
                        grid[i, col_index] = cell_median_final_test_mse(c)
                        break

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Log-scale colormap (test MSE spans orders of magnitude)
    log_grid = np.log10(np.maximum(grid, np.finfo(grid.dtype).tiny))
    im = ax.imshow(log_grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"ω₀={int(w)}" for w in OMEGA_VALUES], fontsize=9)
    for i in range(n_rows):
        for j in range(n_cols):
            v = grid[i, j]
            label = f"{v:.3g}" if np.isfinite(v) else "NaN"
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(median final test MSE)", fontsize=9)
    ax.set_title(f"Phase 5: final test MSE heatmap (median across K={K_SEED} seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_sigma_vs_alpha(cells: list[dict], save_path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # One scatter per omega value to colour-code
    for i, w in enumerate(OMEGA_VALUES):
        xs: list[float] = []
        ys: list[float] = []
        for c in cells:
            if c["omega"] != w:
                continue
            for s in c["seeds"]:
                for k in range(NUM_CHECKPOINTS + 1):
                    smin = s["sigma_min"][k]
                    a = s["alphas"][k]
                    if np.isfinite(smin) and np.isfinite(a) and smin > 0 and a > 0:
                        xs.append(smin)
                        ys.append(a)
        col = WONG_PALETTE[1 + i]
        ax.scatter(xs, ys, color=col, alpha=0.5, s=14, label=f"ω₀={int(w)}")
    ax.axhspan(1.4, 1.7, color=WONG_PALETTE[3], alpha=0.18, label="trained-net α band")
    ax.set(xscale="log", yscale="log",
           xlabel=r"$\sigma_{\min}$ (renderer Jacobian, top-10 power iter)",
           ylabel=r"HT-SR α (W1)",
           title=f"Phase 5: σ_min vs α across all cells × seeds × checkpoints (K={K_SEED})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- Research log generation ----------------------------------------------
def build_results_table(cells: list[dict]) -> tuple[str, dict]:
    """Return the markdown table of all cells + the closest-to-band cell info."""
    rows: list[str] = []
    closest_cell = None
    closest_distance = float("inf")
    band_target = 1.55  # midpoint of [1.4, 1.7]
    any_in_band_global = False
    any_in_band_cell_id: str | None = None

    for c in cells:
        med_alpha = cell_median_final_alpha(c)
        med_mse = cell_median_final_test_mse(c)
        min_a = cell_min_alpha_over_training(c)
        in_band = any_alpha_in_band(c)
        if in_band:
            any_in_band_global = True
            if any_in_band_cell_id is None:
                any_in_band_cell_id = c["cell_id"]
        d = abs(med_alpha - band_target) if np.isfinite(med_alpha) else float("inf")
        if d < closest_distance:
            closest_distance = d
            closest_cell = c
        rows.append(
            f"| {int(c['omega'])} | {c['num_outer_steps']} | {c['lr_name']} | "
            f"{med_alpha:.4g} | {min_a:.4g} | {med_mse:.4g} | {'yes' if in_band else 'no'} |"
        )

    table = (
        "| ω₀ | steps | LR scheme | median final α (W1) | min α over training (any seed/step) | "
        "median final test MSE | any α in [1.4, 1.7] at any step (any seed)? |\n"
        "|----|-------|-----------|----------------------|--------------------------------------|"
        "------------------------|---------------------------------------------|\n"
        + "\n".join(rows)
    )
    return table, {
        "closest_cell": closest_cell,
        "closest_distance": closest_distance,
        "any_in_band_global": any_in_band_global,
        "any_in_band_cell_id": any_in_band_cell_id,
    }


def describe_marginal_directions(cells: list[dict]) -> dict:
    """Median final α per axis value, to report direction of effect."""

    def med_alpha_for(filter_fn):
        vals = []
        for c in cells:
            if filter_fn(c):
                for s in c["seeds"]:
                    a = s["alphas"][-1]
                    if np.isfinite(a):
                        vals.append(a)
        return float(np.median(vals)) if vals else float("nan")

    by_omega = {int(w): med_alpha_for(lambda c, w=w: c["omega"] == w) for w in OMEGA_VALUES}
    by_steps = {int(ns): med_alpha_for(lambda c, ns=ns: c["num_outer_steps"] == ns) for ns in NUM_OUTER_STEPS_VALUES}
    by_lr = {nm: med_alpha_for(lambda c, nm=nm: c["lr_name"] == nm) for nm in LR_SCHEMES}
    return {"by_omega": by_omega, "by_steps": by_steps, "by_lr": by_lr}


def _direction_phrase(d: dict[int | str, float], axis_name: str) -> str:
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    finite_pairs = [(k, v) for k, v in zip(keys, vals, strict=True) if np.isfinite(v)]
    if len(finite_pairs) < 2:
        return f"{axis_name}: insufficient data"
    finite_pairs.sort(key=lambda kv: kv[1])
    lo_key, lo_val = finite_pairs[0]
    hi_key, hi_val = finite_pairs[-1]
    return f"{axis_name}: median final α lowest at {lo_key} ({lo_val:.4g}), highest at {hi_key} ({hi_val:.4g})"


# --- Main -----------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Phase 5 — hyperparameter sweep: SIREN ω₀ × num_outer_steps × LR scheme")
    print("=" * 72)
    print(f"Sweep grid: ω₀ ∈ {OMEGA_VALUES}, steps ∈ {NUM_OUTER_STEPS_VALUES}, LR ∈ {list(LR_SCHEMES)}")
    print(f"K_seed = {K_SEED}; total = {len(OMEGA_VALUES) * len(NUM_OUTER_STEPS_VALUES) * len(LR_SCHEMES) * K_SEED} runs")
    print(f"Phase 4 cell for reference: {PHASE4_CELL}")
    print("ondes SIREN defaults: omega_first=6.0, omega_hidden=1.0 (we override both with the swept ω₀)")
    print()

    cells, nan_or_crash = run_sweep()

    table, closest_info = build_results_table(cells)
    marginals = describe_marginal_directions(cells)

    print()
    print("RESULTS TABLE:")
    print(table)
    print()
    print("Marginal medians:")
    for axis_name, d in [("by_omega", marginals["by_omega"]),
                         ("by_steps", marginals["by_steps"]),
                         ("by_lr", marginals["by_lr"])]:
        print(f"  {axis_name}: {d}")
    if closest_info["closest_cell"]:
        c = closest_info["closest_cell"]
        print()
        print(f"Cell closest to α band midpoint (1.55): {c['cell_id']}")
        print(f"  median final α (W1) = {cell_median_final_alpha(c):.4g}")
        print(f"  median final test MSE = {cell_median_final_test_mse(c):.4g}")
    print()
    print(f"Any cell with α ∈ [1.4, 1.7] at any step (any seed)? "
          f"{'YES — ' + closest_info['any_in_band_cell_id'] if closest_info['any_in_band_global'] else 'no'}")
    print()
    if nan_or_crash:
        print("NaN / crash report:")
        for line in nan_or_crash:
            print(f"  {line}")
    else:
        print("No NaNs or crashes.")

    # --- Plots ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    alpha_grid_png = FIGURES_DIR / "2026-06-06-phase5-alpha-trajectory-grid.png"
    alpha_axis_png = FIGURES_DIR / "2026-06-06-phase5-alpha-by-axis.png"
    test_heatmap_png = FIGURES_DIR / "2026-06-06-phase5-test-mse-heatmap.png"
    sigma_vs_alpha_png = FIGURES_DIR / "2026-06-06-phase5-sigma-vs-alpha.png"

    plot_alpha_trajectory_grid(cells, alpha_grid_png)
    plot_alpha_by_axis(cells, alpha_axis_png)
    plot_test_mse_heatmap(cells, test_heatmap_png)
    plot_sigma_vs_alpha(cells, sigma_vs_alpha_png)
    print(f"Wrote plots to {FIGURES_DIR}")

    # --- Research log ---
    closest_cell = closest_info["closest_cell"]
    closest_alpha = cell_median_final_alpha(closest_cell) if closest_cell else float("nan")
    closest_mse = cell_median_final_test_mse(closest_cell) if closest_cell else float("nan")
    band_finding = (
        f"YES — cell '{closest_info['any_in_band_cell_id']}' produced α in [1.4, 1.7] "
        f"at some checkpoint for at least one seed."
        if closest_info["any_in_band_global"]
        else "no cell produced α in [1.4, 1.7] at any checkpoint across all seeds."
    )
    omega_phrase = _direction_phrase(marginals["by_omega"], "ω₀")
    steps_phrase = _direction_phrase(marginals["by_steps"], "num_outer_steps")
    lr_phrase = _direction_phrase(marginals["by_lr"], "LR scheme")

    nan_summary = (
        "\n".join(f"- {line}" for line in nan_or_crash) if nan_or_crash else "- none."
    )

    md = f"""# Phase 5 — hyperparameter sweep: is the α=9.8 collapse fixable? — 2026-06-06

## Background

Phase 4 (commits `32f2952` fws-bench, `adebeb1` fws — local pending push) ran a 3-arm benchmark (FWS, W matched, W overparam) on a non-linear 2-layer SiLU teacher-student task. Result: FWS underperformed both W arms (median test MSE 0.329 vs 0.007 vs 0.0003), and the FWS-rendered $W_1$ leaf had HT-SR power-law exponent $\\alpha = 9.81$ — far steeper than the [1.4, 1.7] range characteristic of trained-network weight spectra. The mechanistic question phase 4 left open: is the α collapse an artefact of FWS hyperparameters (ω₀ too low to give the SIREN body enough frequency, too few outer steps, or matched LRs preventing G/z from disentangling) — or is the FWS coordinate / FiLM / SIREN composition fundamentally incapable of producing trained-network-like spectra on this task family?

Phase 5 sweeps three hyperparameters and tracks α throughout training in every cell. The horizontal band at $\\alpha \\in [1.4, 1.7]$ on the α plots is reference, not a gate.

## Setup

Same 2-layer SiLU MLP teacher as phase 4 (in=4, hidden=8, out=2), same 200 train + 50 test points per seed, same FWS topology (`ondes.SIREN` in_dim=5, hidden_dim=32, layers=3, `Generator` with FiLM($z$), $z \\in \\mathbb{{R}}^{{32}}$). Only the swept hyperparameters change. The W matched and W overparam arms from phase 4 are dropped — they are not under test here.

### Sweep grid

- $\\omega_0 \\in \\{{10, 30, 100\\}}$, applied to both `omega_first` and `omega_hidden` (ondes defaults are 6.0 and 1.0; phase 4 used the defaults, so all three swept values push initial-frequency higher than phase 4)
- `num_outer_steps` $\\in \\{{1000, 5000\\}}$
- LR scheme: **matched** (Adam, $G$ lr $= 10^{{-3}}$, $z$ lr $= 10^{{-3}}$) or **split** (Adam, $G$ lr $= 10^{{-4}}$, $z$ lr $= 10^{{-3}}$)

$3 \\times 2 \\times 2 = 12$ cells, $K_{{\\text{{seed}}}} = {K_SEED}$ seeds per cell, 36 runs total. Per cell, per seed, per checkpoint (10 evenly-spaced + initial): HT-SR α and $R^2$ on rendered $W_1$ leaf, test MSE on 50-point held-out batch, $\\sigma_{{\\min}}$ / $\\sigma_{{\\max}}$ / $\\text{{cond}}(J)$ of $\\partial \\text{{render}} / \\partial z$. Hessian top-{SIGMA_TOP_K} eigenvalues at final step only.

Phase 4 fixed cell for reference: `{cell_id(30, 1000, 'matched')}`.

## Results

### Per-cell summary

{table}

The "any α in [1.4, 1.7]" column reports whether *any* (seed × checkpoint) pair in that cell landed in the trained-network band during training, not just the final step.

### Cell closest to α band

Cell closest to the α band midpoint (1.55): **{closest_cell['cell_id'] if closest_cell else 'n/a'}**, with median final α (W1) = {closest_alpha:.4g}, median final test MSE = {closest_mse:.4g}.

### Did any cell reach α ∈ [1.4, 1.7]?

**{band_finding}**

### Direction of each axis (medians of final α across all matching cells)

- {omega_phrase}
- {steps_phrase}
- {lr_phrase}

### α trajectory grid

![α trajectory grid](figures/2026-06-06-phase5-alpha-trajectory-grid.png)

### α by axis (marginals)

![α by axis](figures/2026-06-06-phase5-alpha-by-axis.png)

### Final test MSE heatmap

![test MSE heatmap](figures/2026-06-06-phase5-test-mse-heatmap.png)

### σ_min vs α correlation

![σ_min vs α](figures/2026-06-06-phase5-sigma-vs-alpha.png)

## NaN / crash report

{nan_summary}

## What the data shows

The verbatim table above is the primary artefact. The shape of each axis is summarised in the marginals section. Whether the [1.4, 1.7] band was ever reached at any checkpoint is reported in the "any α in [1.4, 1.7]" column and the band-finding summary. **No "FWS recovered" or "FWS failed" framing is added on top of the numbers**; that interpretation depends on follow-up work (K=10 confirmation, paired test against phase 4's cell, or moving to a different architectural change if the band was never reached).

## Honest caveats

- Sweep range was finite. The grid does not cover ω₀ $\\geq$ 300, num_outer_steps $\\geq$ 50 000, alternative optimisers (SGD, Lion), or different basis families (H-SIREN, WIRE, Fourier features). A "no cell in band" result here does not rule out fixability outside this grid.
- $K_{{\\text{{seed}}}} = {K_SEED}$ per cell, below the falsifier-convention floor of K=5. Numbers report distribution shape and direction of effect; no statistical test.
- α is fit per-checkpoint on a single leaf ($W_1$, the largest one with $\\geq 4$ eigenvalues); rendered $W_2$ has only 2 eigenvalues and was skipped (a $W_2 \\in \\mathbb{{R}}^{{2 \\times 8}}$ leaf's covariance has rank $\\leq 2$, so the power-law fit on $n < 5$ ranks is uninformative — see phase 4 caveat).
- One single 2-layer SiLU teacher per seed; this result may not generalise to larger or differently-structured targets.
- $\\sigma_{{\\min}}$ is from a top-{SIGMA_TOP_K} power iteration on the renderer Jacobian; it is the smallest of the top {SIGMA_TOP_K}, not the global smallest of the spectrum.
- Hessian top eigenvalue at final step is a one-shot diagnostic, not a primary sweep axis; numbers are recorded but not plotted as a separate panel here.
- The horizontal band at $\\alpha \\in [1.4, 1.7]$ is a reference for trained-network spectra (per the FWS programme design doc). It is not a calibrated falsifier and not a gate; cells outside it are not "failures" and cells inside are not "successes" — what matters is the per-cell verbatim α.

## What this implies for phase 6+

If any cell did reach $\\alpha \\in [1.4, 1.7]$ at any point: phase 6 should focus on the winning cell (or the cell closest to band midpoint), run it at K=10 with the same diagnostics, and add a paired Wilcoxon vs the phase 4 cell on final test MSE. Then test transfer to a different teacher topology and/or width.

If no cell reached the band: phase 6 should move to architectural changes (per-leaf $G$'s, drop the 1-D-within-leaf coord in favour of a 2-D-within-leaf coord, switch to a non-SIREN basis from ondes such as H-SIREN, or replace FiLM with a different conditioning scheme). The current sweep grid would have ruled out the "just train longer / start at a higher frequency / give $z$ more headroom" hypotheses.

The decision between these two phase 6 directions follows from the **any α in [1.4, 1.7] at any step (any seed)** column. If it is "yes" in $\\geq 1$ cell: pursue the winning cell. If "no" in all cells: pursue an architectural change.
"""

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_FILE.write_text(md)
    print(f"Wrote research log to {RESEARCH_FILE}")


if __name__ == "__main__":
    main()
