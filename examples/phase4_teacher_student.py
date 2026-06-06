"""Phase 4: non-linear teacher-student with a 2-layer SiLU MLP target.

Phase 3 showed the linear task is structurally degenerate: an over-parameterised
``W = A_2 @ A_1`` student has rank ``<= 4`` by construction and so matches the
rank-4 target trivially. Phase 4 swaps the target for a 2-layer SiLU MLP
teacher. The student now has four parameter leaves (``W1, b1, W2, b2``) and
must learn the non-linear hidden mixing — rank tricks no longer trivialise the
task.

Three arms, all learning the same 58-element ``(W1, b1, W2, b2)`` pytree:

- **W matched (58 params)** — direct training of the 58 scalars.
- **W overparam (~8700 params)** — every weight matrix replaced by
  ``A_2 @ A_1``; biases kept 1-D. Total ``= 22 P + 10`` with ``P = 395``.
- **FWS (~8740 params)** — ``ondes.SIREN + FiLM`` body, ``z in R^32``. The
  renderer evaluates ``G(coord, film(z))`` at a coord that concatenates a
  4-way one-hot leaf identifier with a 1-D within-leaf index.

Run:
    PHASE4_K_SEED=1 uv run python examples/phase4_teacher_student.py  # pilot
    uv run python examples/phase4_teacher_student.py                  # K=5
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
RESEARCH_FILE = RESEARCH_DIR / "2026-06-06-phase4-teacher-student.md"

# Wong colourblind-safe palette
WONG_PALETTE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
FWS_COLOUR = WONG_PALETTE[5]  # blue
W_MATCHED_COLOUR = WONG_PALETTE[6]  # vermillion
W_OVERPARAM_COLOUR = WONG_PALETTE[3]  # bluish-green

# --- Student / teacher topology ----------------------------------------------
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
TOTAL_W_PARAMS = sum(LEAF_SIZES.values())  # 32 + 8 + 16 + 2 = 58

# --- FWS renderer hyperparameters -------------------------------------------
DIM_Z = 32
HIDDEN_DIM_SIREN = 32
NUM_HIDDEN_LAYERS = 3
COORD_DIM = 1 + N_LEAVES  # 1-D within-leaf coord + 4-way leaf one-hot

# --- Over-parameterised arm: per-weight P giving ~FWS total -----------------
OVERPARAM_P = 395  # total = 22 * P + 10 = 8700 scalars (biases stay 1-D)

# --- Training schedule ------------------------------------------------------
NUM_OUTER_STEPS = 1000
NUM_CHECKPOINTS = 10
CHECKPOINT_STRIDE = NUM_OUTER_STEPS // NUM_CHECKPOINTS  # 100
K_SEED = int(os.environ.get("PHASE4_K_SEED", 5))

# Hessian / σ probe parameters
SIGMA_TOP_K = 10
SIGMA_POWER_ITERS = 80
HESSIAN_POWER_ITERS = 60


# --- Teacher-student data ----------------------------------------------------
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
    # Offset the teacher key so teacher != student init under any seed re-use.
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

    return task_loss_fn, train_batch, test_batch, teacher


# --- FWS generator -----------------------------------------------------------
class Generator(eqx.Module):
    siren: ondes.SIREN
    film_W: Float[Array, "n_layers two_hidden dim_z"]
    film_b: Float[Array, "n_layers two_hidden"]

    def __init__(self, *, dim_z: int, hidden_dim: int, num_hidden_layers: int, coord_dim: int, key: Array) -> None:
        k_siren, k_w, k_b = jax.random.split(key, 3)
        self.siren = ondes.SIREN(
            in_dim=coord_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, key=k_siren
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


# --- Coordinate stack for the multi-leaf renderer ---------------------------
def _within_leaf_coord(size: int) -> Float[Array, " size"]:
    """1-D positions on [-1, 1], one per leaf entry. Size-1 leaves collapse to 0."""
    if size == 1:
        return jnp.zeros((1,))
    return jnp.linspace(-1.0, 1.0, size)


def build_global_coords() -> tuple[Float[Array, "total coord_dim"], dict[str, slice]]:
    """All ``sum_k size_k`` coordinates concatenated; slices index each leaf back out."""
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
    """Render all four leaves through one shared SIREN+FiLM body."""
    P = {name: jnp.zeros(shape) for name, shape in LEAF_SHAPES.items()}

    def f(path, shape, dtype, params):
        G_in, z_in = params
        film = G_in.film_from_z(z_in)
        # The leaf name is in path[0].key for a dict-keyed pytree path.
        leaf_name = path[0].key
        sl = LEAF_SLICES[leaf_name]
        coords = GLOBAL_COORDS[sl]
        values = jax.vmap(lambda c: G_in.siren(c, film=film))(coords)
        return values.reshape(shape).astype(dtype)

    return loom.render(P, f, (G, z))


# --- Over-parameterised arm --------------------------------------------------
class OverparamMLP(eqx.Module):
    """Same SiLU MLP topology, but each W_k is the product ``A_2_k @ A_1_k``.

    Biases stay 1-D. Total scalars = 2 * P * (in1 + h1) + h + 2 * P * (h + o) + o
    = 22 P + 10 for (in=4, h=8, o=2).
    """

    A_W1_1: Float[Array, "P in_dim"]
    A_W1_2: Float[Array, "hidden_dim P"]
    A_W2_1: Float[Array, "P hidden_dim"]
    A_W2_2: Float[Array, "out_dim P"]
    b1: Float[Array, " hidden_dim"]
    b2: Float[Array, " out_dim"]

    def __init__(self, *, P: int, key: Array) -> None:
        k1a, k1b, k2a, k2b = jax.random.split(key, 4)
        # Match the matched-arm init scale of 0.1 entrywise std on each W:
        # Var(W) ≈ Var(A_1) * Var(A_2) * P. Pick std on each = (0.01 / P)^(1/4)
        # so the product's entrywise std is ~0.1.
        scale = (0.01 / P) ** 0.25
        self.A_W1_1 = jax.random.normal(k1a, (P, IN_DIM)) * scale
        self.A_W1_2 = jax.random.normal(k1b, (HIDDEN_DIM_STUDENT, P)) * scale
        self.A_W2_1 = jax.random.normal(k2a, (P, HIDDEN_DIM_STUDENT)) * scale
        self.A_W2_2 = jax.random.normal(k2b, (OUT_DIM, P)) * scale
        self.b1 = jnp.zeros((HIDDEN_DIM_STUDENT,))
        self.b2 = jnp.zeros((OUT_DIM,))

    def materialise(self) -> dict[str, Array]:
        return {
            "W1": self.A_W1_2 @ self.A_W1_1,
            "b1": self.b1,
            "W2": self.A_W2_2 @ self.A_W2_1,
            "b2": self.b2,
        }


# --- JIT'd scan segments -----------------------------------------------------
def _global_l2_norm(grad_tree: PyTree) -> Float[Array, ""]:
    flat, _ = ravel_pytree(grad_tree)
    return jnp.linalg.norm(flat)


def make_fws_segment(task_loss_fn, task_batch, G_opt, z_opt):
    fws_optimiser = optax.multi_transform({"G": G_opt, "z": z_opt}, {"G": "G", "z": "z"})

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


def make_direct_segment(loss_fn_W, task_batch, optimiser):
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


# --- Spectral probes ---------------------------------------------------------
def sigma_at(G: Generator, z: Array, *, seed: int = 7) -> Array:
    """Top-k singular values of ``d render / dz`` (full pytree output)."""
    operator = lambda z_var: siren_render_fn(G, z_var)  # noqa: E731
    return singular_spectrum(operator, z, k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS, key=jax.random.key(seed))


def hessian_top_at(params: PyTree, task_loss_fn, batch: dict, *, seed: int = 11) -> Array:
    """Top-k eigenvalues of the loss Hessian at ``params`` via Jacobian of ``jax.grad``."""
    grad_fn = jax.grad(lambda p: task_loss_fn(p, batch))
    return singular_spectrum(grad_fn, params, k=SIGMA_TOP_K, num_iterations=HESSIAN_POWER_ITERS, key=jax.random.key(seed))


# --- HT-SR α fit -------------------------------------------------------------
def ht_sr_alpha(W: Array) -> tuple[float, float]:
    """Power-law fit p_k ∝ k^{-α} to descending eigenvalues of W^T W.

    Returns ``(alpha, R_squared)``. Uses all available eigenvalues > eps. The
    eps floor is derived from the dtype so we never log a denormalised number.
    """
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


# --- Per-seed run ------------------------------------------------------------
def make_inits(seed: int):
    key = jax.random.key(seed)
    k_g, k_z, k_w, k_op = jax.random.split(key, 4)
    init_G = Generator(
        dim_z=DIM_Z,
        hidden_dim=HIDDEN_DIM_SIREN,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        coord_dim=COORD_DIM,
        key=k_g,
    )
    init_z = jax.random.normal(k_z, (DIM_Z,)) * 0.1
    k_w1, k_w2 = jax.random.split(k_w)
    init_W: dict[str, Array] = {
        "W1": jax.random.normal(k_w1, LEAF_SHAPES["W1"]) * 0.1,
        "b1": jnp.zeros(LEAF_SHAPES["b1"]),
        "W2": jax.random.normal(k_w2, LEAF_SHAPES["W2"]) * 0.1,
        "b2": jnp.zeros(LEAF_SHAPES["b2"]),
    }
    init_op = OverparamMLP(P=OVERPARAM_P, key=k_op)
    return init_G, init_z, init_W, init_op


def run_seed(seed: int, task_loss_fn, train_batch, test_batch, print_fn):
    init_G, init_z, init_W, init_op = make_inits(seed)

    # --- FWS arm ---
    fws_combined = {"G": init_G, "z": init_z}
    fws_opt, fws_run = make_fws_segment(task_loss_fn, train_batch, optax.adam(1e-3), optax.adam(1e-3))
    fws_state = fws_opt.init(fws_combined)

    # --- W matched arm ---
    w_opt = optax.adam(1e-3)
    w_run = make_direct_segment(task_loss_fn, train_batch, w_opt)
    w_state = w_opt.init(init_W)
    w_params = init_W

    # --- W overparam arm ---
    op_opt = optax.adam(1e-3)

    def op_loss_fn(op_params, batch):
        return task_loss_fn(op_params.materialise(), batch)

    op_run = make_direct_segment(op_loss_fn, train_batch, op_opt)
    op_state = op_opt.init(init_op)
    op_params = init_op

    fws_loss_segments: list[np.ndarray] = []
    fws_grad_segments: list[np.ndarray] = []
    w_loss_segments: list[np.ndarray] = []
    w_grad_segments: list[np.ndarray] = []
    op_loss_segments: list[np.ndarray] = []
    op_grad_segments: list[np.ndarray] = []
    sigma_by_checkpoint: list[np.ndarray] = []

    # Checkpoint 0: before training.
    sigma_by_checkpoint.append(np.asarray(sigma_at(init_G, init_z)))

    for ckpt in range(NUM_CHECKPOINTS):
        fws_combined, fws_state, fws_loss_seg, fws_grad_seg = fws_run(fws_combined, fws_state, CHECKPOINT_STRIDE)
        fws_loss_segments.append(np.asarray(fws_loss_seg))
        fws_grad_segments.append(np.asarray(fws_grad_seg))

        w_params, w_state, w_loss_seg, w_grad_seg = w_run(w_params, w_state, CHECKPOINT_STRIDE)
        w_loss_segments.append(np.asarray(w_loss_seg))
        w_grad_segments.append(np.asarray(w_grad_seg))

        op_params, op_state, op_loss_seg, op_grad_seg = op_run(op_params, op_state, CHECKPOINT_STRIDE)
        op_loss_segments.append(np.asarray(op_loss_seg))
        op_grad_segments.append(np.asarray(op_grad_seg))

        sigma_by_checkpoint.append(np.asarray(sigma_at(fws_combined["G"], fws_combined["z"])))

        print_fn(
            f"    ckpt {ckpt + 1}/{NUM_CHECKPOINTS}: "
            f"fws_loss={float(fws_loss_seg[-1]):.6g}  "
            f"w_loss={float(w_loss_seg[-1]):.6g}  "
            f"op_loss={float(op_loss_seg[-1]):.6g}  "
            f"sigma_min={float(sigma_by_checkpoint[-1][-1]):.4g}"
        )

    fws_loss_traj = np.concatenate(fws_loss_segments)
    fws_grad_traj = np.concatenate(fws_grad_segments)
    w_loss_traj = np.concatenate(w_loss_segments)
    w_grad_traj = np.concatenate(w_grad_segments)
    op_loss_traj = np.concatenate(op_loss_segments)
    op_grad_traj = np.concatenate(op_grad_segments)
    sigma_arr = np.stack(sigma_by_checkpoint, axis=0)

    # --- Final-state quantities -------------------------------------------
    final_G = fws_combined["G"]
    final_z = fws_combined["z"]
    fws_W_tree = siren_render_fn(final_G, final_z)
    fws_test = float(task_loss_fn(fws_W_tree, test_batch))
    w_test = float(task_loss_fn(w_params, test_batch))
    op_W_tree = op_params.materialise()
    op_test = float(task_loss_fn(op_W_tree, test_batch))

    # Hessian top-k for each arm at convergence.
    hess_fws = np.asarray(hessian_top_at(fws_W_tree, task_loss_fn, train_batch))
    hess_w = np.asarray(hessian_top_at(w_params, task_loss_fn, train_batch))
    hess_op = np.asarray(hessian_top_at(op_W_tree, task_loss_fn, train_batch))

    # HT-SR α on W1, W2 leaves of FWS and W-matched.
    alpha_fws_W1, r2_fws_W1 = ht_sr_alpha(fws_W_tree["W1"])
    alpha_fws_W2, r2_fws_W2 = ht_sr_alpha(fws_W_tree["W2"])
    alpha_w_W1, r2_w_W1 = ht_sr_alpha(w_params["W1"])
    alpha_w_W2, r2_w_W2 = ht_sr_alpha(w_params["W2"])
    alpha_op_W1, r2_op_W1 = ht_sr_alpha(op_W_tree["W1"])
    alpha_op_W2, r2_op_W2 = ht_sr_alpha(op_W_tree["W2"])

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
        "sigma_trajectory": sigma_arr,
        "fws_test": fws_test,
        "w_test": w_test,
        "op_test": op_test,
        "hess_fws": hess_fws,
        "hess_w": hess_w,
        "hess_op": hess_op,
        "alpha": {
            "fws_W1": alpha_fws_W1, "fws_W2": alpha_fws_W2,
            "w_W1": alpha_w_W1, "w_W2": alpha_w_W2,
            "op_W1": alpha_op_W1, "op_W2": alpha_op_W2,
        },
        "r2": {
            "fws_W1": r2_fws_W1, "fws_W2": r2_fws_W2,
            "w_W1": r2_w_W1, "w_W2": r2_w_W2,
            "op_W1": r2_op_W1, "op_W2": r2_op_W2,
        },
        "any_nan": any_nan,
    }


# --- Reductions / plotting ---------------------------------------------------
def median(xs):
    return float(np.median(np.asarray(xs)))


def iqr(xs):
    a = np.asarray(xs)
    return float(np.quantile(a, 0.75) - np.quantile(a, 0.25))


def _band(arr_2d):
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
        ("fws_loss_traj", FWS_COLOUR, "FWS (G+z, ~8740 params)"),
        ("w_loss_traj", W_MATCHED_COLOUR, "W (matched, 58 params)"),
        ("op_loss_traj", W_OVERPARAM_COLOUR, "W (overparam, ~8700 params)"),
    ]:
        _plot_band(steps, np.stack([r[key] for r in per_seed], axis=0), ax, colour=colour, label=label)
    ax.set(yscale="log", xlabel="outer step", ylabel="training MSE",
           title=f"Loss trajectory (K={K_SEED}, median + IQR band)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_sigma_min(per_seed, save_path):
    ckpt_steps = np.arange(NUM_CHECKPOINTS + 1) * CHECKPOINT_STRIDE
    sigma_min_stack = np.stack([r["sigma_trajectory"][:, -1] for r in per_seed], axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _plot_band(ckpt_steps, sigma_min_stack, ax, colour=FWS_COLOUR, label="median", marker="o")
    ax.set(yscale="log", xlabel="outer step", ylabel=r"$\sigma_{\min}$ of renderer Jacobian",
           title=f"σ_min trajectory (K={K_SEED}, log y)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_hessian_top(per_seed, save_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    arms = [
        ("hess_fws", FWS_COLOUR, "FWS"),
        ("hess_w", W_MATCHED_COLOUR, "W matched"),
        ("hess_op", W_OVERPARAM_COLOUR, "W overparam"),
    ]
    for x_pos, (key, colour, label) in enumerate(arms):
        top1 = np.array([r[key][0] for r in per_seed])  # top eigenvalue per seed
        ax.scatter([x_pos] * len(top1), top1, color=colour, s=40, zorder=3)
        med = np.median(top1)
        ax.hlines(med, x_pos - 0.25, x_pos + 0.25, colors=colour, linewidth=2.5,
                  label=f"{label} (median = {med:.3g})")
    ax.set(yscale="log", xlabel="arm", ylabel=r"top eigenvalue of $\nabla^2 L$",
           title=f"Loss Hessian top eigenvalue at convergence (K={K_SEED})")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[2] for a in arms])
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_alpha(per_seed, save_path):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    leaf_arms = [
        ("fws_W1", FWS_COLOUR, "o", "FWS · W1"),
        ("fws_W2", FWS_COLOUR, "s", "FWS · W2"),
        ("w_W1", W_MATCHED_COLOUR, "o", "W-matched · W1"),
        ("w_W2", W_MATCHED_COLOUR, "s", "W-matched · W2"),
        ("op_W1", W_OVERPARAM_COLOUR, "o", "W-overparam · W1"),
        ("op_W2", W_OVERPARAM_COLOUR, "s", "W-overparam · W2"),
    ]
    for x_pos, (key, colour, marker, label) in enumerate(leaf_arms):
        alphas = np.array([r["alpha"][key] for r in per_seed])
        r2s = np.array([r["r2"][key] for r in per_seed])
        finite = np.isfinite(alphas) & np.isfinite(r2s)
        if finite.any():
            ax.scatter([x_pos] * finite.sum(), alphas[finite], color=colour, marker=marker, s=50, zorder=3)
            med_a = float(np.median(alphas[finite]))
            med_r = float(np.median(r2s[finite]))
            ax.hlines(med_a, x_pos - 0.25, x_pos + 0.25, colors=colour, linewidth=2.0)
            ax.annotate(f"R²={med_r:.2f}", (x_pos, med_a), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)
    ax.set(xlabel="leaf · arm", ylabel=r"HT-SR α ($p_k \propto k^{-\alpha}$)",
           title=f"HT-SR α on rendered fc leaves at convergence (K={K_SEED})")
    ax.set_xticks(range(len(leaf_arms)))
    ax.set_xticklabels([a[3] for a in leaf_arms], rotation=20, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- σ_min lift shape diagnosis ---------------------------------------------
def diagnose_sigma_min_shape(per_seed):
    sigma_min_stack = np.stack([r["sigma_trajectory"][:, -1] for r in per_seed], axis=0)
    sigma_min_med = np.median(sigma_min_stack, axis=0)
    floor = np.finfo(sigma_min_med.dtype).tiny
    log_min = np.log10(np.maximum(sigma_min_med, floor))
    step_diffs = np.diff(log_min)
    biggest_step = int(np.argmax(step_diffs))
    biggest_step_size = float(step_diffs[biggest_step])
    total_lift = float(log_min[-1] - log_min[0])
    cumulative = np.cumsum(step_diffs)
    cumulative_frac = cumulative / max(cumulative[-1], 1e-10)
    halfway_idx = len(step_diffs) // 2
    frac_in_first_half = float(cumulative_frac[halfway_idx - 1]) if halfway_idx > 0 else float("nan")
    sorted_steps = np.sort(step_diffs)[::-1]
    if frac_in_first_half >= 0.80:
        shape = (
            f"ramp-then-plateau: ~{frac_in_first_half * 100:.0f}% of the lift in the first "
            f"{halfway_idx} segments (steps 0–{halfway_idx * CHECKPOINT_STRIDE}); roughly flat after"
        )
    elif len(sorted_steps) >= 2 and sorted_steps[0] > 2.0 * max(sorted_steps[1], 1e-10):
        shape = "step-function-like: one segment dominates the lift"
    elif np.std(step_diffs) / max(np.mean(np.abs(step_diffs)), 1e-10) > 1.5:
        shape = "jumpy: lift concentrated in 2-3 segments, not gradual"
    else:
        shape = "gradual: lift roughly even across training"
    return sigma_min_med, log_min, total_lift, biggest_step, biggest_step_size, shape


# --- Main --------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Phase 4 — non-linear teacher-student, 3-arm + Hessian + HT-SR α")
    print("=" * 72)
    print(f"teacher: 2-layer SiLU MLP (in={IN_DIM}, hidden={HIDDEN_DIM_STUDENT}, out={OUT_DIM})")
    print(f"student params (matched arm): {TOTAL_W_PARAMS}")
    print(f"overparam P={OVERPARAM_P} -> total {22 * OVERPARAM_P + 10} scalars")
    print(f"FWS: dim_z={DIM_Z}, hidden={HIDDEN_DIM_SIREN}, layers={NUM_HIDDEN_LAYERS}, coord_dim={COORD_DIM}")
    print(f"num_outer_steps={NUM_OUTER_STEPS}, K_seed={K_SEED}")
    print()

    per_seed: list[dict] = []
    for seed in range(K_SEED):
        task_loss_fn, train_batch, test_batch, _teacher = make_synthetic_task(seed)
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
        f"  W (matched, 58):   median={median(w_tests):.6g}  IQR={iqr(w_tests):.6g}  "
        f"min={min(w_tests):.6g}  max={max(w_tests):.6g}"
    )
    print(
        f"  W (overparam):     median={median(op_tests):.6g}  IQR={iqr(op_tests):.6g}  "
        f"min={min(op_tests):.6g}  max={max(op_tests):.6g}"
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    loss_png = FIGURES_DIR / "2026-06-06-phase4-loss-trajectory.png"
    sigma_min_png = FIGURES_DIR / "2026-06-06-phase4-sigma_min-trajectory.png"
    hess_png = FIGURES_DIR / "2026-06-06-phase4-hessian-top.png"
    alpha_png = FIGURES_DIR / "2026-06-06-phase4-htsr-alpha.png"

    plot_loss_trajectory(per_seed, loss_png)
    plot_sigma_min(per_seed, sigma_min_png)
    plot_hessian_top(per_seed, hess_png)
    plot_alpha(per_seed, alpha_png)
    print(f"Wrote plots to {FIGURES_DIR}")

    # σ_min shape
    sigma_min_med, log_min, total_lift, biggest_step, biggest_step_size, lift_shape = (
        diagnose_sigma_min_shape(per_seed)
    )

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

    # Hessian aggregate
    def _hess_med(key):
        return float(np.median(np.array([r[key][0] for r in per_seed])))

    h_fws = _hess_med("hess_fws")
    h_w = _hess_med("hess_w")
    h_op = _hess_med("hess_op")
    print()
    print("Hessian top-eigenvalue medians:")
    print(f"  FWS: {h_fws:.6g}")
    print(f"  W (matched): {h_w:.6g}")
    print(f"  W (overparam): {h_op:.6g}")

    # HT-SR α aggregates
    def _alpha_med(key):
        vals = np.array([r["alpha"][key] for r in per_seed])
        r2_vals = np.array([r["r2"][key] for r in per_seed])
        finite = np.isfinite(vals) & np.isfinite(r2_vals)
        if not finite.any():
            return float("nan"), float("nan")
        return float(np.median(vals[finite])), float(np.median(r2_vals[finite]))

    alpha_meds = {k: _alpha_med(k) for k in ["fws_W1", "fws_W2", "w_W1", "w_W2", "op_W1", "op_W2"]}
    print()
    print("HT-SR α (median across seeds, R² median):")
    for k, (a, r) in alpha_meds.items():
        print(f"  {k}: α = {a:.4g}  R² = {r:.4g}")

    # --- Research log ---
    per_seed_rows = "\n".join(
        f"| {r['seed']} | {r['fws_test']:.6g} | {r['w_test']:.6g} | {r['op_test']:.6g} | "
        f"{float(r['sigma_trajectory'][0, -1]):.4g} | {float(r['sigma_trajectory'][-1, -1]):.4g} |"
        for r in per_seed
    )
    sigma_min_table = "\n".join(
        f"| {idx * CHECKPOINT_STRIDE} | {sigma_min_med[idx]:.6g} |" for idx in range(len(sigma_min_med))
    )
    hess_table = (
        f"| FWS | {h_fws:.6g} |\n"
        f"| W (matched) | {h_w:.6g} |\n"
        f"| W (overparam) | {h_op:.6g} |"
    )
    alpha_table_rows = "\n".join(
        f"| {k.replace('_', ' · ')} | {a:.4g} | {r:.4g} |" for k, (a, r) in alpha_meds.items()
    )

    # Honest interpretation gate.
    fws_med = median(fws_tests)
    op_med = median(op_tests)
    w_med = median(w_tests)
    if fws_med < 0.5 * min(op_med, w_med):
        verdict = (
            "FWS's median test MSE is below half of both W arms', so the FWS advantage that survived "
            "phase 3's rank-trivialisation appears to persist on a non-linear target. Whether it "
            "would survive larger task families, deeper students, or seed counts beyond K=5 is not "
            "tested here."
        )
    elif op_med < 0.5 * fws_med:
        verdict = (
            "the over-parameterised W arm beats FWS on the non-linear target as well — the FWS "
            "advantage observed in phase 2 / partially attributed to parameter count in phase 3 does "
            "not transfer to this task."
        )
    elif abs(fws_med - op_med) / max(fws_med, op_med) < 0.5:
        verdict = (
            "FWS and over-parameterised W are within 50% of each other on test MSE — the phase 2 gap "
            "appears to have been parameter-count, not reparameterisation."
        )
    else:
        verdict = (
            "the three-arm comparison sits between the clean outcomes above; no single sentence "
            "captures it. Inspect the per-seed table."
        )

    md = f"""# Phase 4 — non-linear teacher-student — 2026-06-06

Phase 3 found the linear regression task was structurally degenerate for the over-parameterised $W = A_2 A_1$ baseline: the product has rank $\\leq 4$ by construction, which is exactly the rank of $W_{{\\text{{target}}}} \\in \\mathbb{{R}}^{{4 \\times 8}}$, so the overparam arm reached test MSE $\\sim 10^{{-12}}$ — eight orders of magnitude below FWS — by trivially rank-matching, not by leveraging the extra parameters in any meaningful sense. Phase 4 swaps the target for a non-linear teacher (2-layer SiLU MLP) so the rank trick is blocked.

## Setup

- **Teacher**: 2-layer SiLU MLP with $\\text{{in}} = {IN_DIM}$, $\\text{{hidden}} = {HIDDEN_DIM_STUDENT}$, $\\text{{out}} = {OUT_DIM}$. Weights $\\mathcal{{N}}(0, 1 / \\text{{fan-in}})$, biases zero. One teacher per seed; teacher key offset from student-init key so they cannot coincide.
- **Synthetic data**: $x \\sim \\mathcal{{N}}(0, I_{{{IN_DIM}}})$, $y = \\text{{teacher}}(x)$. $n_{{\\text{{train}}}} = 200$, $n_{{\\text{{test}}}} = 50$, full-batch MSE.
- **Student**: same topology as teacher; learns 4 leaves $\\{{W_1, b_1, W_2, b_2\\}}$ with shapes $({HIDDEN_DIM_STUDENT}, {IN_DIM}), ({HIDDEN_DIM_STUDENT},), ({OUT_DIM}, {HIDDEN_DIM_STUDENT}), ({OUT_DIM},)$ — **58 scalars total**.

### Three arms (all 1 000 steps, Adam(1e-3))

| arm | params | reparam |
|---|---|---|
| W matched | {TOTAL_W_PARAMS} | none — train the 58 scalars directly |
| W overparam | {22 * OVERPARAM_P + 10} | $W_k = A_{{2,k}} A_{{1,k}}$ with $P = {OVERPARAM_P}$ inner rank, biases left 1-D |
| FWS | ~8740 | `ondes.SIREN` (in_dim={COORD_DIM}, hidden_dim={HIDDEN_DIM_SIREN}, layers={NUM_HIDDEN_LAYERS}) + FiLM($z$), $z \\in \\mathbb{{R}}^{{{DIM_Z}}}$ |

The FWS renderer reads a coordinate of dimension {COORD_DIM} per scalar: a 4-way one-hot of leaf identity concatenated with a 1-D within-leaf position on $[-1, 1]$. Each scalar of $\\{{W_1, b_1, W_2, b_2\\}}$ is the SIREN's scalar readout at its leaf-specific coord. One SIREN body, one FiLM schedule per $z$ — the only thing varying across leaves is the coord.

- $K_{{\\text{{seed}}}} = {K_SEED}$
- 10 σ-spectrum checkpoints at steps $\\{{0, 100, 200, \\ldots, 1000\\}}$
- Hessian top-{SIGMA_TOP_K} eigenvalues at convergence (one set per arm)
- HT-SR α power-law fit to $W^T W$ eigenvalues at convergence on $W_1$ and $W_2$ leaves (all three arms)

## Final-step test MSE (verbatim)

| arm | median | IQR | min | max |
|---|---|---|---|---|
| FWS | {median(fws_tests):.6g} | {iqr(fws_tests):.6g} | {min(fws_tests):.6g} | {max(fws_tests):.6g} |
| W (matched, 58 params) | {median(w_tests):.6g} | {iqr(w_tests):.6g} | {min(w_tests):.6g} | {max(w_tests):.6g} |
| W (overparam, ~{22 * OVERPARAM_P + 10} params) | {median(op_tests):.6g} | {iqr(op_tests):.6g} | {min(op_tests):.6g} | {max(op_tests):.6g} |

### Per-seed table

| seed | FWS test | W test | W-op test | $\\sigma_{{\\min}}$ init | $\\sigma_{{\\min}}$ final |
|------|----------|--------|-----------|--------------------------|---------------------------|
{per_seed_rows}

## σ_min trajectory shape (FWS arm)

![loss trajectory](figures/2026-06-06-phase4-loss-trajectory.png)

![σ_min trajectory](figures/2026-06-06-phase4-sigma_min-trajectory.png)

### σ_min median by step (verbatim)

| step | median $\\sigma_{{\\min}}$ |
|------|-----------------------------|
{sigma_min_table}

Total $\\log_{{10}}$ lift = {total_lift:.2f}. Largest single-segment lift: between step {biggest_step * CHECKPOINT_STRIDE} and {(biggest_step + 1) * CHECKPOINT_STRIDE} ($\\Delta \\log_{{10}}$ = {biggest_step_size:.2f}). Shape: **{lift_shape}**.

## Hessian top eigenvalue at convergence

![Hessian top](figures/2026-06-06-phase4-hessian-top.png)

| arm | median top eigenvalue of $\\nabla^2 L$ |
|---|---|
{hess_table}

Read this as a loss-landscape sharpness comparison: larger top eigenvalue means a sharper minimum (in the direction of the principal curvature) at the point of convergence. No claim is made about flatness in directions not picked up by the top-10 power iteration.

## HT-SR α on rendered fc weights at convergence

![HT-SR α](figures/2026-06-06-phase4-htsr-alpha.png)

| leaf · arm | median α | median R² |
|---|---|---|
{alpha_table_rows}

The fit is to the descending eigenvalues of $W_k^T W_k$ (or $W_k W_k^T$, whichever has the larger dimension) on log-rank vs log-eigenvalue axes. $b_1$ and $b_2$ leaves are 1-D and not eligible for HT-SR. The W-matched arm's $W_2 \\in \\mathbb{{R}}^{{{OUT_DIM} \\times {HIDDEN_DIM_STUDENT}}}$ has only {min(OUT_DIM, HIDDEN_DIM_STUDENT)} eigenvalues; α from $n < 5$ points is reported but the R² is uninformative — this is a small-sample issue intrinsic to the topology, not the fit method.

## What the data shows

{verdict}

The non-linearity blocks the rank-trivialisation route phase 3 exposed — both W arms had to learn the SiLU mixing, not just project onto a rank-4 subspace. The verdict above is over the **single task family** of teacher-student with this exact teacher shape and 200/50 train/test split; no claim about other targets, scales, or data regimes.

## Honest caveats

- Single task family (2-layer SiLU MLP teacher, fixed dimensions). No transfer-target sweep, no depth or width ablation, no width/depth variation in the SIREN body.
- $K_{{\\text{{seed}}}} = {K_SEED}$ — minimum K for the falsifier-convention floor (see [FWS empirical-card falsifier convention](MEMORY)); enough to report a median with IQR but not enough to support paired Wilcoxon (n = 5).
- σ-spectrum is the renderer-Jacobian power-iteration estimate of top-{SIGMA_TOP_K} singular values; the true rank of the Jacobian is at most $\\min(\\text{{dim}}_z, \\text{{output dim}}) = \\min(32, 58) = 32$, so reported $\\sigma_{{\\min}}$ is the smallest of the top {SIGMA_TOP_K}, not the global smallest of the spectrum.
- Hessian top-eigenvalue likewise comes from {HESSIAN_POWER_ITERS}-step power iteration; the reported numbers are estimates, not exact eigenvalues. Different K, different seeds, and different power-iter counts could shift them — interpret as orders of magnitude, not precise comparisons.
- HT-SR α is fit on a power-iteration spectrum that has very few points for the 2×8 and 8×4 W leaves. The R² column is the load-bearing diagnostic: low R² means the power-law model didn't fit, even if α prints a number.
- No statistical test between arms.
- Adam(1e-3) for all three arms; no LR or optimiser sweep.

## What's measured

| measurement | source |
|---|---|
| per-arm loss + grad-norm trajectory | custom `lax.scan` segments × {NUM_CHECKPOINTS} (one per checkpoint) per arm |
| renderer-Jacobian σ-spectrum ({NUM_CHECKPOINTS + 1} checkpoints, FWS only) | landscape-archaeology `singular_spectrum` on `operator = render_fn(G, ·)` |
| loss-Hessian top-{SIGMA_TOP_K} eigenvalues (all three arms, final state) | landscape-archaeology `singular_spectrum` on `operator = jax.grad(task_loss_fn)` |
| HT-SR α on $W_1, W_2$ leaves (all three arms, final state) | inline `eigvalsh(W^T W)` + `polyfit` on log-rank vs log-eigenvalue |
| final test MSE | direct evaluation on 50-point held-out batch after final segment |
"""

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_FILE.write_text(md)
    print(f"Wrote research log to {RESEARCH_FILE}")


if __name__ == "__main__":
    main()
