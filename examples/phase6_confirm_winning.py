"""Phase 6: K=10 confirmation of the winning cell + paired Wilcoxon.

Phase 5 (commits `c8f1722` fws-bench, `ecbd37d` fws — local) identified
``(omega_first=omega_hidden=100, num_outer_steps=5000, lr=matched)`` as the
cell achieving median final α(W1) = 1.456 — inside the trained-network band
[1.4, 1.7] — and median test MSE 3.55e-4 across K=3 seeds. Phase 6 confirms
this at K=10 with the fws empirical-card statistical convention.

Two paired comparisons:

- **FWS vs W matched** — does FWS beat the 58-param baseline?
- **FWS vs W overparam** — does FWS close the gap to the ~8700-param
  ``A_2 @ A_1`` factored baseline?

All three arms run at ``num_outer_steps = 5000`` so the FWS step budget from
phase 5 matches the W arms; this means the W arms here have 5× more steps
than phase 4 — reported as a caveat, not used as a cross-phase comparison.

Statistical tests (per pair):

- Paired Wilcoxon signed-rank on (FWS_test_MSE_i - baseline_test_MSE_i)
- Bootstrap-BCa 95% CI on the median paired difference (10 000 resamples)
- Hodges-Lehmann effect size (median of all pairwise differences)

Run:
    PHASE6_K_SEED=1 uv run python examples/phase6_confirm_winning.py  # pilot
    uv run python examples/phase6_confirm_winning.py                  # K=10
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
import scipy.stats as sp_stats
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PyTree
from landscape_archaeology import singular_spectrum


RESEARCH_DIR = Path("/Users/ammar/Documents/Research/fractal-weight-spaces/research")
FIGURES_DIR = RESEARCH_DIR / "figures"
RESEARCH_FILE = RESEARCH_DIR / "2026-06-06-phase6-confirmation.md"

# Wong colourblind-safe palette
WONG_PALETTE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
FWS_COLOUR = WONG_PALETTE[5]            # blue
W_MATCHED_COLOUR = WONG_PALETTE[6]      # vermillion
W_OVERPARAM_COLOUR = WONG_PALETTE[3]    # bluish-green

# --- Student / teacher topology (matches phases 4-5) -------------------------
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
TOTAL_W_PARAMS = sum(LEAF_SIZES.values())  # 58

# --- FWS renderer hyperparameters (winning cell from phase 5) ---------------
DIM_Z = 32
HIDDEN_DIM_SIREN = 32
NUM_HIDDEN_LAYERS = 3
COORD_DIM = 1 + N_LEAVES  # 5
OMEGA_FIRST = 100.0
OMEGA_HIDDEN = 100.0
G_LR = 1e-3
Z_LR = 1e-3

# --- Over-parameterised arm: same P as phase 4 -------------------------------
OVERPARAM_P = 395  # total = 22 P + 10 = 8700 scalars

# --- Training schedule ------------------------------------------------------
NUM_OUTER_STEPS = 5000  # all three arms; phase 5 winning cell at this length
NUM_CHECKPOINTS = 10
CHECKPOINT_STRIDE = NUM_OUTER_STEPS // NUM_CHECKPOINTS  # 500
K_SEED = int(os.environ.get("PHASE6_K_SEED", 10))

# Spectral probes
SIGMA_TOP_K = 10
SIGMA_POWER_ITERS = 80
HESSIAN_POWER_ITERS = 60

# Stats
BOOTSTRAP_N_RESAMPLES = 10_000
BOOTSTRAP_CI_LEVEL = 0.95


# --- Teacher-student data (identical to phases 4-5) -------------------------
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


# --- FWS generator (omega-configurable, matches phase 5) --------------------
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
        omega_first: float,
        omega_hidden: float,
        key: Array,
    ) -> None:
        k_siren, k_w, k_b = jax.random.split(key, 3)
        self.siren = ondes.SIREN(
            in_dim=coord_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            key=k_siren,
            omega_first=omega_first,
            omega_hidden=omega_hidden,
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


# --- Over-parameterised arm (identical to phase 4) --------------------------
class OverparamMLP(eqx.Module):
    A_W1_1: Float[Array, "P in_dim"]
    A_W1_2: Float[Array, "hidden_dim P"]
    A_W2_1: Float[Array, "P hidden_dim"]
    A_W2_2: Float[Array, "out_dim P"]
    b1: Float[Array, " hidden_dim"]
    b2: Float[Array, " out_dim"]

    def __init__(self, *, P: int, key: Array) -> None:
        k1a, k1b, k2a, k2b = jax.random.split(key, 4)
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


# --- Spectral probes -------------------------------------------------------
def sigma_at(G: Generator, z: Array, *, seed: int = 7) -> Array:
    operator = lambda z_var: siren_render_fn(G, z_var)  # noqa: E731
    return singular_spectrum(operator, z, k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS, key=jax.random.key(seed))


def hessian_top_at(params: PyTree, task_loss_fn, batch: dict, *, seed: int = 11) -> Array:
    grad_fn = jax.grad(lambda p: task_loss_fn(p, batch))
    return singular_spectrum(grad_fn, params, k=SIGMA_TOP_K, num_iterations=HESSIAN_POWER_ITERS, key=jax.random.key(seed))


# --- HT-SR α ---------------------------------------------------------------
def ht_sr_alpha(W: Array) -> tuple[float, float]:
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
        omega_first=OMEGA_FIRST,
        omega_hidden=OMEGA_HIDDEN,
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


def run_seed(seed: int, task_loss_fn, train_batch, test_batch):
    """Run all three arms for one seed; collect per-checkpoint diagnostics."""
    init_G, init_z, init_W, init_op = make_inits(seed)

    # --- FWS arm ---
    fws_combined = {"G": init_G, "z": init_z}
    fws_opt, fws_run = make_fws_segment(task_loss_fn, train_batch, G_LR, Z_LR)
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

    # Per-checkpoint FWS-side diagnostics
    alphas = np.full(NUM_CHECKPOINTS + 1, np.nan)
    r2s = np.full(NUM_CHECKPOINTS + 1, np.nan)
    sigma_min = np.full(NUM_CHECKPOINTS + 1, np.nan)
    sigma_max = np.full(NUM_CHECKPOINTS + 1, np.nan)
    cond_J = np.full(NUM_CHECKPOINTS + 1, np.nan)

    def measure_fws(ckpt_idx: int):
        G_now = fws_combined["G"]
        z_now = fws_combined["z"]
        W_tree = siren_render_fn(G_now, z_now)
        a, r = ht_sr_alpha(W_tree["W1"])
        alphas[ckpt_idx] = a
        r2s[ckpt_idx] = r
        sigma = np.asarray(sigma_at(G_now, z_now))
        sigma_min[ckpt_idx] = float(sigma[-1])
        sigma_max[ckpt_idx] = float(sigma[0])
        cond_J[ckpt_idx] = float(sigma[0] / max(sigma[-1], np.finfo(sigma.dtype).tiny))

    fws_loss_segments: list[np.ndarray] = []
    w_loss_segments: list[np.ndarray] = []
    op_loss_segments: list[np.ndarray] = []

    measure_fws(0)

    for ckpt in range(NUM_CHECKPOINTS):
        fws_combined, fws_state, fws_loss_seg, _ = fws_run(fws_combined, fws_state, CHECKPOINT_STRIDE)
        fws_loss_segments.append(np.asarray(fws_loss_seg))

        w_params, w_state, w_loss_seg, _ = w_run(w_params, w_state, CHECKPOINT_STRIDE)
        w_loss_segments.append(np.asarray(w_loss_seg))

        op_params, op_state, op_loss_seg, _ = op_run(op_params, op_state, CHECKPOINT_STRIDE)
        op_loss_segments.append(np.asarray(op_loss_seg))

        measure_fws(ckpt + 1)

    fws_loss_traj = np.concatenate(fws_loss_segments)
    w_loss_traj = np.concatenate(w_loss_segments)
    op_loss_traj = np.concatenate(op_loss_segments)

    # Final test
    final_G = fws_combined["G"]
    final_z = fws_combined["z"]
    fws_W_tree = siren_render_fn(final_G, final_z)
    fws_test = float(task_loss_fn(fws_W_tree, test_batch))
    w_test = float(task_loss_fn(w_params, test_batch))
    op_W_tree = op_params.materialise()
    op_test = float(task_loss_fn(op_W_tree, test_batch))

    # Hessian top eigenvalue at convergence (all three arms)
    hess_fws = float(np.asarray(hessian_top_at(fws_W_tree, task_loss_fn, train_batch))[0])
    hess_w = float(np.asarray(hessian_top_at(w_params, task_loss_fn, train_batch))[0])
    hess_op = float(np.asarray(hessian_top_at(op_W_tree, task_loss_fn, train_batch))[0])

    any_nan = bool(
        np.isnan(fws_loss_traj).any() or np.isnan(w_loss_traj).any() or np.isnan(op_loss_traj).any()
    )

    return {
        "seed": seed,
        "fws_test": fws_test,
        "w_test": w_test,
        "op_test": op_test,
        "fws_loss_traj": fws_loss_traj,
        "w_loss_traj": w_loss_traj,
        "op_loss_traj": op_loss_traj,
        "alphas": alphas,
        "r2s": r2s,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "cond_J": cond_J,
        "hess_fws": hess_fws,
        "hess_w": hess_w,
        "hess_op": hess_op,
        "any_nan": any_nan,
        "checkpoint_steps": np.arange(NUM_CHECKPOINTS + 1) * CHECKPOINT_STRIDE,
    }


# --- Statistics ------------------------------------------------------------
def hodges_lehmann(diffs: np.ndarray) -> float:
    """Hodges-Lehmann estimator: median of all pairwise averages.

    For the one-sample case (paired differences), HL is the median of all
    (d_i + d_j) / 2 for i <= j, which is the location estimator paired with
    the Wilcoxon signed-rank test.
    """
    n = diffs.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i, n):
            pairs.append((diffs[i] + diffs[j]) / 2.0)
    return float(np.median(np.asarray(pairs)))


def bca_ci_median_diff(diffs: np.ndarray, n_resamples: int, ci_level: float) -> tuple[float, float]:
    """Bootstrap-BCa 95% CI on the median of paired differences."""
    rng = np.random.default_rng(0)
    res = sp_stats.bootstrap(
        (diffs,),
        np.median,
        n_resamples=n_resamples,
        confidence_level=ci_level,
        method="BCa",
        random_state=rng,
        vectorized=False,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Returns (W_statistic, p_value, paired_diffs)."""
    diffs = a - b
    # zero_method="wilcox" (default) drops zeros — fine for continuous MSE.
    res = sp_stats.wilcoxon(a, b, zero_method="wilcox", correction=False, alternative="two-sided", method="exact")
    return float(res.statistic), float(res.pvalue), diffs


# --- Plotting --------------------------------------------------------------
def plot_mse_boxplot(per_seed: list[dict], save_path: Path):
    fws = np.array([r["fws_test"] for r in per_seed])
    w = np.array([r["w_test"] for r in per_seed])
    op = np.array([r["op_test"] for r in per_seed])
    data = [fws, w, op]
    colours = [FWS_COLOUR, W_MATCHED_COLOUR, W_OVERPARAM_COLOUR]
    labels = ["FWS", "W matched (58)", "W overparam (~8700)"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, c in zip(bp["boxes"], colours, strict=True):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(c)
    for median_line, c in zip(bp["medians"], colours, strict=True):
        median_line.set_color(c)
        median_line.set_linewidth(2.0)
    for whisker in bp["whiskers"]:
        whisker.set_color("#444")
    for cap in bp["caps"]:
        cap.set_color("#444")
    # Overlay individual points (jittered).
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(data, colours, strict=True), start=1):
        jitter = rng.uniform(-0.10, 0.10, size=vals.size)
        ax.scatter(np.full_like(vals, i) + jitter, vals, color=c, s=40, zorder=3, edgecolor="white", linewidths=0.6)
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("final test MSE")
    ax.set_title(f"Phase 6: final test MSE distribution (K={K_SEED})")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_alpha_trajectory(per_seed: list[dict], save_path: Path):
    ckpt_steps = per_seed[0]["checkpoint_steps"]
    alphas_stack = np.stack([r["alphas"] for r in per_seed], axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    # Individual seeds
    for r in per_seed:
        ax.plot(ckpt_steps, r["alphas"], color=FWS_COLOUR, alpha=0.30, linewidth=1.0)
    med = np.nanmedian(alphas_stack, axis=0)
    q25 = np.nanquantile(alphas_stack, 0.25, axis=0)
    q75 = np.nanquantile(alphas_stack, 0.75, axis=0)
    ax.plot(ckpt_steps, med, color=FWS_COLOUR, linewidth=2.2, label="median across seeds")
    ax.fill_between(ckpt_steps, q25, q75, color=FWS_COLOUR, alpha=0.18, label="IQR")
    ax.axhspan(1.4, 1.7, color=WONG_PALETTE[3], alpha=0.18, label="trained-net α band [1.4, 1.7]")
    ax.set_yscale("log")
    ax.set(xlabel="outer step", ylabel="HT-SR α (W1)",
           title=f"Phase 6: FWS-arm α trajectory (K={K_SEED}, winning cell ω₀=100, steps=5000)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_loss_trajectories(per_seed: list[dict], save_path: Path):
    steps = np.arange(NUM_OUTER_STEPS)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    def _band(key, colour, label):
        stack = np.stack([r[key] for r in per_seed], axis=0)
        med = np.median(stack, axis=0)
        q25 = np.quantile(stack, 0.25, axis=0)
        q75 = np.quantile(stack, 0.75, axis=0)
        ax.plot(steps, med, color=colour, label=label, linewidth=1.5)
        ax.fill_between(steps, q25, q75, color=colour, alpha=0.18)

    _band("fws_loss_traj", FWS_COLOUR, "FWS")
    _band("w_loss_traj", W_MATCHED_COLOUR, "W matched (58)")
    _band("op_loss_traj", W_OVERPARAM_COLOUR, "W overparam (~8700)")
    ax.set(yscale="log", xlabel="outer step", ylabel="training MSE",
           title=f"Phase 6: training-loss trajectories (K={K_SEED}, median + IQR)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_paired_diffs(
    diffs_vs_matched: np.ndarray,
    ci_vs_matched: tuple[float, float],
    diffs_vs_overparam: np.ndarray,
    ci_vs_overparam: tuple[float, float],
    save_path: Path,
):
    med_m = float(np.median(diffs_vs_matched))
    med_o = float(np.median(diffs_vs_overparam))

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    # Two horizontal CI bars
    ax.errorbar(
        [med_m],
        [1.0],
        xerr=[[med_m - ci_vs_matched[0]], [ci_vs_matched[1] - med_m]],
        fmt="o",
        color=W_MATCHED_COLOUR,
        ecolor=W_MATCHED_COLOUR,
        elinewidth=2.5,
        capsize=8,
        markersize=10,
        label=f"FWS − W matched (median = {med_m:.3g})",
    )
    ax.errorbar(
        [med_o],
        [0.0],
        xerr=[[med_o - ci_vs_overparam[0]], [ci_vs_overparam[1] - med_o]],
        fmt="o",
        color=W_OVERPARAM_COLOUR,
        ecolor=W_OVERPARAM_COLOUR,
        elinewidth=2.5,
        capsize=8,
        markersize=10,
        label=f"FWS − W overparam (median = {med_o:.3g})",
    )
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--", label="zero (no difference)")
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["FWS vs W overparam", "FWS vs W matched"])
    ax.set_xlabel("paired test-MSE difference (FWS − baseline)")
    ax.set_title(f"Phase 6: bootstrap-BCa 95% CI on median paired test-MSE difference (K={K_SEED})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- Research log ----------------------------------------------------------
def median_iqr_min_max(xs: np.ndarray) -> tuple[float, float, float, float]:
    return float(np.median(xs)), float(np.quantile(xs, 0.75) - np.quantile(xs, 0.25)), float(xs.min()), float(xs.max())


def main() -> None:
    print("=" * 72)
    print("Phase 6 — K=10 confirmation of winning cell + paired Wilcoxon")
    print("=" * 72)
    print(f"Winning cell: ω₀=({OMEGA_FIRST}, {OMEGA_HIDDEN}), steps={NUM_OUTER_STEPS}, "
          f"G_lr={G_LR}, z_lr={Z_LR}")
    print(f"Three arms (all {NUM_OUTER_STEPS} steps): FWS, W matched (58), W overparam ({22 * OVERPARAM_P + 10})")
    print(f"K_seed = {K_SEED}; bootstrap n_resamples = {BOOTSTRAP_N_RESAMPLES}")
    print()

    per_seed: list[dict] = []
    nan_or_crash: list[str] = []
    t0 = time.time()

    for seed in range(K_SEED):
        t_seed = time.time()
        try:
            task_loss_fn, train_batch, test_batch = make_synthetic_task(seed)
            row = run_seed(seed, task_loss_fn, train_batch, test_batch)
            per_seed.append(row)
            if row["any_nan"]:
                nan_or_crash.append(f"seed={seed}: NaN in a loss trajectory")
            print(
                f"  seed {seed:2d}: fws={row['fws_test']:.6g}  w={row['w_test']:.6g}  "
                f"op={row['op_test']:.6g}  alpha_final={row['alphas'][-1]:.4g}  "
                f"any_nan={row['any_nan']}  ({time.time() - t_seed:.1f}s)"
            )
            # Liveness ping after first and last seeds
            if seed == 0:
                print(f"  --- first seed complete, total elapsed {time.time() - t0:.1f}s ---")
        except Exception as e:  # noqa: BLE001
            nan_or_crash.append(f"seed={seed}: CRASH — {type(e).__name__}: {e}")
            print(f"  seed {seed}: CRASH — {type(e).__name__}: {e}")

    print(f"\nAll seeds complete: {time.time() - t0:.1f}s total wall.")

    # --- Final test MSE distribution ---
    fws_tests = np.array([r["fws_test"] for r in per_seed])
    w_tests = np.array([r["w_test"] for r in per_seed])
    op_tests = np.array([r["op_test"] for r in per_seed])

    fws_stats = median_iqr_min_max(fws_tests)
    w_stats = median_iqr_min_max(w_tests)
    op_stats = median_iqr_min_max(op_tests)
    print()
    print("Final test MSE (verbatim):")
    print(f"  FWS:           median={fws_stats[0]:.6g}  IQR={fws_stats[1]:.6g}  min={fws_stats[2]:.6g}  max={fws_stats[3]:.6g}")
    print(f"  W matched:     median={w_stats[0]:.6g}  IQR={w_stats[1]:.6g}  min={w_stats[2]:.6g}  max={w_stats[3]:.6g}")
    print(f"  W overparam:   median={op_stats[0]:.6g}  IQR={op_stats[1]:.6g}  min={op_stats[2]:.6g}  max={op_stats[3]:.6g}")

    # --- Paired tests ---
    W_m, p_m, diffs_m = paired_wilcoxon(fws_tests, w_tests)
    W_o, p_o, diffs_o = paired_wilcoxon(fws_tests, op_tests)
    ci_m = bca_ci_median_diff(diffs_m, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CI_LEVEL)
    ci_o = bca_ci_median_diff(diffs_o, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CI_LEVEL)
    hl_m = hodges_lehmann(diffs_m)
    hl_o = hodges_lehmann(diffs_o)
    med_diff_m = float(np.median(diffs_m))
    med_diff_o = float(np.median(diffs_o))

    print()
    print("Paired Wilcoxon results (verbatim):")
    print(f"  FWS - W matched:    W = {W_m:.6g}, p = {p_m:.6g}, median_diff = {med_diff_m:.6g}, "
          f"BCa 95% CI = ({ci_m[0]:.6g}, {ci_m[1]:.6g}), HL = {hl_m:.6g}")
    print(f"  FWS - W overparam:  W = {W_o:.6g}, p = {p_o:.6g}, median_diff = {med_diff_o:.6g}, "
          f"BCa 95% CI = ({ci_o[0]:.6g}, {ci_o[1]:.6g}), HL = {hl_o:.6g}")

    # --- Fraction in α band ---
    final_alphas = np.array([r["alphas"][-1] for r in per_seed])
    in_band = int(((final_alphas >= 1.4) & (final_alphas <= 1.7)).sum())
    print()
    print(f"Final α (W1) distribution: median={np.nanmedian(final_alphas):.4g}, "
          f"IQR={float(np.nanquantile(final_alphas, 0.75) - np.nanquantile(final_alphas, 0.25)):.4g}, "
          f"min={float(np.nanmin(final_alphas)):.4g}, max={float(np.nanmax(final_alphas)):.4g}")
    print(f"Fraction of seeds with final α in [1.4, 1.7]: {in_band}/{K_SEED}")

    # --- Hessian top eigenvalues ---
    hess_fws_med = float(np.median(np.array([r["hess_fws"] for r in per_seed])))
    hess_w_med = float(np.median(np.array([r["hess_w"] for r in per_seed])))
    hess_op_med = float(np.median(np.array([r["hess_op"] for r in per_seed])))

    # --- Plots ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    mse_box_png = FIGURES_DIR / "2026-06-06-phase6-mse-boxplot.png"
    alpha_traj_png = FIGURES_DIR / "2026-06-06-phase6-alpha-trajectory.png"
    loss_traj_png = FIGURES_DIR / "2026-06-06-phase6-loss-trajectories.png"
    paired_diff_png = FIGURES_DIR / "2026-06-06-phase6-paired-differences.png"

    plot_mse_boxplot(per_seed, mse_box_png)
    plot_alpha_trajectory(per_seed, alpha_traj_png)
    plot_loss_trajectories(per_seed, loss_traj_png)
    plot_paired_diffs(diffs_m, ci_m, diffs_o, ci_o, paired_diff_png)
    print(f"Wrote plots to {FIGURES_DIR}")

    # --- Determine interpretation phrasing ---
    def _ci_excludes_zero(ci):
        return (ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)

    fws_vs_matched_excludes = _ci_excludes_zero(ci_m)
    fws_vs_overparam_excludes = _ci_excludes_zero(ci_o)

    if fws_vs_matched_excludes and med_diff_m < 0:
        phrase_m = (
            "the bootstrap-BCa 95% CI on the median paired test-MSE difference "
            "(FWS − W matched) excludes zero and the median is negative — "
            "FWS beats the matched-parameter baseline on this task with K=10 evidence."
        )
    elif fws_vs_matched_excludes and med_diff_m > 0:
        phrase_m = (
            "the bootstrap-BCa 95% CI on the median paired test-MSE difference "
            "(FWS − W matched) excludes zero and the median is positive — "
            "W matched beats FWS on this task with K=10 evidence."
        )
    else:
        phrase_m = (
            "the bootstrap-BCa 95% CI on the median paired test-MSE difference "
            "(FWS − W matched) includes zero — no K=10 evidence that FWS and W matched differ on this task."
        )

    if fws_vs_overparam_excludes and med_diff_o < 0:
        phrase_o = (
            "the bootstrap-BCa 95% CI on the median paired test-MSE difference "
            "(FWS − W overparam) excludes zero and the median is negative — "
            "FWS beats the over-parameterised baseline on this task."
        )
    elif fws_vs_overparam_excludes and med_diff_o > 0:
        phrase_o = (
            "the bootstrap-BCa 95% CI on the median paired test-MSE difference "
            "(FWS − W overparam) excludes zero and the median is positive — "
            "W overparam beats FWS on this task; a residual gap to the over-parameterised baseline remains."
        )
    else:
        phrase_o = (
            "the bootstrap-BCa 95% CI on the median paired test-MSE difference "
            "(FWS − W overparam) includes zero — FWS has closed the gap to the over-parameterised baseline "
            "within K=10 evidence."
        )

    # --- Research log ---
    nan_summary = (
        "\n".join(f"- {line}" for line in nan_or_crash) if nan_or_crash else "- none."
    )

    paired_table = (
        "| comparison | Wilcoxon W | p-value | median paired diff | BCa 95% CI (lower, upper) | Hodges-Lehmann |\n"
        "|---|---|---|---|---|---|\n"
        f"| FWS − W matched | {W_m:.6g} | {p_m:.6g} | {med_diff_m:.6g} | ({ci_m[0]:.6g}, {ci_m[1]:.6g}) | {hl_m:.6g} |\n"
        f"| FWS − W overparam | {W_o:.6g} | {p_o:.6g} | {med_diff_o:.6g} | ({ci_o[0]:.6g}, {ci_o[1]:.6g}) | {hl_o:.6g} |"
    )
    mse_table = (
        "| arm | median | IQR | min | max |\n"
        "|---|---|---|---|---|\n"
        f"| FWS | {fws_stats[0]:.6g} | {fws_stats[1]:.6g} | {fws_stats[2]:.6g} | {fws_stats[3]:.6g} |\n"
        f"| W matched ({TOTAL_W_PARAMS}) | {w_stats[0]:.6g} | {w_stats[1]:.6g} | {w_stats[2]:.6g} | {w_stats[3]:.6g} |\n"
        f"| W overparam (~{22 * OVERPARAM_P + 10}) | {op_stats[0]:.6g} | {op_stats[1]:.6g} | {op_stats[2]:.6g} | {op_stats[3]:.6g} |"
    )
    per_seed_rows = "\n".join(
        f"| {r['seed']} | {r['fws_test']:.6g} | {r['w_test']:.6g} | {r['op_test']:.6g} | "
        f"{r['alphas'][-1]:.4g} | {r['hess_fws']:.4g} | {r['hess_w']:.4g} | {r['hess_op']:.4g} |"
        for r in per_seed
    )

    # Phase 7 phrasing
    if fws_vs_matched_excludes and med_diff_m < 0:
        next_phase = (
            "Phase 7 should test transfer of the winning cell to different task families: "
            "deeper teacher (e.g. 3-layer), different non-linearity (tanh, GELU), and "
            "classification rather than regression. If FWS beat W matched on the SiLU teacher-student "
            "but loses on a 3-layer teacher, the result is task-specific rather than a property of FWS."
        )
    else:
        next_phase = (
            "Phase 7 should pivot to architectural changes: per-leaf $G$ (one SIREN body per leaf instead "
            "of a shared body with leaf-one-hot), alternative basis families from ondes (H-SIREN, WIRE, "
            "Fourier features), or a different conditioning scheme than FiLM."
        )

    md = f"""# Phase 6 — winning cell confirmation, K=10 + paired Wilcoxon — 2026-06-06

## Background

Phase 5 (commits `c8f1722` fws-bench, `ecbd37d` fws — local) identified
$(\\omega_0 = 100, \\text{{steps}} = 5000, \\text{{lr scheme}} = \\text{{matched}})$ as the
cell achieving median final $\\alpha(W_1) = 1.456$ — inside the trained-network band
$[1.4, 1.7]$ — and median test MSE $3.55 \\times 10^{{-4}}$ at $K=3$.
Phase 6 confirms at $K=10$ with the fws empirical-card statistical convention
(paired Wilcoxon signed-rank test + bootstrap-BCa 95% CI on the median paired difference).

## Setup

- **Task**: 2-layer SiLU MLP teacher-student (in=4, hidden=8, out=2), 200 train + 50 test, full-batch MSE loss. Same as phases 4–5.
- **FWS arm**: `ondes.SIREN` (in_dim={COORD_DIM}, hidden_dim={HIDDEN_DIM_SIREN}, layers={NUM_HIDDEN_LAYERS}) with `omega_first={OMEGA_FIRST}`, `omega_hidden={OMEGA_HIDDEN}`; FiLM($z$) with $z \\in \\mathbb{{R}}^{{{DIM_Z}}}$. `optax.multi_transform` with `Adam({G_LR})` on $G$ and `Adam({Z_LR})` on $z$.
- **W matched arm**: direct training of the {TOTAL_W_PARAMS} student scalars with Adam(1e-3).
- **W overparam arm**: each $W_k = A_{{2,k}} A_{{1,k}}$ with $P = {OVERPARAM_P}$ (total $22 P + 10 = {22 * OVERPARAM_P + 10}$ scalars), biases 1-D. Adam(1e-3). Same $P$ as phase 4.
- All three arms run at `num_outer_steps = {NUM_OUTER_STEPS}`. Note: the W arms now have 5× more steps than phase 4, so direct cross-phase comparisons are muddied — phase 4's W numbers are not the relevant baseline here. The relevant baseline is the within-phase paired comparison reported below.
- $K_{{\\text{{seed}}}} = {K_SEED}$.
- Statistical tests: paired Wilcoxon signed-rank (two-sided, exact method, zero-method "wilcox") and bootstrap-BCa 95% CI on the median paired difference with $n_{{\\text{{resamples}}}} = {BOOTSTRAP_N_RESAMPLES}$, plus Hodges-Lehmann location estimator.

## Results

### Final test MSE distribution

{mse_table}

### Paired Wilcoxon results

{paired_table}

**Interpretation (using "CI excludes zero" as the gate, since the FWS empirical-card convention treats the bootstrap-BCa CI as the load-bearing test, not the Wilcoxon $p$-value alone):**

- FWS vs W matched: {phrase_m}
- FWS vs W overparam: {phrase_o}

### Per-seed table

| seed | FWS test | W matched test | W overparam test | final α(W1) | hess top FWS | hess top W matched | hess top W overparam |
|------|----------|----------------|-------------------|-------------|---------------|---------------------|----------------------|
{per_seed_rows}

### Final α(W1) distribution (FWS arm)

- median = {float(np.nanmedian(final_alphas)):.4g}
- IQR = {float(np.nanquantile(final_alphas, 0.75) - np.nanquantile(final_alphas, 0.25)):.4g}
- min = {float(np.nanmin(final_alphas)):.4g}
- max = {float(np.nanmax(final_alphas)):.4g}
- fraction of seeds finishing inside $[1.4, 1.7]$: **{in_band}/{K_SEED}**

The horizontal band on the α plot is reference, not a gate; the per-seed numbers above are the load-bearing artefact.

### Hessian top eigenvalue at convergence (median across seeds)

- FWS: {hess_fws_med:.6g}
- W matched: {hess_w_med:.6g}
- W overparam: {hess_op_med:.6g}

All three arms now run for the same {NUM_OUTER_STEPS} steps, so these are like-for-like.

### Plots

![α trajectory (FWS arm)](figures/2026-06-06-phase6-alpha-trajectory.png)

![final test MSE distribution](figures/2026-06-06-phase6-mse-boxplot.png)

![loss trajectories](figures/2026-06-06-phase6-loss-trajectories.png)

![paired differences with bootstrap-BCa 95% CI](figures/2026-06-06-phase6-paired-differences.png)

## NaN / crash report

{nan_summary}

## What the data shows

The verbatim numbers above are the primary artefact. The two paired comparisons are interpreted using the bootstrap-BCa 95% CI on the median paired difference — the FWS empirical-card convention. Both phrasings (FWS vs W matched, FWS vs W overparam) are stated above and follow strictly from whether the CI excludes zero, without rhetorical hedging. The Wilcoxon $p$-value is reported for completeness but is not the gate (with $K=10$ the exact-method Wilcoxon is necessarily coarse; the BCa CI is the convention-load-bearing test).

## Honest caveats

- **W arms have 5× more steps than phase 4** ({NUM_OUTER_STEPS} vs 1 000 in phase 4). This matches the FWS step budget but means the W-matched / W-overparam test MSEs here are not directly comparable to phase 4's. The within-phase paired comparison is unaffected by this; cross-phase narratives are.
- **Single task family**: 2-layer SiLU teacher-student with fixed dimensions (in=4, hidden=8, out=2). No transfer test to other architectures, depths, non-linearities, or classification objectives. A confirmation on this single task family is not evidence that FWS generalises.
- $K_{{\\text{{seed}}}} = {K_SEED}$ is the falsifier-convention floor; not a population-scale study. The bootstrap-BCa CI is the right tool for this regime, but it is not a substitute for a larger $K$.
- **Hyperparameter search was over 12 cells**: the winning cell may not generalise to other task families. If phase 7 picks a different teacher and the same hyperparameter cell fails, that is informative about the brittleness of the cell, not the FWS construction.
- $\\sigma$-spectrum probes are from a top-{SIGMA_TOP_K} power iteration on the renderer Jacobian; $\\sigma_{{\\min}}$ is the smallest of the top {SIGMA_TOP_K}, not the global minimum.
- Hessian top eigenvalue likewise comes from a {HESSIAN_POWER_ITERS}-step power iteration; the reported numbers are estimates, not exact eigenvalues — interpret as orders of magnitude.
- HT-SR α is fit on a small-spectrum problem ($W_1 \\in \\mathbb{{R}}^{{{HIDDEN_DIM_STUDENT} \\times {IN_DIM}}}$, 4 eigenvalues); the R² column from phases 4–5 is the relevant fit-quality diagnostic.
- All three arms use Adam(1e-3); no optimiser sweep.

## What's next

{next_phase}
"""

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_FILE.write_text(md)
    print(f"Wrote research log to {RESEARCH_FILE}")


if __name__ == "__main__":
    main()
