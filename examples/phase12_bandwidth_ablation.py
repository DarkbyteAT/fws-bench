"""Phase 12 — Bandwidth (ω₀) ablation on L2 distillation.

Phase 11B found case 3 (high L2 + low fidelity) for every (arm, loss) at
$\\omega_0 = 30$, $\\text{hidden\\_dim} = 8$, $\\text{depth} = 2$ on
both arms. That run used the canonical SIREN ω₀ from phases 8/9/10. The
user's open question: maybe the per-leaf $G_{\\text{leaf}}$'s bandwidth
is structurally too low to express trained CIFAR weights. If trained
conv/fc kernels carry spectral content above what an
$\\omega_0 = 30, H = 8$ SIREN can represent, phase 11B's
high-L2 ceiling is a *representation* limit baked into the bandwidth
knob, not an outer-optimisation failure or an architecture-shape
failure.

Phase 12 tests this with a single-axis sweep on $\\omega_0$. Hidden-dim
and depth sweeps are deferred — if $\\omega_0$ shows no signal, the
bandwidth hypothesis is dropped and the next phase moves to alternative
fixes (FiLM, alternative ondes basis, skip connections).

Cells (held: phase 11B Adam(1e-3) × 5000 outer steps × leaf-mean loss):

- **FWS-parallel** — $\\omega_0^{G_\\text{leaf}} \\in \\{30, 100, 300\\}$
  (no $G_H$).
- **FWS-hyper, $G_\\text{leaf}$-side** — vary
  $\\omega_0^{G_\\text{leaf}} \\in \\{30, 100, 300\\}$ holding
  $\\omega_0^{G_H} = 30$.
- **FWS-hyper, $G_H$-side** — vary
  $\\omega_0^{G_H} \\in \\{100, 300\\}$ holding
  $\\omega_0^{G_\\text{leaf}} = 30$. The
  $(\\omega_0^{G_\\text{leaf}} = 30, \\omega_0^{G_H} = 30)$ baseline
  is shared with the $G_\\text{leaf}$-side sweep above.

Eight cells in total × $K = 3$ seeds.

**Pre-registered escalation rule (box non-overlap).** For each arm
family the "winner" cell is identified only when its $K=3$ median
functional fidelity exceeds the runner-up's median by at least
$\\max(\\text{IQR}_\\text{winner}, \\text{IQR}_\\text{runner-up})$. If
boxes overlap, no cell is selected at $K=3$ and the top-2 cells
escalate to $K=10$. Recorded before reading the numbers.

The leaf-mean loss (NeRN-style) is used throughout: phase 11B showed
identical ceilings for leaf-mean and layer-wise normalised, so the
second loss costs wall-time without informing the bandwidth axis.

Run::

    uv run python examples/phase12_bandwidth_ablation.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float


sys.path.insert(0, str(Path(__file__).parent))
from _common import data, diagnostics, mainnet, reporting  # noqa: E402
from _common.arms import (  # noqa: E402
    DIM_Z,
    FWS_HYPER_COLOUR,
    FWS_PARALLEL_COLOUR,
    G_LEAF_HIDDEN_DIM,
    G_LEAF_PARAM_SIZE,
    LEAF_COORDS,
    MAX_G_LEAF_PARAM_SIZE,
    kaiming_leaf_scale,
    slice_g_leaf_flat,
)

# Reuse phase 10's polar projection and phase 11B's distillation machinery
# verbatim — phase 12 changes ω₀, not the projection or the loss.
from phase10_polar_projection import project_pseudo_orthonormal  # noqa: E402
from phase11b_supervised_distillation import (  # noqa: E402
    DISTILL_LR,
    DISTILL_STEPS,
    K_SEED,
    LOG_EVERY,
    TARGET_BATCH_SIZE,
    TARGET_EPOCHS,
    eval_test_loss_acc,
    leaf_weights,
    mse_to_target,
    per_leaf_l2_sse,
    train_target_cnn,
)

from fws_bench import Arm  # noqa: E402


RESEARCH_DIR = Path("/Users/ammar/Documents/Research/fractal-weight-spaces/research")
FIGURES_DIR = RESEARCH_DIR / "figures"
DATE = os.environ.get("PHASE12_DATE", "2026-06-08")
RESEARCH_FILE = RESEARCH_DIR / f"{DATE}-phase12-bandwidth-ablation.md"
FIG_PREFIX = f"{DATE}-phase12"


# --- Phase-12 G_H constants (identical to phase 10's) ----------------------
DIM_LEAF_EMB = 8
G_H_HIDDEN_DIM = 100
G_H_DEPTH = 3  # sine layers + 1 linear readout


# --- ω₀-parameterised G_leaf forward ---------------------------------------
def g_leaf_forward_omega(
    coord: Array, params: dict[str, Array], *, omega_0: float,
    film: Array | None = None,
) -> Array:
    """Sine-only G_leaf forward with a custom ω₀.

    Mirrors :func:`_common.arms.g_leaf_forward` but threads ``omega_0``
    through the two sine layers instead of using the module-level
    constant. The output activation is the final ``sin(pre_out[0])`` —
    phase 11B / phase 10 shape, unchanged.
    """
    pre1 = params["W_in"] @ coord + params["b_in"]
    h1 = jnp.sin(omega_0 * pre1)
    pre2 = params["W_h"] @ h1 + params["b_h"]
    if film is not None:
        H = G_LEAF_HIDDEN_DIM
        gamma = film[:H]
        beta = film[H:]
        pre2 = gamma * pre2 + beta
    h2 = jnp.sin(omega_0 * pre2)
    pre_out = params["W_out"] @ h2 + params["b_out"]
    return jnp.sin(pre_out[0])


# --- ω₀-parameterised ParallelGLeaves (FWS-parallel arm) -------------------
class ParallelGLeavesOmega(eqx.Module):
    """ω₀-aware FWS-parallel G_leaves.

    The hidden-layer SIREN init bound is $\\sqrt{6/H} / \\omega_0$. The
    first-layer bound (``1 / rank``) is rank-dependent, not ω₀-dependent —
    matched to the canonical SIREN init scheme. Storing ``omega_0`` as a
    static field rather than a pytree leaf so it doesn't enter the optax
    parameter set.
    """

    g_leaf_rank1: dict[str, Array]
    g_leaf_rank2: dict[str, Array]
    g_leaf_rank4: dict[str, Array]
    film_W_rank1: Float[Array, "two_h dim_z"]
    film_b_rank1: Float[Array, " two_h"]
    film_W_rank2: Float[Array, "two_h dim_z"]
    film_b_rank2: Float[Array, " two_h"]
    film_W_rank4: Float[Array, "two_h dim_z"]
    film_b_rank4: Float[Array, " two_h"]
    omega_0: float = eqx.field(static=True)

    def __init__(self, *, key: Array, omega_0: float) -> None:
        keys = jax.random.split(key, 6)
        H = G_LEAF_HIDDEN_DIM
        sqrt_H = float(jnp.sqrt(jnp.array(H, dtype=jnp.float32)))
        self.omega_0 = omega_0

        def init_g_leaf(k: Array, rank: int) -> dict[str, Array]:
            kw_in, kb_in, kw_h, kb_h, kw_out, kb_out = jax.random.split(k, 6)
            bound_in = 1.0 / rank
            W_in = jax.random.uniform(kw_in, (H, rank), minval=-bound_in, maxval=bound_in)
            b_in = jax.random.uniform(kb_in, (H,), minval=-bound_in, maxval=bound_in)
            bound_h = float(jnp.sqrt(jnp.array(6.0 / H, dtype=jnp.float32))) / omega_0
            W_h = jax.random.uniform(kw_h, (H, H), minval=-bound_h, maxval=bound_h)
            b_h = jax.random.uniform(kb_h, (H,), minval=-bound_h, maxval=bound_h)
            W_out = jax.random.uniform(kw_out, (1, H), minval=-bound_h, maxval=bound_h)
            b_out = jax.random.uniform(kb_out, (1,), minval=-bound_h, maxval=bound_h)
            return {"W_in": W_in, "b_in": b_in, "W_h": W_h, "b_h": b_h,
                    "W_out": W_out, "b_out": b_out}

        self.g_leaf_rank1 = init_g_leaf(keys[0], 1)
        self.g_leaf_rank2 = init_g_leaf(keys[1], 2)
        self.g_leaf_rank4 = init_g_leaf(keys[2], 4)

        bound_film = 1.0 / sqrt_H
        kw, _ = jax.random.split(keys[3])
        self.film_W_rank1 = jax.random.uniform(kw, (2 * H, DIM_Z),
                                               minval=-bound_film, maxval=bound_film)
        self.film_b_rank1 = jnp.zeros((2 * H,))
        kw, _ = jax.random.split(keys[4])
        self.film_W_rank2 = jax.random.uniform(kw, (2 * H, DIM_Z),
                                               minval=-bound_film, maxval=bound_film)
        self.film_b_rank2 = jnp.zeros((2 * H,))
        kw, _ = jax.random.split(keys[5])
        self.film_W_rank4 = jax.random.uniform(kw, (2 * H, DIM_Z),
                                               minval=-bound_film, maxval=bound_film)
        self.film_b_rank4 = jnp.zeros((2 * H,))

    def g_leaf_for(self, rank: int) -> dict[str, Array]:
        return {1: self.g_leaf_rank1, 2: self.g_leaf_rank2, 4: self.g_leaf_rank4}[rank]

    def film_for(self, rank: int, z: Array) -> Array:
        W, b = {
            1: (self.film_W_rank1, self.film_b_rank1),
            2: (self.film_W_rank2, self.film_b_rank2),
            4: (self.film_W_rank4, self.film_b_rank4),
        }[rank]
        return W @ z + b


def make_fws_parallel_omega(
    *,
    omega_0_g_leaf: float,
    lr: float = DISTILL_LR,
    z_init_std: float = 0.1,
) -> Arm:
    """FWS-parallel-no-G_H with a per-cell ω₀ on G_leaf.

    Uses the phase-10 polar projection and the phase-10 Kaiming leaf
    pre-factor verbatim; the only knob this constructor exposes is
    G_leaf's ω₀.
    """
    import loom

    scale_fn = kaiming_leaf_scale
    proj_fn = project_pseudo_orthonormal

    def init(key: Array) -> dict:
        k_par, k_z = jax.random.split(key)
        return {
            "P": ParallelGLeavesOmega(key=k_par, omega_0=omega_0_g_leaf),
            "z": jax.random.normal(k_z, (DIM_Z,)) * z_init_std,
        }

    def render(state: dict) -> dict[str, Array]:
        P_tree = {name: jnp.zeros(shape) for name, shape in mainnet.LEAF_SHAPES.items()}

        def f(path, shape, dtype, params):
            par_in, z_in = params
            leaf_name = path[0].key
            rank = mainnet.LEAF_RANKS[leaf_name]
            scale = scale_fn(leaf_name)
            leaf_params = par_in.g_leaf_for(rank)
            film = par_in.film_for(rank, z_in)
            coords = LEAF_COORDS[leaf_name]
            values = jax.vmap(
                lambda c: g_leaf_forward_omega(
                    c, leaf_params, omega_0=par_in.omega_0, film=film,
                )
            )(coords)
            W_raw = scale * values.reshape(shape)
            return proj_fn(W_raw, leaf_name).astype(dtype)

        return loom.render(P_tree, f, (state["P"], state["z"]))

    def loss_fn(state: dict, batch: dict) -> Array:
        W = render(state)
        return mainnet.cross_entropy_loss(W, batch)

    return Arm(
        name=f"FWS-parallel ω₀={omega_0_g_leaf:g}",
        short=f"fws_par_w{omega_0_g_leaf:g}",
        color=FWS_PARALLEL_COLOUR,
        init=init,
        loss_fn=loss_fn,
        render_W=render,
        optimiser=optax.adam(lr),
        diagnostics_at_convergence=None,
    )


# --- ω₀-parameterised HyperRenderer (FWS-hyper arm) ------------------------
class HyperRendererOmega(eqx.Module):
    """Phase 10 G_H with configurable hidden-layer ω₀.

    Same shape as :class:`phase10_polar_projection.HyperRenderer`:
    SIREN body + linear readout + per-leaf z-score normalisation of the
    output slice. ``omega_0`` is a static field — pytree-stable across
    optax updates.
    """

    leaf_embedding: Float[Array, "n_leaves dim_leaf_emb"]
    W_in: Float[Array, "hidden in_total"]
    b_in: Float[Array, " hidden"]
    W_h: tuple[Float[Array, "hidden hidden"], ...]
    b_h: tuple[Float[Array, " hidden"], ...]
    W_out: Float[Array, "g_leaf_param_size hidden"]
    b_out: Float[Array, " g_leaf_param_size"]
    omega_0: float = eqx.field(static=True)

    def __init__(self, *, key: Array, omega_0: float) -> None:
        keys = jax.random.split(key, 5 + 2 * (G_H_DEPTH - 2))
        k_emb, k_w_in, k_b_in, k_w_out, k_b_out, *k_hidden = keys
        in_total = DIM_Z + DIM_LEAF_EMB
        self.omega_0 = omega_0

        emb_bound = 1.0 / jnp.sqrt(jnp.array(DIM_LEAF_EMB, dtype=jnp.float32))
        self.leaf_embedding = jax.random.uniform(
            k_emb, (mainnet.N_LEAVES, DIM_LEAF_EMB),
            minval=-emb_bound, maxval=emb_bound,
        )

        # First-layer SIREN init: bound = 1 / in_total.
        bound_first = 1.0 / in_total
        self.W_in = jax.random.uniform(k_w_in, (G_H_HIDDEN_DIM, in_total),
                                       minval=-bound_first, maxval=bound_first)
        self.b_in = jax.random.uniform(k_b_in, (G_H_HIDDEN_DIM,),
                                       minval=-bound_first, maxval=bound_first)

        # Hidden SIREN layers: bound = sqrt(6/H) / omega_0.
        bound_hidden = float(
            jnp.sqrt(jnp.array(6.0 / G_H_HIDDEN_DIM, dtype=jnp.float32))
        ) / omega_0
        n_hidden = G_H_DEPTH - 2
        W_h_list: list[Array] = []
        b_h_list: list[Array] = []
        for j in range(n_hidden):
            kw, kb = k_hidden[2 * j], k_hidden[2 * j + 1]
            W_h_list.append(jax.random.uniform(
                kw, (G_H_HIDDEN_DIM, G_H_HIDDEN_DIM),
                minval=-bound_hidden, maxval=bound_hidden))
            b_h_list.append(jax.random.uniform(
                kb, (G_H_HIDDEN_DIM,), minval=-bound_hidden, maxval=bound_hidden))
        self.W_h = tuple(W_h_list)
        self.b_h = tuple(b_h_list)

        # Linear readout: canonical Xavier 1 / sqrt(hidden_dim) (phase 10's
        # default; ω₀ does not enter — the output layer is linear).
        bound_out = 1.0 / float(jnp.sqrt(jnp.array(G_H_HIDDEN_DIM, dtype=jnp.float32)))
        self.W_out = jax.random.uniform(
            k_w_out, (MAX_G_LEAF_PARAM_SIZE, G_H_HIDDEN_DIM),
            minval=-bound_out, maxval=bound_out)
        self.b_out = jax.random.uniform(k_b_out, (MAX_G_LEAF_PARAM_SIZE,),
                                        minval=-bound_out, maxval=bound_out)

    def produce_flat(self, z: Array, leaf_id: int) -> Array:
        emb = self.leaf_embedding[leaf_id]
        inp = jnp.concatenate([z, emb])
        h = jnp.sin(self.omega_0 * (self.W_in @ inp + self.b_in))
        for W, b in zip(self.W_h, self.b_h, strict=True):
            h = jnp.sin(self.omega_0 * (W @ h + b))
        return self.W_out @ h + self.b_out

    def produce(self, z: Array, leaf_id: int, rank: int) -> dict[str, Array]:
        flat = self.produce_flat(z, leaf_id)
        slice_size = G_LEAF_PARAM_SIZE[rank]
        leaf_slice = flat[:slice_size]
        eps = jnp.finfo(leaf_slice.dtype).eps * slice_size
        mean = jnp.mean(leaf_slice)
        std = jnp.std(leaf_slice) + eps
        target_std = jnp.sqrt(jnp.array(2.0 / slice_size, dtype=leaf_slice.dtype))
        normalised = (leaf_slice - mean) / std * target_std
        return slice_g_leaf_flat(normalised, rank)


def make_fws_hyper_omega(
    *,
    omega_0_g_leaf: float,
    omega_0_g_h: float,
    g_lr: float = DISTILL_LR,
    z_lr: float = DISTILL_LR,
    z_init_std: float = 0.1,
) -> Arm:
    """FWS-hyper-polar with per-cell ω₀ on both G_leaf and G_H.

    Phase-10 polar projection + Kaiming leaf pre-factor; the only knobs
    are the two ω₀ values. The G_H output slice is z-scored per leaf
    (phase 10's fix), so changing ω₀_G_H affects how G_H *internally*
    represents leaf-specific information but not the final G_leaf
    parameter magnitudes.
    """
    import loom

    scale_fn = kaiming_leaf_scale
    proj_fn = project_pseudo_orthonormal

    def init(key: Array) -> dict:
        k_g, k_z = jax.random.split(key)
        return {
            "G": HyperRendererOmega(key=k_g, omega_0=omega_0_g_h),
            "z": jax.random.normal(k_z, (DIM_Z,)) * z_init_std,
        }

    def render(state: dict) -> dict[str, Array]:
        P_tree = {name: jnp.zeros(shape) for name, shape in mainnet.LEAF_SHAPES.items()}

        def f(path, shape, dtype, params):
            G_H_in, z_in = params
            leaf_name = path[0].key
            leaf_id = mainnet.LEAF_ORDER.index(leaf_name)
            rank = mainnet.LEAF_RANKS[leaf_name]
            scale = scale_fn(leaf_name)
            leaf_params = G_H_in.produce(z_in, leaf_id, rank)
            coords = LEAF_COORDS[leaf_name]
            values = jax.vmap(
                lambda c: g_leaf_forward_omega(c, leaf_params, omega_0=omega_0_g_leaf)
            )(coords)
            W_raw = scale * values.reshape(shape)
            return proj_fn(W_raw, leaf_name).astype(dtype)

        return loom.render(P_tree, f, (state["G"], state["z"]))

    def loss_fn(state: dict, batch: dict) -> Array:
        W = render(state)
        return mainnet.cross_entropy_loss(W, batch)

    optimiser = optax.multi_transform(
        {"G": optax.adam(g_lr), "z": optax.adam(z_lr)},
        {"G": "G", "z": "z"},
    )
    return Arm(
        name=f"FWS-hyper ω₀ᴳ={omega_0_g_leaf:g}/ᴴ={omega_0_g_h:g}",
        short=f"fws_hyp_gl{omega_0_g_leaf:g}_gh{omega_0_g_h:g}",
        color=FWS_HYPER_COLOUR,
        init=init,
        loss_fn=loss_fn,
        render_W=render,
        optimiser=optimiser,
        diagnostics_at_convergence=None,
    )


# --- Cell definitions ------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    cell_id: str           # short tag for tables / plots
    arm_family: str        # "parallel" | "hyper"
    omega_0_g_leaf: float
    omega_0_g_h: float | None  # None for FWS-parallel
    axis: str              # which axis varies — "g_leaf" | "g_h"


CELLS: list[Cell] = [
    # FWS-parallel: ω₀_G_leaf axis
    Cell("par-w30",   "parallel", 30.0,  None, "g_leaf"),
    Cell("par-w100",  "parallel", 100.0, None, "g_leaf"),
    Cell("par-w300",  "parallel", 300.0, None, "g_leaf"),
    # FWS-hyper: G_leaf side (G_H ω₀ held at 30)
    Cell("hyp-gl30-gh30",   "hyper", 30.0,  30.0,  "g_leaf"),
    Cell("hyp-gl100-gh30",  "hyper", 100.0, 30.0,  "g_leaf"),
    Cell("hyp-gl300-gh30",  "hyper", 300.0, 30.0,  "g_leaf"),
    # FWS-hyper: G_H side (G_leaf ω₀ held at 30; gl30-gh30 baseline shared above)
    Cell("hyp-gl30-gh100",  "hyper", 30.0,  100.0, "g_h"),
    Cell("hyp-gl30-gh300",  "hyper", 30.0,  300.0, "g_h"),
]


def make_arm_for_cell(cell: Cell) -> Arm:
    if cell.arm_family == "parallel":
        return make_fws_parallel_omega(omega_0_g_leaf=cell.omega_0_g_leaf)
    if cell.arm_family == "hyper":
        assert cell.omega_0_g_h is not None
        return make_fws_hyper_omega(
            omega_0_g_leaf=cell.omega_0_g_leaf,
            omega_0_g_h=cell.omega_0_g_h,
        )
    raise ValueError(f"unknown arm_family {cell.arm_family!r}")


# --- Distillation (leaf-mean only, reusing phase 11B's machinery) ---------
LOSS_KIND = "leaf_mean"


def distill_arm_for_cell(
    arm: Arm, target_W: dict[str, Array],
    *, seed: int, num_steps: int, lr: float,
) -> tuple[Any, list[tuple[int, float]], dict[str, Array]]:
    """L2-distill ``arm`` against ``target_W`` for ``num_steps`` Adam steps.

    Carbon-copy of phase 11B's ``distill_arm`` with the loss kind pinned
    to ``leaf_mean`` and the inner JIT closed over a phase-12 arm. The
    duplication keeps phase 11B's machinery untouched while letting
    phase 12 close over its own (per-cell) arm reference.
    """
    key = jax.random.key(seed)
    state = arm.init(key)
    optimiser = optax.adam(lr)
    opt_state = optimiser.init(state)
    weights = leaf_weights(target_W, LOSS_KIND)

    def loss_fn(state: Any) -> Array:
        rendered = arm.render_W(state)
        return mse_to_target(rendered, target_W, weights=weights)

    @jax.jit
    def step(state: Any, opt_state: Any) -> tuple[Any, Any, Array]:
        loss, grads = jax.value_and_grad(loss_fn)(state)
        updates, new_opt = optimiser.update(grads, opt_state, state)
        return optax.apply_updates(state, updates), new_opt, loss

    trajectory: list[tuple[int, float]] = []
    for s in range(num_steps):
        state, opt_state, loss = step(state, opt_state)
        if s % LOG_EVERY == 0 or s == num_steps - 1:
            trajectory.append((s, float(loss)))
    rendered = {k: np.asarray(v) for k, v in arm.render_W(state).items()}
    return state, trajectory, rendered


# --- Box-non-overlap escalation rule (pre-registered) ----------------------
def box_non_overlap_verdict(
    func_acc_per_cell: dict[str, list[float]],
    *, family_cells: list[Cell],
) -> str:
    """Apply the pre-registered box-non-overlap rule within a family.

    A cell is the family winner iff its median functional fidelity
    exceeds the runner-up's by ≥ max(IQR_winner, IQR_runner_up).
    Otherwise: no winner at K=3, top-2 escalate to K=10.
    """
    if len(family_cells) < 2:
        return "Only one cell in family — no comparison."
    stats: list[tuple[Cell, float, float]] = []
    for c in family_cells:
        v = np.asarray(func_acc_per_cell[c.cell_id], dtype=float)
        med = float(np.median(v))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        stats.append((c, med, iqr))
    stats.sort(key=lambda t: t[1], reverse=True)
    winner, w_med, w_iqr = stats[0]
    runner, r_med, r_iqr = stats[1]
    gap = w_med - r_med
    threshold = max(w_iqr, r_iqr)
    if gap >= threshold and threshold > 0.0:
        return (
            f"Winner: **{winner.cell_id}** (median {w_med:.4f}). "
            f"Runner-up: {runner.cell_id} (median {r_med:.4f}). "
            f"Gap {gap:.4f} ≥ max(IQR) {threshold:.4f} — box-non-overlap rule fires."
        )
    return (
        f"No winner at K=3. Top-2 by median: {winner.cell_id} ({w_med:.4f}) and "
        f"{runner.cell_id} ({r_med:.4f}); gap {gap:.4f} < max(IQR) "
        f"{threshold:.4f}. Escalate to K=10 if pursuing further."
    )


def best_cell_in_family(
    func_acc_per_cell: dict[str, list[float]],
    *, family_cells: list[Cell],
) -> tuple[str, float, float, float, float]:
    """Return ``(cell_id, median, iqr, min, max)`` of the highest-median cell.

    Threshold-free: returns the distribution summary of the best-median
    cell in the family. Whether the lift is "real" against the chance
    baseline of 0.10 is left to the reader against the verbatim
    per-cell distribution.
    """
    best_cell = family_cells[0]
    best_med = -float("inf")
    for c in family_cells:
        med = float(np.median(np.asarray(func_acc_per_cell[c.cell_id], dtype=float)))
        if med > best_med:
            best_med = med
            best_cell = c
    v = np.asarray(func_acc_per_cell[best_cell.cell_id], dtype=float)
    return (
        best_cell.cell_id, float(np.median(v)),
        float(np.percentile(v, 75) - np.percentile(v, 25)),
        float(v.min()), float(v.max()),
    )


# --- Plotting --------------------------------------------------------------
def plot_l2_and_fidelity(
    cells: list[Cell],
    final_loss: dict[str, list[float]],
    func_acc: dict[str, list[float]],
    target_acc: float,
    save_path: Path,
    *, title: str,
) -> None:
    """Two-panel figure: per-cell box of (final L2, functional fidelity)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))
    cell_ids = [c.cell_id for c in cells]
    cols = [FWS_PARALLEL_COLOUR if c.arm_family == "parallel" else FWS_HYPER_COLOUR
            for c in cells]
    rng = np.random.default_rng(0)
    for ax, key, ylabel, logy in (
        (axes[0], final_loss, "final leaf-mean L2 loss", True),
        (axes[1], func_acc, "functional fidelity (test acc)", False),
    ):
        data_lists = [np.asarray(key[cid], dtype=float) for cid in cell_ids]
        bp = ax.boxplot(
            data_lists, patch_artist=True, widths=0.55, showfliers=False,
            tick_labels=cell_ids,
        )
        for patch, c in zip(bp["boxes"], cols, strict=True):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
            patch.set_edgecolor(c)
        for ml, c in zip(bp["medians"], cols, strict=True):
            ml.set_color(c)
            ml.set_linewidth(2.0)
        for i, (vals, c) in enumerate(zip(data_lists, cols, strict=True), start=1):
            jitter = rng.uniform(-0.10, 0.10, size=vals.size)
            ax.scatter(np.full_like(vals, i) + jitter, vals, color=c, s=40, zorder=3,
                       edgecolor="white", linewidths=0.6)
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        if key is func_acc:
            ax.axhline(target_acc, color="#000000", linestyle="--", linewidth=1.5,
                       label=f"target ({target_acc:.3f})")
            ax.axhline(0.10, color="#888888", linestyle=":", linewidth=1.0, label="chance")
            ax.axhline(0.40, color="#888888", linestyle="-.", linewidth=1.0,
                       label="interp. threshold 0.40")
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_omega_axis(
    cells: list[Cell],
    final_loss: dict[str, list[float]],
    func_acc: dict[str, list[float]],
    target_acc: float,
    save_path: Path,
    *, title: str,
) -> None:
    """ω₀-axis × arm × (L2, fidelity) panels.

    Three columns per arm (FWS-parallel ω₀^G_leaf, FWS-hyper varying
    G_leaf, FWS-hyper varying G_H) × two rows (L2, fidelity). Within
    each panel, ω₀ is the x-axis; markers are per-seed.
    """
    import matplotlib.pyplot as plt

    groups: list[tuple[str, list[Cell], str, str]] = [
        ("FWS-par ω₀^G_leaf",
         [c for c in cells if c.arm_family == "parallel"],
         "omega_0_g_leaf", FWS_PARALLEL_COLOUR),
        ("FWS-hyper ω₀^G_leaf  (ω₀^G_H=30)",
         [c for c in cells if c.arm_family == "hyper" and c.axis == "g_leaf"],
         "omega_0_g_leaf", FWS_HYPER_COLOUR),
        ("FWS-hyper ω₀^G_H  (ω₀^G_leaf=30)",
         [c for c in cells if c.arm_family == "hyper"
          and (c.axis == "g_h" or c.cell_id == "hyp-gl30-gh30")],
         "omega_0_g_h", FWS_HYPER_COLOUR),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0))
    for col, (label, group_cells, axis_key, colour) in enumerate(groups):
        xs = [getattr(c, axis_key) for c in group_cells]
        for row, (data_dict, ylabel, logy, ref_chance) in enumerate((
            (final_loss, "final L2 loss", True, False),
            (func_acc, "functional fidelity", False, True),
        )):
            ax = axes[row, col]
            for c, x in zip(group_cells, xs, strict=True):
                vals = np.asarray(data_dict[c.cell_id], dtype=float)
                ax.scatter([x] * vals.size, vals, color=colour, s=44, alpha=0.6,
                           edgecolor="white", linewidths=0.6)
                ax.scatter([x], [float(np.median(vals))], color=colour, s=120,
                           marker="_", linewidths=3.0)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel(axis_key)
            ax.set_ylabel(ylabel)
            ax.set_title(label if row == 0 else "")
            ax.grid(True, which="both", alpha=0.3)
            if ref_chance:
                ax.axhline(target_acc, color="#000000", linestyle="--", linewidth=1.2,
                           label=f"target ({target_acc:.3f})")
                ax.axhline(0.10, color="#888888", linestyle=":", linewidth=1.0,
                           label="chance")
                ax.axhline(0.40, color="#888888", linestyle="-.", linewidth=1.0,
                           label="interp. threshold")
                ax.legend(loc="best", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- Reporting -------------------------------------------------------------
def per_cell_table_md(
    cells: list[Cell],
    final_loss: dict[str, list[float]],
    per_leaf_l2: dict[str, list[dict[str, float]]],
    func_acc: dict[str, list[float]],
    target_acc: float,
) -> str:
    """Eight-cell distribution table: medians + IQR + min/max for L2 and fidelity."""
    rows: list[str] = []
    rows.append(
        "| cell | arm | ω₀^G_leaf | ω₀^G_H | L2 median | L2 IQR | "
        "L2 min | L2 max | fidelity median | fidelity IQR | fidelity min | fidelity max |"
    )
    rows.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        L = np.asarray(final_loss[c.cell_id], dtype=float)
        F = np.asarray(func_acc[c.cell_id], dtype=float)
        L_iqr = float(np.percentile(L, 75) - np.percentile(L, 25))
        F_iqr = float(np.percentile(F, 75) - np.percentile(F, 25))
        gh = "—" if c.omega_0_g_h is None else f"{c.omega_0_g_h:g}"
        rows.append(
            f"| {c.cell_id} | {c.arm_family} | {c.omega_0_g_leaf:g} | {gh} "
            f"| {np.median(L):.4e} | {L_iqr:.4e} | {L.min():.4e} | {L.max():.4e} "
            f"| {np.median(F):.4f} | {F_iqr:.4f} | {F.min():.4f} | {F.max():.4f} |"
        )
    return ("Target CNN test accuracy: " + f"**{target_acc:.4f}**\n\n"
            + "\n".join(rows))


def per_leaf_l2_table_md(
    cells: list[Cell],
    per_leaf_l2: dict[str, list[dict[str, float]]],
) -> str:
    """Per-leaf raw L2 (sum-of-squares) medians per cell."""
    rows: list[str] = []
    hdr = "| cell | " + " | ".join(mainnet.LEAF_ORDER) + " |"
    sep = "|---|" + "|".join(["---"] * len(mainnet.LEAF_ORDER)) + "|"
    rows.extend([hdr, sep])
    for c in cells:
        medians = {
            leaf: float(np.median([d[leaf] for d in per_leaf_l2[c.cell_id]]))
            for leaf in mainnet.LEAF_ORDER
        }
        body = " | ".join(f"{medians[leaf]:.3e}" for leaf in mainnet.LEAF_ORDER)
        rows.append(f"| {c.cell_id} | {body} |")
    return "\n".join(rows)


def alpha_table_md(
    cells: list[Cell],
    alphas_target: dict[str, tuple[float, float]],
    alphas_per_cell: dict[str, list[dict[str, tuple[float, float]]]],
) -> str:
    leaves = ("conv1", "conv2", "fc1", "fc2")
    hdr = "| cell | " + " | ".join(f"{leaf} α (R²)" for leaf in leaves) + " |"
    sep = "|---|" + "|".join(["---"] * len(leaves)) + "|"
    rows = [hdr, sep]
    target_cells = " | ".join(
        f"{alphas_target[leaf][0]:.3f} ({alphas_target[leaf][1]:.3f})"
        for leaf in leaves
    )
    rows.append(f"| **target** | {target_cells} |")
    for c in cells:
        leaf_strs: list[str] = []
        for leaf in leaves:
            a_med = float(np.median([v[leaf][0] for v in alphas_per_cell[c.cell_id]]))
            r2_med = float(np.median([v[leaf][1] for v in alphas_per_cell[c.cell_id]]))
            leaf_strs.append(f"{a_med:.3f} ({r2_med:.3f})")
        rows.append(f"| {c.cell_id} | " + " | ".join(leaf_strs) + " |")
    return "\n".join(rows)


# --- Main ------------------------------------------------------------------
_BACKGROUND_MD = """
Phase 11B (fws-bench `8c3c1c7`, fws `1678cfa`) ran direct L2
distillation against a trained CIFAR-10 ``WideKernelCNN-SiLU`` and found
case 3 of the pre-registered grid (high L2 + low fidelity) for every
(arm, loss) at the canonical SIREN bandwidth $\\omega_0 = 30$, hidden
dim $H = 8$, depth $= 2$. Functional fidelity for FWS-hyper-polar sat
at $\\approx 0.10$ (chance) on CIFAR-10; FWS-parallel-no-$G_H$ sat at
$\\approx 0.08$.

The pre-registered verdict labelled this a robust pivot signal — but
that conclusion holds only if bandwidth is not the load-bearing knob.
A per-leaf $G_\\text{leaf}$ SIREN at $\\omega_0 = 30, H = 8, \\text{depth}=2$
is deliberately compressive (phases 8/9/10 used those constants to
enforce a "compress not memorise" invariant). If trained CIFAR
classifier weights carry spectral content above what that
$G_\\text{leaf}$ can express, distillation hits a representation
ceiling at the bandwidth knob, not at the architecture shape.

Phase 12 tests the bandwidth hypothesis directly: sweep $\\omega_0$
along three cells per arm side, holding everything else from phase 11B
(Adam(1e-3) × 5000 outer steps, leaf-mean loss, polar projection,
Kaiming pre-factor). One axis only ($\\omega_0$); hidden-dim and depth
sweeps are deferred to phase 13 if $\\omega_0$ shows signal.
"""

_SETUP_MD = """
- **Target tree $W^{{\\star}}$**: ``WideKernelCNN-SiLU`` trained on
  CIFAR-10 with Adam(1e-3) for {target_epochs} epochs (seed 0,
  deterministic with phase 11B). One target, all cells.
- **Distillation**: each (cell, seed) trains the FWS arm's state via
  Adam({distill_lr}) for {distill_steps} outer steps on the leaf-mean
  L2 loss (NeRN-default) against $W^{{\\star}}$. The layer-wise
  normalised cell is **not** rerun — phase 11B showed identical
  ceilings.
- **K = {k_seed} seeds per cell.** Eight cells total:
  three for FWS-parallel-no-$G_H$ ($\\omega_0^{{G_\\text{{leaf}}}}
  \\in \\{{30, 100, 300\\}}$), three for FWS-hyper-polar varying
  $\\omega_0^{{G_\\text{{leaf}}}} \\in \\{{30, 100, 300\\}}$ at
  $\\omega_0^{{G_H}} = 30$, and two more for FWS-hyper-polar varying
  $\\omega_0^{{G_H}} \\in \\{{100, 300\\}}$ at
  $\\omega_0^{{G_\\text{{leaf}}}} = 30$. The
  $(\\omega_0^{{G_\\text{{leaf}}}}=30, \\omega_0^{{G_H}}=30)$ cell is
  shared between the two hyper sub-sweeps.
- **Architecture**: phase 10's polar projection $P$ + Kaiming pre-factor
  on every cell. The only thing that changes between cells is
  $\\omega_0$. SIREN hidden-bound scales as
  $\\sqrt{{6/H}} / \\omega_0$ so the init distribution stays
  paper-faithful at higher $\\omega_0$.

**Pre-registered escalation rule.** Within each arm family
(parallel; hyper varying G_leaf; hyper varying G_H) the winner cell
is identified only when its $K=3$ median functional fidelity exceeds
the runner-up's median by at least $\\max(\\text{{IQR}}_{{\\text{{w}}}},
\\text{{IQR}}_{{\\text{{r}}}})$. If boxes overlap, no cell is selected
at $K=3$ and the top-2 escalate to $K=10$. Recorded before reading
the numbers.

**No fixed lift threshold.** The chance baseline of 0.10 is visible
in the per-cell distribution table and in the fidelity figure;
whether any cell's distribution clears chance is read against the
verbatim distribution per cell, not against a pre-set threshold.
"""

_CAVEATS_MD = """
- **K = {k_seed} smoke** — escalation to $K=10$ is reserved for the
  winner cell if box-non-overlap fires.
- **One target tree** — same $W^{{\\star}}$ as phase 11B (Adam(1e-3) ×
  {target_epochs} epochs, seed 0, no smoothness regulariser). Different
  targets might be easier or harder to fit at high $\\omega_0$;
  phase 12 reports relative-to-baseline at one target.
- **Only $\\omega_0$ varies.** Hidden-dim and depth are held at
  $H = 8, \\text{{depth}} = 2$ for $G_\\text{{leaf}}$ and
  $G_H\\_\\text{{hidden}} = 100, G_H\\_\\text{{depth}} = 3$ for $G_H$,
  unchanged from phases 10 / 11B. If $\\omega_0$ shows no signal, phase
  13 moves to non-bandwidth fixes (FiLM modulation, alternative ondes
  basis, skip connections, curriculum).
- **SIREN init-distribution invariance.** The hidden-layer bound scales
  as $\\sqrt{{6/H}} / \\omega_0$; a 10× $\\omega_0$ implies a 10× drop in
  init weight magnitude. This is the SIREN paper-faithful convention
  and is the same scaling phases 10 / 11B used at $\\omega_0 = 30$.
- **Distillation wall**: {distill_wall:.1f}s.
"""


def main() -> None:
    print("=" * 72)
    print("Phase 12 — Bandwidth (ω₀) ablation on L2 distillation")
    print("=" * 72)
    print("Loading CIFAR-10 ...")
    train_x, train_y, test_x, test_y = data.load_cifar10()
    print(f"  train: {train_x.shape} | test: {test_x.shape}")
    print(f"  K_seed={K_SEED}, distill_steps={DISTILL_STEPS}, lr={DISTILL_LR}")
    print(f"  loss kind: {LOSS_KIND}")
    print(f"  cells: {[c.cell_id for c in CELLS]}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: train target W*. ----------------------------------------
    print("\n--- Step 1: train target WideKernelCNN-SiLU on CIFAR-10 ---")
    t0 = time.time()
    target_W, target_acc, target_ckpts = train_target_cnn(
        train_x, train_y, test_x, test_y,
        epochs=TARGET_EPOCHS, batch_size=TARGET_BATCH_SIZE, seed=0,
    )
    target_wall = time.time() - t0
    n_target_params = sum(int(np.asarray(v).size) for v in target_W.values())
    print(f"  target final test acc: {target_acc:.4f}  (wall {target_wall:.1f}s, "
          f"{n_target_params} params)")
    target_alphas = diagnostics.per_leaf_alphas(target_W)
    print("  target per-leaf α (R²):")
    for leaf, (a, r2) in target_alphas.items():
        print(f"    {leaf}: α={a:.3f} R²={r2:.3f}")

    # --- Storage ---------------------------------------------------------
    final_loss: dict[str, list[float]] = {c.cell_id: [] for c in CELLS}
    per_leaf_l2: dict[str, list[dict[str, float]]] = {c.cell_id: [] for c in CELLS}
    func_acc: dict[str, list[float]] = {c.cell_id: [] for c in CELLS}
    alphas_per_cell: dict[str, list[dict[str, tuple[float, float]]]] = {
        c.cell_id: [] for c in CELLS
    }

    # --- Step 2: distill each cell × K seeds. ----------------------------
    distill_wall = 0.0
    for ci, cell in enumerate(CELLS):
        print(f"\n--- Cell {ci + 1}/{len(CELLS)}: {cell.cell_id} "
              f"(arm={cell.arm_family}, ω₀_G_leaf={cell.omega_0_g_leaf}, "
              f"ω₀_G_H={cell.omega_0_g_h}, axis={cell.axis}) ---", flush=True)
        for seed in range(K_SEED):
            t_seed = time.time()
            arm = make_arm_for_cell(cell)
            _state, traj, rendered = distill_arm_for_cell(
                arm, target_W,
                seed=seed, num_steps=DISTILL_STEPS, lr=DISTILL_LR,
            )
            final = traj[-1][1]
            leaf_l2 = per_leaf_l2_sse(rendered, target_W)
            _, ta = eval_test_loss_acc(rendered, test_x, test_y)
            alphas = diagnostics.per_leaf_alphas(rendered)
            final_loss[cell.cell_id].append(final)
            per_leaf_l2[cell.cell_id].append(leaf_l2)
            func_acc[cell.cell_id].append(ta)
            alphas_per_cell[cell.cell_id].append(alphas)
            seed_wall = time.time() - t_seed
            distill_wall += seed_wall
            print(
                f"  [{cell.cell_id}] seed {seed}: final_loss={final:.4e}  "
                f"func_acc={ta:.4f}  ({seed_wall:.1f}s)",
                flush=True,
            )

    # --- Step 3: numbers, figures, log. ----------------------------------
    print("\n--- Step 3: aggregate and write log ---")

    plot_l2_and_fidelity(
        CELLS, final_loss, func_acc, target_acc,
        FIGURES_DIR / f"{FIG_PREFIX}-l2-and-fidelity.png",
        title=f"Phase 12 — L2 + functional fidelity per cell (K={K_SEED})",
    )
    plot_omega_axis(
        CELLS, final_loss, func_acc, target_acc,
        FIGURES_DIR / f"{FIG_PREFIX}-omega-axis.png",
        title=f"Phase 12 — ω₀ axis × arm × (L2, fidelity), K={K_SEED}",
    )

    # Verdict per family using the pre-registered rule.
    parallel_cells = [c for c in CELLS if c.arm_family == "parallel"]
    hyper_g_leaf_cells = [c for c in CELLS if c.arm_family == "hyper"
                          and c.axis == "g_leaf"]
    hyper_g_h_cells = [c for c in CELLS if c.arm_family == "hyper"
                       and (c.axis == "g_h" or c.cell_id == "hyp-gl30-gh30")]

    par_verdict = box_non_overlap_verdict(func_acc, family_cells=parallel_cells)
    hyp_gl_verdict = box_non_overlap_verdict(func_acc, family_cells=hyper_g_leaf_cells)
    hyp_gh_verdict = box_non_overlap_verdict(func_acc, family_cells=hyper_g_h_cells)

    par_best, par_med, par_iqr, par_min, par_max = best_cell_in_family(
        func_acc, family_cells=parallel_cells,
    )
    hyp_best, hyp_med, hyp_iqr, hyp_min, hyp_max = best_cell_in_family(
        func_acc, family_cells=[c for c in CELLS if c.arm_family == "hyper"],
    )

    # Compose log sections.
    sections: list[tuple[str, str]] = [
        ("Background", _BACKGROUND_MD),
        ("Setup", _SETUP_MD.format(
            target_epochs=TARGET_EPOCHS, distill_steps=DISTILL_STEPS,
            distill_lr=DISTILL_LR, k_seed=K_SEED,
        )),
        ("Step 1 — target CNN", (
            f"Target ``WideKernelCNN-SiLU`` trained on CIFAR-10 with "
            f"Adam(1e-3) for {TARGET_EPOCHS} epochs reached "
            f"**test accuracy {target_acc:.4f}** ({n_target_params} params, "
            f"target wall {target_wall:.1f}s).\n\n"
            "Per-epoch checkpoints:\n\n| epoch | test loss | test acc |\n|---|---|---|\n"
            + "\n".join(f"| {e} | {tl:.4f} | {ta:.4f} |"
                        for e, tl, ta in target_ckpts)
        )),
        (f"Step 2 — eight-cell ω₀ sweep (K={K_SEED} per cell)",
         per_cell_table_md(CELLS, final_loss, per_leaf_l2, func_acc, target_acc)
         + "\n\n### Per-leaf raw L2 (sum-of-squares) — median across seeds\n\n"
         + per_leaf_l2_table_md(CELLS, per_leaf_l2)
         + "\n\n### Per-leaf α at convergence — median across seeds\n\n"
         + alpha_table_md(CELLS, target_alphas, alphas_per_cell)
         + f"\n\n![L2 and functional fidelity per cell]"
           f"(figures/{FIG_PREFIX}-l2-and-fidelity.png)\n\n"
           f"![ω₀ axis × arm × (L2, fidelity)]"
           f"(figures/{FIG_PREFIX}-omega-axis.png)"),
        ("Interpretation (pre-registered box-non-overlap rule per family)", (
            "**FWS-parallel ω₀^G_leaf sweep.** " + par_verdict + "\n\n"
            "**FWS-hyper ω₀^G_leaf sweep (G_H ω₀ held at 30).** "
            + hyp_gl_verdict + "\n\n"
            "**FWS-hyper ω₀^G_H sweep (G_leaf ω₀ held at 30).** "
            + hyp_gh_verdict + "\n\n"
            "**Best-median cell per arm (threshold-free; chance is 0.10).** "
            f"FWS-parallel best: cell {par_best} with K=3 fidelity "
            f"median {par_med:.4f}, IQR {par_iqr:.4f}, "
            f"min {par_min:.4f}, max {par_max:.4f}. "
            f"FWS-hyper best (across both axes): cell {hyp_best} with K=3 "
            f"fidelity median {hyp_med:.4f}, IQR {hyp_iqr:.4f}, "
            f"min {hyp_min:.4f}, max {hyp_max:.4f}. Read these against "
            "the verbatim per-cell distribution and the chance baseline "
            "(0.10) visible in the eight-cell table.\n\n"
            "**Phase 11B baseline reference (verbatim from "
            "`2026-06-07-phase11b-supervised-distillation.md`)**: "
            "FWS-hyper-polar leaf-mean median fidelity 0.1007 (chance), "
            "FWS-parallel-no-G_H leaf-mean median fidelity 0.0793 (chance), "
            "target acc 0.6152 — both at ω₀ = 30. The phase 12 baseline "
            "cells (par-w30 and hyp-gl30-gh30) should reproduce these "
            "numbers up to seed-to-seed variability."
        )),
        ("Honest caveats", _CAVEATS_MD.format(
            k_seed=K_SEED, target_epochs=TARGET_EPOCHS,
            distill_steps=DISTILL_STEPS, distill_wall=distill_wall,
        )),
    ]
    reporting.render_research_log(reporting.ReportContext(
        title=f"Phase 12 — Bandwidth (ω₀) ablation on L2 distillation — {DATE}",
        out_path=RESEARCH_FILE,
        sections=sections,
        figures_dir=FIGURES_DIR,
    ))
    reporting.render_pdf(RESEARCH_FILE)

    # --- Print verbatim summary to stdout so the orchestrator sees it. ---
    print("\n" + "=" * 72)
    print("VERBATIM SUMMARY")
    print("=" * 72)
    print(f"Target acc: {target_acc:.4f}")
    print("\nPer-cell median (L2, fidelity):")
    for c in CELLS:
        L = float(np.median(np.asarray(final_loss[c.cell_id])))
        F = float(np.median(np.asarray(func_acc[c.cell_id])))
        gh = "—" if c.omega_0_g_h is None else f"{c.omega_0_g_h:g}"
        print(f"  {c.cell_id:22s} (arm={c.arm_family:8s} "
              f"ω₀^Gl={c.omega_0_g_leaf:>5g} ω₀^Gh={gh:>4s}): "
              f"L2 median = {L:.4e}, fidelity median = {F:.4f}")
    print("\nFWS-parallel ω₀^G_leaf verdict: " + par_verdict)
    print("FWS-hyper ω₀^G_leaf verdict:    " + hyp_gl_verdict)
    print("FWS-hyper ω₀^G_H verdict:       " + hyp_gh_verdict)
    print(
        f"\nFWS-parallel best cell: {par_best} median {par_med:.4f} "
        f"IQR {par_iqr:.4f} min {par_min:.4f} max {par_max:.4f}"
    )
    print(
        f"FWS-hyper best cell:    {hyp_best} median {hyp_med:.4f} "
        f"IQR {hyp_iqr:.4f} min {hyp_min:.4f} max {hyp_max:.4f}"
    )
    print("(chance is 0.10; read against the per-cell distribution above)")
    print(f"\nDistillation wall: {distill_wall:.1f}s ({distill_wall / 60:.1f} min)")


if __name__ == "__main__":
    main()
