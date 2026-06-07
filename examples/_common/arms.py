"""Baseline arm instantiations + G_leaf machinery for the CIFAR / WideKernelCNN phases.

Three baseline arms (W matched, W overparam, FWS-parallel-no-G_H) and an
``FWS-hyper`` constructor live here. Each returns an :class:`fws_bench.Arm`
— the contract value type owned by ``fws-bench`` — wired to the CIFAR
WideKernelCNN-SiLU mainnet from :mod:`._common.mainnet`.

This module is examples-side glue: it composes the ``fws-bench`` harness
machinery with the specific CIFAR mainnet and ``G_leaf`` parameter
layout shared across phases 8/9/10. The phase scripts construct their
phase-specific ``G_H`` and pass it to :func:`make_fws_hyper`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float

from fws_bench import Arm

from . import mainnet


# --- Shared G_leaf hyperparameters ------------------------------------------
# Depth-2 sine-only MLP, identical in shape across phases 8/9/10:
#   sin(W_out · sin(W_h · sin(omega_0 · W_in · coord + b_in) + b_h) + b_out)
# with W_out: (1, H) so the final activation IS the scalar output (no
# separate linear readout layer).
G_LEAF_HIDDEN_DIM: int = 8
OMEGA_0: float = 30.0


def g_leaf_param_size(rank: int) -> int:
    """Per-rank flat parameter vector length for ``G_leaf``.

    Layout: ``W_in (H, rank) | b_in (H) | W_h (H, H) | b_h (H) |
    W_out (1, H) | b_out (1)``.
    """
    H = G_LEAF_HIDDEN_DIM
    return H * rank + H + H * H + H + H + 1


G_LEAF_PARAM_SIZE: dict[int, int] = {r: g_leaf_param_size(r) for r in mainnet.DISTINCT_RANKS}
MAX_G_LEAF_PARAM_SIZE: int = max(G_LEAF_PARAM_SIZE.values())


def slice_g_leaf_flat(flat: Array, rank: int) -> dict[str, Array]:
    """Unpack the first ``G_LEAF_PARAM_SIZE[rank]`` entries of ``flat``."""
    H = G_LEAF_HIDDEN_DIM
    offset = 0

    def take(size: int, shape: tuple[int, ...]) -> Array:
        nonlocal offset
        chunk = flat[offset:offset + size].reshape(shape)
        offset += size
        return chunk

    W_in = take(H * rank, (H, rank))
    b_in = take(H, (H,))
    W_h = take(H * H, (H, H))
    b_h = take(H, (H,))
    W_out = take(H, (1, H))
    b_out = take(1, (1,))
    return {"W_in": W_in, "b_in": b_in, "W_h": W_h, "b_h": b_h, "W_out": W_out, "b_out": b_out}


def g_leaf_forward(coord: Array, params: dict[str, Array], *, film: Array | None = None) -> Array:
    """Evaluate the sine-only ``G_leaf`` at a single coordinate; returns a scalar.

    ``film``: optional ``(2H,)`` vector providing ``gamma`` (first H) and
    ``beta`` (last H) to modulate the hidden-layer pre-activation. Used by
    :class:`ParallelGLeaves` where shared ``z`` FiLM-modulates each
    independent ``G_leaf``.
    """
    pre1 = params["W_in"] @ coord + params["b_in"]
    h1 = jnp.sin(OMEGA_0 * pre1)
    pre2 = params["W_h"] @ h1 + params["b_h"]
    if film is not None:
        H = G_LEAF_HIDDEN_DIM
        gamma = film[:H]
        beta = film[H:]
        pre2 = gamma * pre2 + beta
    h2 = jnp.sin(OMEGA_0 * pre2)
    pre_out = params["W_out"] @ h2 + params["b_out"]
    return jnp.sin(pre_out[0])


# --- Per-leaf coordinate grids ---------------------------------------------
def _norm_linspace(size: int) -> np.ndarray:
    if size == 1:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(-1.0, 1.0, size, dtype=np.float32)


def _leaf_coords(name: str) -> Array:
    """Natural-rank coordinate grid for the leaf.

    Returns shape ``(numel, rank)`` where ``rank == len(LEAF_SHAPES[name])``.
    Coordinates are normalised anisotropically to ``[-1, 1]`` along each
    axis; the leaf identity flows through the learned leaf embedding into
    ``G_H``, not through the coord grid.
    """
    shape = mainnet.LEAF_SHAPES[name]
    rank = len(shape)
    if rank == 1:
        (k,) = shape
        return jnp.asarray(_norm_linspace(k)[:, None])
    if rank == 2:
        o, i = shape
        oo, ii = np.meshgrid(_norm_linspace(o), _norm_linspace(i), indexing="ij")
        return jnp.asarray(np.stack([oo, ii], axis=-1).reshape(-1, 2))
    if rank == 4:
        o, i, h, w = shape
        oo, ii, hh, ww = np.meshgrid(
            _norm_linspace(o), _norm_linspace(i),
            _norm_linspace(h), _norm_linspace(w),
            indexing="ij",
        )
        return jnp.asarray(np.stack([oo, ii, hh, ww], axis=-1).reshape(-1, 4))
    raise ValueError(f"unsupported leaf rank {rank}")


LEAF_COORDS: dict[str, Array] = {name: _leaf_coords(name) for name in mainnet.LEAF_ORDER}


# --- FWS-parallel-no-G_H arm -----------------------------------------------
# Per-rank trainable G_leaf (one per distinct rank), each FiLM-modulated by
# a per-rank linear projection of the shared z. No hyper-renderer G_H.
DIM_Z: int = 64


class ParallelGLeaves(eqx.Module):
    """Three independent ``G_leaf`` instances (one per distinct rank),
    each receiving ``(gamma, beta)`` from a per-rank linear projection of
    ``z``. No ``G_H``.
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

    def __init__(self, *, key: Array) -> None:
        keys = jax.random.split(key, 6)
        H = G_LEAF_HIDDEN_DIM
        sqrt_H = float(jnp.sqrt(jnp.array(H, dtype=jnp.float32)))

        def init_g_leaf(k: Array, rank: int) -> dict[str, Array]:
            kw_in, kb_in, kw_h, kb_h, kw_out, kb_out = jax.random.split(k, 6)
            bound_in = 1.0 / rank
            W_in = jax.random.uniform(kw_in, (H, rank), minval=-bound_in, maxval=bound_in)
            b_in = jax.random.uniform(kb_in, (H,), minval=-bound_in, maxval=bound_in)
            bound_h = float(jnp.sqrt(jnp.array(6.0 / H, dtype=jnp.float32))) / OMEGA_0
            W_h = jax.random.uniform(kw_h, (H, H), minval=-bound_h, maxval=bound_h)
            b_h = jax.random.uniform(kb_h, (H,), minval=-bound_h, maxval=bound_h)
            W_out = jax.random.uniform(kw_out, (1, H), minval=-bound_h, maxval=bound_h)
            b_out = jax.random.uniform(kb_out, (1,), minval=-bound_h, maxval=bound_h)
            return {"W_in": W_in, "b_in": b_in, "W_h": W_h, "b_h": b_h, "W_out": W_out, "b_out": b_out}

        self.g_leaf_rank1 = init_g_leaf(keys[0], 1)
        self.g_leaf_rank2 = init_g_leaf(keys[1], 2)
        self.g_leaf_rank4 = init_g_leaf(keys[2], 4)

        bound_film = 1.0 / sqrt_H
        # Initialise all three FiLM blocks; storing into named attributes for
        # static jitted lookup in g_leaf_for / film_for.
        kw, kb = jax.random.split(keys[3])
        self.film_W_rank1 = jax.random.uniform(kw, (2 * H, DIM_Z), minval=-bound_film, maxval=bound_film)
        self.film_b_rank1 = jnp.zeros((2 * H,))
        kw, kb = jax.random.split(keys[4])
        self.film_W_rank2 = jax.random.uniform(kw, (2 * H, DIM_Z), minval=-bound_film, maxval=bound_film)
        self.film_b_rank2 = jnp.zeros((2 * H,))
        kw, kb = jax.random.split(keys[5])
        self.film_W_rank4 = jax.random.uniform(kw, (2 * H, DIM_Z), minval=-bound_film, maxval=bound_film)
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


# --- Wong colourblind-safe palette -----------------------------------------
WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
FWS_HYPER_COLOUR = WONG[5]          # blue
FWS_PARALLEL_COLOUR = WONG[1]       # orange
W_MATCHED_COLOUR = WONG[6]          # vermillion
W_OVERPARAM_COLOUR = WONG[3]        # bluish-green


# --- The three shared arms -------------------------------------------------
def make_w_matched(*, lr: float = 1e-3) -> Arm:
    """Direct WideKernelCNN-SiLU init + cross-entropy + Adam."""
    return Arm(
        name="W matched",
        short="w_matched",
        color=W_MATCHED_COLOUR,
        init=mainnet.init_cnn_params,
        loss_fn=mainnet.cross_entropy_loss,
        render_W=lambda params: params,
        optimiser=optax.adam(lr),
    )


def make_w_overparam(*, lr: float = 1e-3) -> Arm:
    """Factored (rank-P) CNN + cross-entropy on materialised W + Adam."""

    def init(key: Array) -> mainnet.OverparamCNN:
        return mainnet.OverparamCNN(key=key)

    def loss_fn(model: mainnet.OverparamCNN, batch: dict) -> Array:
        return mainnet.cross_entropy_loss(model.materialise(), batch)

    return Arm(
        name="W overparam",
        short="w_overparam",
        color=W_OVERPARAM_COLOUR,
        init=init,
        loss_fn=loss_fn,
        render_W=lambda model: model.materialise(),
        optimiser=optax.adam(lr),
    )


def make_fws_parallel(
    *,
    leaf_scale_fn: Callable[[str], float] | None = None,
    projection_fn: Callable[[Array, str], Array] | None = None,
    lr: float = 1e-3,
    z_init_std: float = 0.1,
) -> Arm:
    """FWS-parallel-no-G_H arm: three rank-keyed ``G_leaf``s + per-rank FiLM.

    ``leaf_scale_fn(leaf_name) -> float`` is multiplied onto ``G_leaf``'s
    scalar output before reshaping. Defaults to no scaling (1.0).

    ``projection_fn(W_tensor, leaf_name) -> W_tensor`` is applied to the
    reshape-and-scaled tensor as the last step of rendering; use it for
    phase-10's polar-decomposition pseudo-orthonormal projection.
    Defaults to no projection (identity).
    """
    import loom

    scale_fn: Callable[[str], float] = leaf_scale_fn or (lambda _name: 1.0)
    proj_fn: Callable[[Array, str], Array] = projection_fn or (lambda t, _name: t)

    def init(key: Array) -> dict:
        k_par, k_z = jax.random.split(key)
        return {
            "P": ParallelGLeaves(key=k_par),
            "z": jax.random.normal(k_z, (DIM_Z,)) * z_init_std,
        }

    def render(state: dict) -> dict[str, Array]:
        P = {name: jnp.zeros(shape) for name, shape in mainnet.LEAF_SHAPES.items()}

        def f(path, shape, dtype, params):
            par_in, z_in = params
            leaf_name = path[0].key
            rank = mainnet.LEAF_RANKS[leaf_name]
            scale = scale_fn(leaf_name)
            leaf_params = par_in.g_leaf_for(rank)
            film = par_in.film_for(rank, z_in)
            coords = LEAF_COORDS[leaf_name]
            values = jax.vmap(lambda c: g_leaf_forward(c, leaf_params, film=film))(coords)
            W_raw = scale * values.reshape(shape)
            return proj_fn(W_raw, leaf_name).astype(dtype)

        return loom.render(P, f, (state["P"], state["z"]))

    def loss_fn(state: dict, batch: dict) -> Array:
        W = render(state)
        return mainnet.cross_entropy_loss(W, batch)

    return Arm(
        name="FWS-parallel-no-G_H",
        short="fws_parallel",
        color=FWS_PARALLEL_COLOUR,
        init=init,
        loss_fn=loss_fn,
        render_W=render,
        optimiser=optax.adam(lr),
    )


# --- FWS-hyper arm constructor (phase-specific G_H plugged in) -------------
def make_fws_hyper(
    *,
    g_h_init: Callable[[Array], Any],
    leaf_scale_fn: Callable[[str], float] | None = None,
    projection_fn: Callable[[Array, str], Array] | None = None,
    g_lr: float = 1e-3,
    z_lr: float = 1e-3,
    z_init_std: float = 0.1,
    name: str = "FWS-hyper",
    short: str = "fws_hyper",
) -> Arm:
    """Construct the FWS-hyper arm given a phase-specific ``G_H`` initializer.

    ``g_h_init(key) -> G_H`` must return an Equinox module exposing
    ``produce(z, leaf_id, rank) -> dict[str, Array]`` matching the
    :func:`slice_g_leaf_flat` layout. Each phase customises ``G_H`` (SiLU
    MLP in phase 8, recursive SIREN in phase 9, SIREN body + linear
    readout in phase 10) but the wrapping arm shape is identical.

    ``leaf_scale_fn(leaf_name) -> float`` is multiplied onto ``G_leaf``'s
    scalar output before reshaping; defaults to no scaling (1.0).

    ``projection_fn(W_tensor, leaf_name) -> W_tensor`` is applied to the
    reshape-and-scaled tensor as the last step of rendering; use it for
    phase-10's polar-decomposition pseudo-orthonormal projection.
    Defaults to no projection (identity).
    """
    import loom

    scale_fn: Callable[[str], float] = leaf_scale_fn or (lambda _name: 1.0)
    proj_fn: Callable[[Array, str], Array] = projection_fn or (lambda t, _name: t)

    def init(key: Array) -> dict:
        k_g, k_z = jax.random.split(key)
        return {"G": g_h_init(k_g), "z": jax.random.normal(k_z, (DIM_Z,)) * z_init_std}

    def render(state: dict) -> dict[str, Array]:
        P = {name_: jnp.zeros(shape) for name_, shape in mainnet.LEAF_SHAPES.items()}

        def f(path, shape, dtype, params):
            G_H_in, z_in = params
            leaf_name = path[0].key
            leaf_id = mainnet.LEAF_ORDER.index(leaf_name)
            rank = mainnet.LEAF_RANKS[leaf_name]
            scale = scale_fn(leaf_name)
            leaf_params = G_H_in.produce(z_in, leaf_id, rank)
            coords = LEAF_COORDS[leaf_name]
            values = jax.vmap(lambda c: g_leaf_forward(c, leaf_params))(coords)
            W_raw = scale * values.reshape(shape)
            return proj_fn(W_raw, leaf_name).astype(dtype)

        return loom.render(P, f, (state["G"], state["z"]))

    def loss_fn(state: dict, batch: dict) -> Array:
        W = render(state)
        return mainnet.cross_entropy_loss(W, batch)

    optimiser = optax.multi_transform(
        {"G": optax.adam(g_lr), "z": optax.adam(z_lr)},
        {"G": "G", "z": "z"},
    )
    return Arm(
        name=name,
        short=short,
        color=FWS_HYPER_COLOUR,
        init=init,
        loss_fn=loss_fn,
        render_W=render,
        optimiser=optimiser,
    )


# --- Kaiming-He fan-in scale (phase 10 init-distribution remap) ------------
def kaiming_leaf_scale(name: str) -> float:
    """Per-leaf Kaiming-He fan-in std multiplier.

    Conv ``(out_ch, in_ch, kH, kW)``: ``sqrt(2 / (in_ch · kH · kW))``.
    Fc ``(out, in)``: ``sqrt(2 / in)``.
    Bias ``(idx,)``: ``0.01`` (small init, since biases don't have fan-in).
    """
    shape = mainnet.LEAF_SHAPES[name]
    if len(shape) == 4:
        _, in_ch, kH, kW = shape
        return float(np.sqrt(2.0 / (in_ch * kH * kW)))
    if len(shape) == 2:
        _, in_features = shape
        return float(np.sqrt(2.0 / in_features))
    if len(shape) == 1:
        return 0.01
    raise ValueError(f"unsupported leaf shape {shape}")
