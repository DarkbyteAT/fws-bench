"""Phase 8 (renumbered) — CIFAR-10 fiducial cell with per-leaf hyper-renderer.

Per the FWS design doc this is the *fiducial cell*: CIFAR-10 +
WideKernelCNN-SiLU, four arms, and a blocking K=1 falsifier on the ``G_H``
output-layer init scale. Two architecture corrections from the user this
session sit at the heart of this phase:

1. **Natural-rank coord scheme** — never 1-D-index a multi-rank tensor.
   2-D fc weights use ``(i, j)``, 1-D biases use ``(i,)``, 4-D conv kernels
   use ``(oc, ic, kh, kw)``. ``G_leaf`` has ``input_dim == rank(leaf)``,
   so we keep three distinct ``G_leaf`` templates (rank 1 / 2 / 4) rather
   than the rank-4-with-zero-padding trick from phase 9.

2. **No separate linear readout on G_leaf** — sine all the way down. The
   last layer is ``sin(W_out @ hidden + b_out)`` with
   ``W_out: (1, hidden_dim)`` producing a *scalar* output directly. There
   is no ``W_readout`` linear layer following.

Plus three substantive engineering-team inputs, all baked in:

3. **Contrarian's ablation control — FWS-parallel-no-G_H** (4th arm).
   Three independent ``G_leaf`` instances (one per distinct rank), shared
   ``z`` FiLM-modulating each at its hidden layer, **no hyper-renderer
   ``G_H``**. Isolates "is the hyper-renderer load-bearing" from "is the
   per-leaf natural-rank coord scheme load-bearing".

4. **ml-engineer's K=1 G_H output-scale falsifier** (Stage 0, BLOCKING).
   Before any K=3 run, vary ``G_H``'s output-layer init scale across three
   choices and watch the trajectory of ``sigma_min(d render / d z)`` at
   steps {0, 100, 1000, 5000}. If the three scales converge to within an
   order of magnitude by step 5000 the architecture is doing init
   recovery, not the FWS prior — stop and report.

5. **staff-architect's library-boundary verdict** — all per-leaf wiring
   lives in ``fws-bench/examples/``. ``loom`` and ``ondes`` are not
   modified. Patching ``ondes.SIREN`` to skip its built-in readout is not
   the right contract for ``ondes`` (which exports a complete basis MLP
   *with* readout); we therefore implement the sine-only ``G_leaf``
   inline here as four ``(W, b)`` matrices applied through ``sin``.

Run::

    PHASE8_STAGE=falsifier uv run python examples/phase8_cifar_fiducial.py
    PHASE8_STAGE=k3       uv run python examples/phase8_cifar_fiducial.py
    PHASE8_STAGE=all      uv run python examples/phase8_cifar_fiducial.py   # default
"""

from __future__ import annotations

import os
import pickle
import tarfile
import time
import urllib.request
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import loom
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PyTree
from landscape_archaeology import singular_spectrum


RESEARCH_DIR = Path("/Users/ammar/Documents/Research/fractal-weight-spaces/research")
FIGURES_DIR = RESEARCH_DIR / "figures"
RESEARCH_FILE = RESEARCH_DIR / "2026-06-07-phase8-cifar-fiducial.md"

# Wong colourblind-safe palette
WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
FWS_HYPER_COLOUR = WONG[5]      # blue
FWS_PAR_COLOUR = WONG[1]        # orange
W_MATCHED_COLOUR = WONG[6]      # vermillion
W_OVERPARAM_COLOUR = WONG[3]    # bluish-green


# --- Mainnet topology (WideKernelCNN-SiLU) -----------------------------------
# Sized small enough that K=3 × 4 arms × 5 epochs is CPU-tractable.
# Kernel size k>=5 to engage the radial-FFT alpha diagnostic.
IN_CHANNELS = 3
NUM_CLASSES = 10
CONV1_OUT = 8
CONV2_OUT = 16
KERNEL_SIZE = 5
FC1_HIDDEN = 64

# 32 -> conv k=5 -> 28 -> pool2 -> 14 -> conv k=5 -> 10 -> pool2 -> 5 -> flat 400
FC1_IN_DIM = 5 * 5 * CONV2_OUT          # 400

LEAF_ORDER: tuple[str, ...] = (
    "conv1_w", "conv1_b",
    "conv2_w", "conv2_b",
    "fc1_w", "fc1_b",
    "fc2_w", "fc2_b",
)
N_LEAVES = len(LEAF_ORDER)

LEAF_SHAPES: dict[str, tuple[int, ...]] = {
    "conv1_w": (CONV1_OUT, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE),
    "conv1_b": (CONV1_OUT,),
    "conv2_w": (CONV2_OUT, CONV1_OUT, KERNEL_SIZE, KERNEL_SIZE),
    "conv2_b": (CONV2_OUT,),
    "fc1_w":   (FC1_HIDDEN, FC1_IN_DIM),
    "fc1_b":   (FC1_HIDDEN,),
    "fc2_w":   (NUM_CLASSES, FC1_HIDDEN),
    "fc2_b":   (NUM_CLASSES,),
}
LEAF_SIZES: dict[str, int] = {k: int(np.prod(s)) for k, s in LEAF_SHAPES.items()}
TOTAL_W_PARAMS = sum(LEAF_SIZES.values())

# Natural rank per leaf — the coord dim that G_leaf gets.
LEAF_RANKS: dict[str, int] = {n: len(LEAF_SHAPES[n]) for n in LEAF_ORDER}
DISTINCT_RANKS: tuple[int, ...] = (1, 2, 4)


# --- G_leaf hyperparameters (shared across FWS-hyper and FWS-parallel) -------
# Depth-2 sine-only: sin(W_out · sin(W_h · sin(omega_0 W_in · coord + b_in) + b_h) + b_out)
# with W_out: (1, hidden_dim), so the final activation is the scalar output.
G_LEAF_HIDDEN_DIM = 8
OMEGA_0 = 30.0                              # canonical SIREN omega_0 (Sitzmann+ 2020)


def _g_leaf_param_size(rank: int) -> int:
    """Per-leaf-rank ``G_leaf`` flat parameter vector length.

    Layout: ``W_in (H, rank) | b_in (H) | W_h (H, H) | b_h (H) | W_out (1, H) | b_out (1)``.
    """
    H = G_LEAF_HIDDEN_DIM
    return H * rank + H + H * H + H + H + 1


G_LEAF_PARAM_SIZE: dict[int, int] = {r: _g_leaf_param_size(r) for r in DISTINCT_RANKS}
MAX_G_LEAF_PARAM_SIZE = max(G_LEAF_PARAM_SIZE.values())


def _slice_g_leaf_flat(flat: Array, rank: int) -> dict[str, Array]:
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
    """Evaluate the sine-only ``G_leaf`` at a single ``coord``.

    Forward: ``sin(W_out @ sin(W_h @ sin(omega_0 W_in @ coord + b_in) + b_h) + b_out)``.
    Returns a scalar.

    ``film``: optional ``(2 * H,)`` vector providing ``gamma`` (first H) and
    ``beta`` (last H) to modulate the hidden-layer pre-activation. Used by
    the FWS-parallel-no-G_H arm where shared ``z`` FiLM-modulates each
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
    return jnp.sin(pre_out[0])              # scalar


# --- G_H: hyper-renderer (FWS-hyper arm) ------------------------------------
# Sized so total trainable count ≈ mainnet count (~30K).
DIM_Z = 64
DIM_LEAF_EMB = 8
G_H_HIDDEN_DIM = 150

# Per leaf, slice the first G_LEAF_PARAM_SIZE[rank] entries from G_H's output;
# leftover slots up to MAX_G_LEAF_PARAM_SIZE are unused for that leaf.


def _g_h_W_out_scale(scale_kind: str) -> float:
    """Output-layer init scale for ``G_H``'s ``W_out`` / ``b_out``.

    Three choices used by Stage 0's falsifier:

    - ``"existing"`` (the phase-7/9 default): ``0.1 / sqrt(G_H_HIDDEN_DIM)``,
      a heuristic shrink to keep the produced ``G_leaf`` parameters small
      at init so the rendered weights are small.
    - ``"siren_derived"``: ``1 / (omega_0 * sqrt(G_H_HIDDEN_DIM))``, the
      SIREN-paper bound for layers *following* a ``sin(omega_0 ·)`` —
      propagated through ``G_H`` to act as the variance-preserving bound
      on ``G_leaf``'s first layer.
    - ``"siren_10x"``: ``10 × siren_derived``, the over-scaled control.

    The three scales are deliberately spread by ~3 OoM (existing is
    ~3× siren_derived at omega_0=30; siren_10x is 10× siren_derived).
    """
    sqrt_H = float(jnp.sqrt(jnp.array(G_H_HIDDEN_DIM, dtype=jnp.float32)))
    if scale_kind == "existing":
        return 0.1 / sqrt_H
    if scale_kind == "siren_derived":
        return 1.0 / (OMEGA_0 * sqrt_H)
    if scale_kind == "siren_10x":
        return 10.0 / (OMEGA_0 * sqrt_H)
    raise ValueError(f"unknown scale_kind={scale_kind!r}")


SCALE_KINDS: tuple[str, ...] = ("existing", "siren_derived", "siren_10x")


class HyperRenderer(eqx.Module):
    """``G_H``: 2-layer SiLU MLP that maps ``(z, leaf_emb[id]) -> flat`` of
    length ``MAX_G_LEAF_PARAM_SIZE``. Per leaf, slice the first
    ``G_LEAF_PARAM_SIZE[rank]`` entries and unpack into the rank-appropriate
    ``G_leaf`` template.
    """

    leaf_embedding: Float[Array, "n_leaves dim_leaf_emb"]
    W_in: Float[Array, "hidden in_total"]
    b_in: Float[Array, "hidden"]
    W_out: Float[Array, "g_leaf_param_size hidden"]
    b_out: Float[Array, "g_leaf_param_size"]

    def __init__(self, *, key: Array, out_scale_kind: str = "existing") -> None:
        k_emb, k_w_in, k_b_in, k_w_out, k_b_out = jax.random.split(key, 5)
        in_total = DIM_Z + DIM_LEAF_EMB

        emb_bound = 1.0 / jnp.sqrt(jnp.array(DIM_LEAF_EMB, dtype=jnp.float32))
        self.leaf_embedding = jax.random.uniform(
            k_emb, (N_LEAVES, DIM_LEAF_EMB), minval=-emb_bound, maxval=emb_bound
        )

        bound_in = jnp.sqrt(jnp.array(6.0 / in_total, dtype=jnp.float32))
        self.W_in = jax.random.uniform(k_w_in, (G_H_HIDDEN_DIM, in_total),
                                       minval=-bound_in, maxval=bound_in)
        self.b_in = jax.random.uniform(k_b_in, (G_H_HIDDEN_DIM,),
                                       minval=-bound_in, maxval=bound_in)

        bound_out = _g_h_W_out_scale(out_scale_kind)
        self.W_out = jax.random.uniform(k_w_out, (MAX_G_LEAF_PARAM_SIZE, G_H_HIDDEN_DIM),
                                        minval=-bound_out, maxval=bound_out)
        self.b_out = jax.random.uniform(k_b_out, (MAX_G_LEAF_PARAM_SIZE,),
                                        minval=-bound_out, maxval=bound_out)

    def produce_flat(self, z: Array, leaf_id: int) -> Array:
        emb = self.leaf_embedding[leaf_id]
        inp = jnp.concatenate([z, emb])
        h = jax.nn.silu(self.W_in @ inp + self.b_in)
        return self.W_out @ h + self.b_out

    def produce(self, z: Array, leaf_id: int, rank: int) -> dict[str, Array]:
        flat = self.produce_flat(z, leaf_id)
        return _slice_g_leaf_flat(flat, rank)


# --- FWS-parallel-no-G_H arm: per-rank trainable G_leaf + FiLM from z -------
# Three independent G_leaf parameter sets (one per distinct rank), each
# FiLM-modulated by a linear map from the shared z (dim 64).


class ParallelGLeaves(eqx.Module):
    """Three independent ``G_leaf`` instances (one per distinct rank),
    each receiving ``gamma, beta`` from a per-rank linear projection of
    ``z``. No ``G_H``.
    """

    g_leaf_rank1: dict[str, Array]
    g_leaf_rank2: dict[str, Array]
    g_leaf_rank4: dict[str, Array]
    film_W_rank1: Float[Array, "two_h dim_z"]
    film_b_rank1: Float[Array, "two_h"]
    film_W_rank2: Float[Array, "two_h dim_z"]
    film_b_rank2: Float[Array, "two_h"]
    film_W_rank4: Float[Array, "two_h dim_z"]
    film_b_rank4: Float[Array, "two_h"]

    def __init__(self, *, key: Array) -> None:
        keys = jax.random.split(key, 6)
        H = G_LEAF_HIDDEN_DIM
        sqrt_H = float(jnp.sqrt(jnp.array(H, dtype=jnp.float32)))

        def init_g_leaf(k: Array, rank: int) -> dict[str, Array]:
            kw_in, kb_in, kw_h, kb_h, kw_out, kb_out = jax.random.split(k, 6)
            # First-layer SIREN init: bound = 1/in_dim
            bound_in = 1.0 / rank
            W_in = jax.random.uniform(kw_in, (H, rank), minval=-bound_in, maxval=bound_in)
            b_in = jax.random.uniform(kb_in, (H,), minval=-bound_in, maxval=bound_in)
            # Hidden- and output-layer SIREN init: bound = sqrt(6/in)/omega
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
        for fk_idx, (rank, key_) in enumerate(zip([1, 2, 4], keys[3:], strict=True)):
            kw, kb = jax.random.split(key_)
            W = jax.random.uniform(kw, (2 * H, DIM_Z), minval=-bound_film, maxval=bound_film)
            b = jax.random.uniform(kb, (2 * H,), minval=-bound_film, maxval=bound_film) * 0.0
            if rank == 1:
                self.film_W_rank1 = W; self.film_b_rank1 = b
            elif rank == 2:
                self.film_W_rank2 = W; self.film_b_rank2 = b
            else:
                self.film_W_rank4 = W; self.film_b_rank4 = b
            del fk_idx                      # silence pyright

    def g_leaf_for(self, rank: int) -> dict[str, Array]:
        return {1: self.g_leaf_rank1, 2: self.g_leaf_rank2, 4: self.g_leaf_rank4}[rank]

    def film_for(self, rank: int, z: Array) -> Array:
        W, b = {
            1: (self.film_W_rank1, self.film_b_rank1),
            2: (self.film_W_rank2, self.film_b_rank2),
            4: (self.film_W_rank4, self.film_b_rank4),
        }[rank]
        return W @ z + b


# --- Coordinate stacks (natural-rank per leaf) -------------------------------
def _norm_linspace(size: int) -> np.ndarray:
    if size == 1:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(-1.0, 1.0, size, dtype=np.float32)


def _leaf_coords(name: str) -> Array:
    """Natural-rank coordinate grid for the leaf.

    Shape: ``(numel, rank)`` where ``rank == len(LEAF_SHAPES[name])``.
    Coords normalised anisotropically to ``[-1, 1]`` along each axis.
    """
    shape = LEAF_SHAPES[name]
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


LEAF_COORDS: dict[str, Array] = {name: _leaf_coords(name) for name in LEAF_ORDER}


# --- Renderers ---------------------------------------------------------------
def hyper_render_fn(G_H: HyperRenderer, z: Array) -> dict[str, Array]:
    """FWS-hyper renderer: each leaf gets its own G_leaf parameters from G_H."""
    P = {name: jnp.zeros(shape) for name, shape in LEAF_SHAPES.items()}

    def f(path, shape, dtype, params):
        G_H_in, z_in = params
        leaf_name = path[0].key
        leaf_id = LEAF_ORDER.index(leaf_name)
        rank = LEAF_RANKS[leaf_name]
        leaf_params = G_H_in.produce(z_in, leaf_id, rank)
        coords = LEAF_COORDS[leaf_name]
        values = jax.vmap(lambda c: g_leaf_forward(c, leaf_params))(coords)
        return values.reshape(shape).astype(dtype)

    return loom.render(P, f, (G_H, z))


def parallel_render_fn(parallel: ParallelGLeaves, z: Array) -> dict[str, Array]:
    """FWS-parallel renderer: three independent G_leaf's, FiLM'd by z, no G_H."""
    P = {name: jnp.zeros(shape) for name, shape in LEAF_SHAPES.items()}

    def f(path, shape, dtype, params):
        par_in, z_in = params
        leaf_name = path[0].key
        rank = LEAF_RANKS[leaf_name]
        leaf_params = par_in.g_leaf_for(rank)
        film = par_in.film_for(rank, z_in)
        coords = LEAF_COORDS[leaf_name]
        values = jax.vmap(lambda c: g_leaf_forward(c, leaf_params, film=film))(coords)
        return values.reshape(shape).astype(dtype)

    return loom.render(P, f, (parallel, z))


# --- CIFAR-10 loader ---------------------------------------------------------
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_CACHE = Path.home() / ".cache" / "cifar10"
CIFAR_NAMES = ("airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck")


def _download_cifar() -> Path:
    CIFAR_CACHE.mkdir(parents=True, exist_ok=True)
    tgz = CIFAR_CACHE / "cifar-10-python.tar.gz"
    extracted = CIFAR_CACHE / "cifar-10-batches-py"
    if not extracted.exists():
        if not tgz.exists():
            print(f"  downloading CIFAR-10 to {tgz} ...")
            urllib.request.urlretrieve(CIFAR_URL, tgz)
        print(f"  extracting CIFAR-10 to {extracted} ...")
        with tarfile.open(tgz) as tf:
            tf.extractall(CIFAR_CACHE)
    return extracted


def _load_pickle_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Safe pickle source: canonical CIFAR-10 distribution from
    # cs.toronto.edu/~kriz/cifar-10-python.tar.gz (Krizhevsky's upstream).
    with path.open("rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    data = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.asarray(d[b"labels"], dtype=np.int32)
    return data, labels


def load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = _download_cifar()
    train_x, train_y = [], []
    for i in range(1, 6):
        x, y = _load_pickle_batch(root / f"data_batch_{i}")
        train_x.append(x); train_y.append(y)
    train_x = np.concatenate(train_x); train_y = np.concatenate(train_y)
    test_x, test_y = _load_pickle_batch(root / "test_batch")
    mean = train_x.mean(axis=(0, 2, 3), keepdims=True)
    std = train_x.std(axis=(0, 2, 3), keepdims=True) + 1e-7
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    return train_x, train_y, test_x, test_y


# --- Mainnet forward + loss --------------------------------------------------
def _conv2d(x: Array, w: Array, b: Array) -> Array:
    out = jax.lax.conv_general_dilated(
        x[None], w, window_strides=(1, 1), padding="VALID",
    )[0]
    return out + b[:, None, None]


def _maxpool2(x: Array) -> Array:
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2), (1, 2, 2), "VALID")


def cnn_forward(params: dict[str, Array], x: Float[Array, "3 32 32"]) -> Float[Array, " 10"]:
    h = jax.nn.silu(_conv2d(x, params["conv1_w"], params["conv1_b"]))
    h = _maxpool2(h)
    h = jax.nn.silu(_conv2d(h, params["conv2_w"], params["conv2_b"]))
    h = _maxpool2(h)
    h = h.reshape(-1)
    h = jax.nn.silu(params["fc1_w"] @ h + params["fc1_b"])
    return params["fc2_w"] @ h + params["fc2_b"]


def cross_entropy_loss(params: dict[str, Array], batch: dict) -> Float[Array, ""]:
    logits = jax.vmap(lambda x: cnn_forward(params, x))(batch["x"])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(batch["y"], NUM_CLASSES)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


# --- Direct CNN init (matched arm) ------------------------------------------
def init_cnn_params(key: Array) -> dict[str, Array]:
    keys = jax.random.split(key, 4)

    def fan_in(shape: tuple[int, ...]) -> int:
        return int(np.prod(shape[1:]))

    def conv_init(k: Array, shape: tuple[int, ...]) -> Array:
        bound = jnp.sqrt(jnp.array(6.0 / fan_in(shape), dtype=jnp.float32))
        return jax.random.uniform(k, shape, minval=-bound, maxval=bound)

    def fc_init(k: Array, shape: tuple[int, ...]) -> Array:
        bound = jnp.sqrt(jnp.array(6.0 / shape[1], dtype=jnp.float32))
        return jax.random.uniform(k, shape, minval=-bound, maxval=bound)

    return {
        "conv1_w": conv_init(keys[0], LEAF_SHAPES["conv1_w"]),
        "conv1_b": jnp.zeros(LEAF_SHAPES["conv1_b"]),
        "conv2_w": conv_init(keys[1], LEAF_SHAPES["conv2_w"]),
        "conv2_b": jnp.zeros(LEAF_SHAPES["conv2_b"]),
        "fc1_w":   fc_init(keys[2], LEAF_SHAPES["fc1_w"]),
        "fc1_b":   jnp.zeros(LEAF_SHAPES["fc1_b"]),
        "fc2_w":   fc_init(keys[3], LEAF_SHAPES["fc2_w"]),
        "fc2_b":   jnp.zeros(LEAF_SHAPES["fc2_b"]),
    }


# --- W-overparam: factored mainnet (~4× matched) -----------------------------
OVERPARAM_CONV_P = 32                       # bottleneck channels for factored conv
OVERPARAM_FC1_P = 200
OVERPARAM_FC2_P = 64


class OverparamCNN(eqx.Module):
    c1_depth: Array
    c1_point: Array
    c1_bias: Array
    c2_depth: Array
    c2_point: Array
    c2_bias: Array
    fc1_A1: Array
    fc1_A2: Array
    fc1_bias: Array
    fc2_A1: Array
    fc2_A2: Array
    fc2_bias: Array

    def __init__(self, *, key: Array) -> None:
        ks = jax.random.split(key, 8)
        P_c = OVERPARAM_CONV_P
        scale_c = (0.5 / P_c) ** 0.5
        self.c1_depth = jax.random.normal(ks[0], (CONV1_OUT, P_c, KERNEL_SIZE, KERNEL_SIZE)) * scale_c
        self.c1_point = jax.random.normal(ks[1], (P_c, IN_CHANNELS)) * scale_c
        self.c1_bias = jnp.zeros((CONV1_OUT,))
        self.c2_depth = jax.random.normal(ks[2], (CONV2_OUT, P_c, KERNEL_SIZE, KERNEL_SIZE)) * scale_c
        self.c2_point = jax.random.normal(ks[3], (P_c, CONV1_OUT)) * scale_c
        self.c2_bias = jnp.zeros((CONV2_OUT,))

        P1 = OVERPARAM_FC1_P
        scale_fc1 = (0.5 / P1) ** 0.5
        self.fc1_A1 = jax.random.normal(ks[4], (P1, FC1_IN_DIM)) * scale_fc1
        self.fc1_A2 = jax.random.normal(ks[5], (FC1_HIDDEN, P1)) * scale_fc1
        self.fc1_bias = jnp.zeros((FC1_HIDDEN,))

        P2 = OVERPARAM_FC2_P
        scale_fc2 = (0.5 / P2) ** 0.5
        self.fc2_A1 = jax.random.normal(ks[6], (P2, FC1_HIDDEN)) * scale_fc2
        self.fc2_A2 = jax.random.normal(ks[7], (NUM_CLASSES, P2)) * scale_fc2
        self.fc2_bias = jnp.zeros((NUM_CLASSES,))

    def materialise(self) -> dict[str, Array]:
        conv1_w = jnp.einsum("opHW,pi->oiHW", self.c1_depth, self.c1_point)
        conv2_w = jnp.einsum("opHW,pi->oiHW", self.c2_depth, self.c2_point)
        return {
            "conv1_w": conv1_w, "conv1_b": self.c1_bias,
            "conv2_w": conv2_w, "conv2_b": self.c2_bias,
            "fc1_w": self.fc1_A2 @ self.fc1_A1, "fc1_b": self.fc1_bias,
            "fc2_w": self.fc2_A2 @ self.fc2_A1, "fc2_b": self.fc2_bias,
        }


# --- Training schedule -------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = int(os.environ.get("PHASE8_EPOCHS", 5))
K_SEED = int(os.environ.get("PHASE8_K_SEED", 3))
LOG_EVERY = 100

# Falsifier
FALSIFIER_STEPS = int(os.environ.get("PHASE8_FALSIFIER_STEPS", 5000))
FALSIFIER_CHECKPOINTS: tuple[int, ...] = (0, 100, 1000, FALSIFIER_STEPS)

# Spectral probes
SIGMA_TOP_K = 10
SIGMA_POWER_ITERS = 60
HESSIAN_POWER_ITERS = 40

EVAL_HESS_BATCH = 1024


G_LR = 1e-3
Z_LR = 1e-3
W_LR = 1e-3
OP_LR = 1e-3
PARALLEL_LR = 1e-3


# --- JIT'd step constructors -------------------------------------------------
def _global_l2_norm(grad_tree: PyTree) -> Array:
    flat, _ = ravel_pytree(grad_tree)
    return jnp.linalg.norm(flat)


def make_fws_hyper_step():
    fws_opt = optax.multi_transform(
        {"G": optax.adam(G_LR), "z": optax.adam(Z_LR)},
        {"G": "G", "z": "z"},
    )

    def loss_fn(combined, batch):
        W_tree = hyper_render_fn(combined["G"], combined["z"])
        return cross_entropy_loss(W_tree, batch)

    @jax.jit
    def step(combined, state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(combined, batch)
        gnorm = _global_l2_norm(grads)
        updates, new_state = fws_opt.update(grads, state, combined)
        new_combined = optax.apply_updates(combined, updates)
        return new_combined, new_state, loss, gnorm

    return fws_opt, step


def make_fws_parallel_step():
    par_opt = optax.adam(PARALLEL_LR)

    def loss_fn(combined, batch):
        W_tree = parallel_render_fn(combined["P"], combined["z"])
        return cross_entropy_loss(W_tree, batch)

    @jax.jit
    def step(combined, state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(combined, batch)
        gnorm = _global_l2_norm(grads)
        updates, new_state = par_opt.update(grads, state, combined)
        new_combined = optax.apply_updates(combined, updates)
        return new_combined, new_state, loss, gnorm

    return par_opt, step


def make_direct_step(loss_fn, optimiser):
    @jax.jit
    def step(params, state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        gnorm = _global_l2_norm(grads)
        updates, new_state = optimiser.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss, gnorm

    return step


# --- Spectral probes ---------------------------------------------------------
def sigma_at_z_hyper(G_H: HyperRenderer, z: Array, *, seed: int = 7) -> Array:
    op = lambda z_var: hyper_render_fn(G_H, z_var)
    return singular_spectrum(op, z, k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS, key=jax.random.key(seed))


def sigma_at_gh_hyper(G_H: HyperRenderer, z: Array, *, seed: int = 13) -> Array:
    op = lambda G_var: hyper_render_fn(G_var, z)
    return singular_spectrum(op, G_H, k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS, key=jax.random.key(seed))


def sigma_at_z_parallel(P_par: ParallelGLeaves, z: Array, *, seed: int = 7) -> Array:
    op = lambda z_var: parallel_render_fn(P_par, z_var)
    return singular_spectrum(op, z, k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS, key=jax.random.key(seed))


def hessian_top(params: PyTree, batch: dict, *, seed: int = 11) -> Array:
    grad_fn = jax.grad(lambda p: cross_entropy_loss(p, batch))
    return singular_spectrum(grad_fn, params, k=SIGMA_TOP_K, num_iterations=HESSIAN_POWER_ITERS, key=jax.random.key(seed))


# --- HT-SR α and radial-FFT α -----------------------------------------------
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


def radial_fft_alpha(conv_w: Array) -> tuple[float, float]:
    W_np = np.asarray(conv_w)
    out_ch, in_ch, kH, kW = W_np.shape
    F = np.fft.fft2(W_np, axes=(-2, -1))
    power = np.abs(F) ** 2
    power_pooled = power.sum(axis=(0, 1))

    ky = np.fft.fftfreq(kH) * kH
    kx = np.fft.fftfreq(kW) * kW
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    R = np.sqrt(KY ** 2 + KX ** 2)
    R_int = np.round(R).astype(int)
    radial_bins = np.arange(0, int(R_int.max()) + 1)
    radial_mean = np.array([
        power_pooled[R_int == r].mean() if (R_int == r).any() else np.nan
        for r in radial_bins
    ])
    finite = np.isfinite(radial_mean) & (radial_mean > 0) & (radial_bins > 0)
    if finite.sum() < 2:
        return float("nan"), float("nan")
    log_k = np.log(radial_bins[finite].astype(np.float64))
    log_p = np.log(radial_mean[finite].astype(np.float64))
    slope, intercept = np.polyfit(log_k, log_p, 1)
    alpha = float(-slope)
    log_p_hat = slope * log_k + intercept
    ss_res = float(np.sum((log_p - log_p_hat) ** 2))
    ss_tot = float(np.sum((log_p - np.mean(log_p)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return alpha, r2


def count_params(tree: PyTree) -> int:
    flat, _ = ravel_pytree(tree)
    return int(flat.size)


# --- Stage 0: G_H W_out init-scale falsifier --------------------------------
def run_stage0_falsifier(train_x: np.ndarray, train_y: np.ndarray, seed: int = 0
                         ) -> dict[str, list[tuple[int, float]]]:
    """Run K=1 FWS-hyper training for each ``out_scale_kind`` and record
    ``sigma_min`` of ``d render / d z`` at the falsifier checkpoints.

    Returns ``{scale_kind: [(step, sigma_min), ...]}``.
    """
    print("=" * 72)
    print("Stage 0 — G_H W_out init-scale falsifier (BLOCKING)")
    print("=" * 72)
    out: dict[str, list[tuple[int, float]]] = {}
    rng = np.random.default_rng(seed)
    n_train = train_x.shape[0]

    for scale_kind in SCALE_KINDS:
        bound = _g_h_W_out_scale(scale_kind)
        print(f"\n  scale_kind={scale_kind}  (bound={bound:.4g})  steps={FALSIFIER_STEPS}", flush=True)

        key = jax.random.key(seed)
        k_g, k_z = jax.random.split(key, 2)
        init_G = HyperRenderer(key=k_g, out_scale_kind=scale_kind)
        init_z = jax.random.normal(k_z, (DIM_Z,)) * 0.1
        combined = {"G": init_G, "z": init_z}

        fws_opt, fws_step = make_fws_hyper_step()
        state = fws_opt.init(combined)

        records: list[tuple[int, float]] = []
        idx = np.arange(n_train)
        rng.shuffle(idx)
        cursor = 0

        last_print = time.time()
        for step_i in range(FALSIFIER_STEPS + 1):
            if step_i in FALSIFIER_CHECKPOINTS:
                sig = np.asarray(sigma_at_z_hyper(combined["G"], combined["z"]))
                sigma_min = float(sig[-1])
                records.append((step_i, sigma_min))
                print(f"    step {step_i:>5d}: sigma_min(d render / d z) = {sigma_min:.6g}", flush=True)
                last_print = time.time()
            if step_i == FALSIFIER_STEPS:
                break
            if cursor + BATCH_SIZE > n_train:
                rng.shuffle(idx); cursor = 0
            batch_idx = idx[cursor:cursor + BATCH_SIZE]
            cursor += BATCH_SIZE
            batch = {"x": jnp.asarray(train_x[batch_idx]),
                     "y": jnp.asarray(train_y[batch_idx])}
            combined, state, _, _ = fws_step(combined, state, batch)
            # Progress ping at most every 60s while running.
            if time.time() - last_print > 60:
                print(f"    progress: step {step_i} (between checkpoints, no probe)", flush=True)
                last_print = time.time()

        out[scale_kind] = records

    return out


def stage0_verdict(records: dict[str, list[tuple[int, float]]]) -> tuple[bool, str]:
    """Decide: do the three scales converge to within 1 OoM by step
    ``FALSIFIER_STEPS``? Returns ``(proceed_to_k3, verdict_text)``.

    Convergence = init recovery (architecture is mechanistically the same
    as phase 6's failure) → stop, do not run K=3.
    Divergence (>=1 OoM apart at the final step) = FWS prior doing
    geometric work → proceed to K=3.
    """
    final = {k: v[-1][1] for k, v in records.items()}
    vals = np.array(list(final.values()))
    log_vals = np.log10(np.maximum(vals, np.finfo(vals.dtype).tiny))
    log_spread = float(log_vals.max() - log_vals.min())
    proceed = log_spread >= 1.0
    text = (
        f"sigma_min at step {FALSIFIER_STEPS}: "
        + ", ".join(f"{k}={final[k]:.4g}" for k in SCALE_KINDS)
        + f"  (log10 spread = {log_spread:.3f})\n"
        + ("DECISION: spread >= 1 OoM — FWS prior doing geometric work; PROCEED to K=3."
           if proceed else
           "DECISION: spread < 1 OoM — init recovery, not FWS prior; STOP, do not run K=3.")
    )
    return proceed, text


def plot_stage0(records: dict[str, list[tuple[int, float]]], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colours = {"existing": WONG[5], "siren_derived": WONG[3], "siren_10x": WONG[6]}
    for scale_kind, recs in records.items():
        steps = [r[0] for r in recs]
        vals = [r[1] for r in recs]
        ax.plot(steps, vals, marker="o", linewidth=2.0, color=colours[scale_kind],
                label=f"{scale_kind} (bound={_g_h_W_out_scale(scale_kind):.3g})")
    ax.set(xlabel="outer step", ylabel="sigma_min(d render / d z)",
           title="Stage 0 — G_H W_out init-scale falsifier")
    ax.set_yscale("log")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --- Per-seed K=3 run --------------------------------------------------------
def make_inits(seed: int):
    key = jax.random.key(seed)
    k_g, k_z, k_par, k_par_z, k_w, k_op = jax.random.split(key, 6)
    init_G = HyperRenderer(key=k_g, out_scale_kind="existing")
    init_z = jax.random.normal(k_z, (DIM_Z,)) * 0.1
    init_par = ParallelGLeaves(key=k_par)
    init_par_z = jax.random.normal(k_par_z, (DIM_Z,)) * 0.1
    init_W = init_cnn_params(k_w)
    init_op = OverparamCNN(key=k_op)
    return init_G, init_z, init_par, init_par_z, init_W, init_op


def _eval_test_loss_acc(params: dict[str, Array], test_x: np.ndarray, test_y: np.ndarray
                        ) -> tuple[float, float]:
    chunk = 1024
    n = test_x.shape[0]
    total_loss = 0.0
    total_correct = 0
    for i in range(0, n, chunk):
        bx = jnp.asarray(test_x[i:i + chunk])
        by = jnp.asarray(test_y[i:i + chunk])
        logits = jax.vmap(lambda x: cnn_forward(params, x))(bx)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(by, NUM_CLASSES)
        loss = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
        preds = jnp.argmax(logits, axis=-1)
        total_loss += float(loss) * bx.shape[0]
        total_correct += int(jnp.sum(preds == by))
    return total_loss / n, total_correct / n


def run_seed(seed: int, train_x: np.ndarray, train_y: np.ndarray,
             test_x: np.ndarray, test_y: np.ndarray) -> dict:
    init_G, init_z, init_par, init_par_z, init_W, init_op = make_inits(seed)

    fws_combined = {"G": init_G, "z": init_z}
    fws_opt, fws_step = make_fws_hyper_step()
    fws_state = fws_opt.init(fws_combined)

    par_combined = {"P": init_par, "z": init_par_z}
    par_opt, par_step = make_fws_parallel_step()
    par_state = par_opt.init(par_combined)

    w_opt = optax.adam(W_LR)
    w_step = make_direct_step(cross_entropy_loss, w_opt)
    w_state = w_opt.init(init_W); w_params = init_W

    op_opt = optax.adam(OP_LR)

    def op_loss_fn(op_params, batch):
        return cross_entropy_loss(op_params.materialise(), batch)

    op_step = make_direct_step(op_loss_fn, op_opt)
    op_state = op_opt.init(init_op); op_params = init_op

    fws_losses: list[tuple[int, float, float]] = []
    par_losses: list[tuple[int, float, float]] = []
    w_losses: list[tuple[int, float, float]] = []
    op_losses: list[tuple[int, float, float]] = []
    fws_ckpts: list[tuple[int, float, float]] = []
    par_ckpts: list[tuple[int, float, float]] = []
    w_ckpts: list[tuple[int, float, float]] = []
    op_ckpts: list[tuple[int, float, float]] = []

    rng = np.random.default_rng(seed)
    n_train = train_x.shape[0]
    steps_per_epoch = n_train // BATCH_SIZE

    global_step = 0
    for epoch in range(EPOCHS):
        idx = np.arange(n_train); rng.shuffle(idx)
        for s in range(steps_per_epoch):
            bi = idx[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]
            batch = {"x": jnp.asarray(train_x[bi]), "y": jnp.asarray(train_y[bi])}

            fws_combined, fws_state, fws_l, fws_gn = fws_step(fws_combined, fws_state, batch)
            par_combined, par_state, par_l, par_gn = par_step(par_combined, par_state, batch)
            w_params, w_state, w_l, w_gn = w_step(w_params, w_state, batch)
            op_params, op_state, op_l, op_gn = op_step(op_params, op_state, batch)

            if global_step % LOG_EVERY == 0:
                fws_losses.append((global_step, float(fws_l), float(fws_gn)))
                par_losses.append((global_step, float(par_l), float(par_gn)))
                w_losses.append((global_step, float(w_l), float(w_gn)))
                op_losses.append((global_step, float(op_l), float(op_gn)))
            global_step += 1

        fws_W = hyper_render_fn(fws_combined["G"], fws_combined["z"])
        par_W = parallel_render_fn(par_combined["P"], par_combined["z"])
        op_W = op_params.materialise()
        fws_tl, fws_ta = _eval_test_loss_acc(fws_W, test_x, test_y)
        par_tl, par_ta = _eval_test_loss_acc(par_W, test_x, test_y)
        w_tl, w_ta = _eval_test_loss_acc(w_params, test_x, test_y)
        op_tl, op_ta = _eval_test_loss_acc(op_W, test_x, test_y)
        fws_ckpts.append((epoch + 1, fws_tl, fws_ta))
        par_ckpts.append((epoch + 1, par_tl, par_ta))
        w_ckpts.append((epoch + 1, w_tl, w_ta))
        op_ckpts.append((epoch + 1, op_tl, op_ta))
        print(f"    [seed {seed}] epoch {epoch + 1}/{EPOCHS}: "
              f"hyper={fws_ta:.4f} par={par_ta:.4f} w={w_ta:.4f} op={op_ta:.4f}", flush=True)

    final_fws_W = hyper_render_fn(fws_combined["G"], fws_combined["z"])
    final_par_W = parallel_render_fn(par_combined["P"], par_combined["z"])
    final_w_W = w_params
    final_op_W = op_params.materialise()

    def _alphas(W_tree: dict[str, Array]) -> dict[str, tuple[float, float]]:
        return {
            "conv1": radial_fft_alpha(W_tree["conv1_w"]),
            "conv2": radial_fft_alpha(W_tree["conv2_w"]),
            "fc1":   ht_sr_alpha(W_tree["fc1_w"]),
            "fc2":   ht_sr_alpha(W_tree["fc2_w"]),
        }

    fws_alphas = _alphas(final_fws_W)
    par_alphas = _alphas(final_par_W)
    w_alphas = _alphas(final_w_W)
    op_alphas = _alphas(final_op_W)

    sigma_z_hyper = np.asarray(sigma_at_z_hyper(fws_combined["G"], fws_combined["z"]))
    sigma_gh_hyper = np.asarray(sigma_at_gh_hyper(fws_combined["G"], fws_combined["z"]))
    sigma_z_par = np.asarray(sigma_at_z_parallel(par_combined["P"], par_combined["z"]))

    # G_leaf pairwise cosine across leaf-ids (hyper only).
    g_leaf_cosines = _g_leaf_cosine_matrix(fws_combined["G"], fws_combined["z"])

    hess_idx = rng.choice(train_x.shape[0], size=EVAL_HESS_BATCH, replace=False)
    hess_batch = {"x": jnp.asarray(train_x[hess_idx]), "y": jnp.asarray(train_y[hess_idx])}
    hess_fws = float(np.asarray(hessian_top(final_fws_W, hess_batch))[0])
    hess_par = float(np.asarray(hessian_top(final_par_W, hess_batch))[0])
    hess_w = float(np.asarray(hessian_top(final_w_W, hess_batch))[0])
    hess_op = float(np.asarray(hessian_top(final_op_W, hess_batch))[0])

    def _all_test_preds(params: dict[str, Array]) -> np.ndarray:
        chunk = 1024
        n = test_x.shape[0]
        preds = np.empty(n, dtype=np.int32)
        for i in range(0, n, chunk):
            bx = jnp.asarray(test_x[i:i + chunk])
            logits = jax.vmap(lambda x: cnn_forward(params, x))(bx)
            preds[i:i + chunk] = np.asarray(jnp.argmax(logits, axis=-1))
        return preds

    fws_preds = _all_test_preds(final_fws_W)
    par_preds = _all_test_preds(final_par_W)
    w_preds = _all_test_preds(final_w_W)
    op_preds = _all_test_preds(final_op_W)

    g_leaf_params = {
        name: {
            k: np.asarray(v) for k, v in
            fws_combined["G"].produce(fws_combined["z"], LEAF_ORDER.index(name), LEAF_RANKS[name]).items()
        }
        for name in LEAF_ORDER
    }

    any_nan = any(
        not np.isfinite(t[1])
        for arr in (fws_losses, par_losses, w_losses, op_losses)
        for t in arr
    )

    return {
        "seed": seed,
        "fws_losses": np.array(fws_losses),
        "par_losses": np.array(par_losses),
        "w_losses": np.array(w_losses),
        "op_losses": np.array(op_losses),
        "fws_ckpts": np.array(fws_ckpts),
        "par_ckpts": np.array(par_ckpts),
        "w_ckpts": np.array(w_ckpts),
        "op_ckpts": np.array(op_ckpts),
        "fws_alphas": fws_alphas, "par_alphas": par_alphas,
        "w_alphas": w_alphas, "op_alphas": op_alphas,
        "sigma_z_hyper": sigma_z_hyper, "sigma_gh_hyper": sigma_gh_hyper,
        "sigma_z_par": sigma_z_par,
        "g_leaf_cosines": g_leaf_cosines,
        "hess_fws": hess_fws, "hess_par": hess_par, "hess_w": hess_w, "hess_op": hess_op,
        "any_nan": any_nan,
        "fws_test_preds": fws_preds, "par_test_preds": par_preds,
        "w_test_preds": w_preds, "op_test_preds": op_preds,
        "fws_W": {k: np.asarray(v) for k, v in final_fws_W.items()},
        "par_W": {k: np.asarray(v) for k, v in final_par_W.items()},
        "w_W": {k: np.asarray(v) for k, v in final_w_W.items()},
        "op_W": {k: np.asarray(v) for k, v in final_op_W.items()},
        "g_leaf_params": g_leaf_params,
        "final_fws_acc": float(fws_ckpts[-1][2]),
        "final_par_acc": float(par_ckpts[-1][2]),
        "final_w_acc": float(w_ckpts[-1][2]),
        "final_op_acc": float(op_ckpts[-1][2]),
    }


def _g_leaf_cosine_matrix(G_H: HyperRenderer, z: Array) -> np.ndarray:
    """Pairwise cosine between flat G_leaf parameter vectors across all
    eight leaves. Rank-1 leaves use the first 97 entries, rank-2 the first
    105, rank-4 the first 121 — so we just take the prefix corresponding
    to the leaf's own rank for the comparison and zero-pad shorter ones
    to the longest length.
    """
    flats: list[np.ndarray] = []
    for name in LEAF_ORDER:
        leaf_id = LEAF_ORDER.index(name)
        rank = LEAF_RANKS[name]
        flat = np.asarray(G_H.produce_flat(z, leaf_id))[:G_LEAF_PARAM_SIZE[rank]]
        padded = np.zeros(MAX_G_LEAF_PARAM_SIZE, dtype=flat.dtype)
        padded[:flat.shape[0]] = flat
        flats.append(padded)
    M = np.stack(flats, axis=0)
    n = M @ M.T
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return n / (norms @ norms.T + 1e-30)


# --- Plots -------------------------------------------------------------------
ARMS = ("fws_hyper", "fws_parallel", "w_matched", "w_overparam")
ARM_LABELS = {
    "fws_hyper": "FWS-hyper",
    "fws_parallel": "FWS-parallel-no-G_H",
    "w_matched": "W matched",
    "w_overparam": "W overparam",
}
ARM_COLOURS = {
    "fws_hyper": FWS_HYPER_COLOUR,
    "fws_parallel": FWS_PAR_COLOUR,
    "w_matched": W_MATCHED_COLOUR,
    "w_overparam": W_OVERPARAM_COLOUR,
}
ARM_LOSS_KEY = {
    "fws_hyper": "fws_losses", "fws_parallel": "par_losses",
    "w_matched": "w_losses", "w_overparam": "op_losses",
}
ARM_CKPT_KEY = {
    "fws_hyper": "fws_ckpts", "fws_parallel": "par_ckpts",
    "w_matched": "w_ckpts", "w_overparam": "op_ckpts",
}
ARM_ALPHA_KEY = {
    "fws_hyper": "fws_alphas", "fws_parallel": "par_alphas",
    "w_matched": "w_alphas", "w_overparam": "op_alphas",
}
ARM_ACC_KEY = {
    "fws_hyper": "final_fws_acc", "fws_parallel": "final_par_acc",
    "w_matched": "final_w_acc", "w_overparam": "final_op_acc",
}
ARM_PREDS_KEY = {
    "fws_hyper": "fws_test_preds", "fws_parallel": "par_test_preds",
    "w_matched": "w_test_preds", "w_overparam": "op_test_preds",
}
ARM_W_KEY = {
    "fws_hyper": "fws_W", "fws_parallel": "par_W",
    "w_matched": "w_W", "w_overparam": "op_W",
}


def plot_loss_trajectories(per_seed: list[dict], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for arm in ARMS:
        key = ARM_LOSS_KEY[arm]; colour = ARM_COLOURS[arm]
        for r in per_seed:
            ax.plot(r[key][:, 0], r[key][:, 1], color=colour, alpha=0.3, linewidth=1.0)
        stack = np.stack([r[key][:, 1] for r in per_seed], axis=0)
        steps = per_seed[0][key][:, 0]
        ax.plot(steps, np.median(stack, axis=0), color=colour, linewidth=2.0, label=ARM_LABELS[arm])
    ax.set_yscale("log")
    ax.set(xlabel="outer step", ylabel="training cross-entropy",
           title=f"Phase 8 — training-loss trajectories (K={K_SEED})")
    ax.legend(loc="best", fontsize=9); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_acc_trajectories(per_seed: list[dict], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for arm in ARMS:
        key = ARM_CKPT_KEY[arm]; colour = ARM_COLOURS[arm]
        for r in per_seed:
            ax.plot(r[key][:, 0], r[key][:, 2], color=colour, alpha=0.4, linewidth=1.0)
        stack = np.stack([r[key][:, 2] for r in per_seed], axis=0)
        ep = per_seed[0][key][:, 0]
        ax.plot(ep, np.median(stack, axis=0), color=colour, linewidth=2.0, label=f"{ARM_LABELS[arm]} (median)")
    ax.set(xlabel="epoch", ylabel="test accuracy",
           title=f"Phase 8 — test-accuracy trajectories (K={K_SEED})")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def _alpha_panel(ax, per_seed, leaf: str, title: str) -> None:
    data = [[r[ARM_ALPHA_KEY[arm]][leaf][0] for r in per_seed] for arm in ARMS]
    labels = [ARM_LABELS[a].replace("FWS-parallel-no-G_H", "FWS-par") for a in ARMS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False, labels=labels)
    cols = [ARM_COLOURS[a] for a in ARMS]
    for patch, c in zip(bp["boxes"], cols, strict=True):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    for ml, c in zip(bp["medians"], cols, strict=True):
        ml.set_color(c); ml.set_linewidth(2.0)
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(data, cols, strict=True), start=1):
        v = np.asarray(vals, dtype=float); v = v[np.isfinite(v)]
        if v.size == 0: continue
        jitter = rng.uniform(-0.10, 0.10, size=v.size)
        ax.scatter(np.full_like(v, i) + jitter, v, color=c, s=30, zorder=3, edgecolor="white", linewidths=0.6)
    ax.set_title(title, fontsize=10); ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=7)


def plot_htsr_box(per_seed: list[dict], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    _alpha_panel(axes[0], per_seed, "fc1", "HT-SR α (fc1)")
    _alpha_panel(axes[1], per_seed, "fc2", "HT-SR α (fc2)")
    fig.suptitle(f"Phase 8 — HT-SR α on fc leaves (K={K_SEED})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_fft_box(per_seed: list[dict], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    _alpha_panel(axes[0], per_seed, "conv1", "radial-FFT α (conv1)")
    _alpha_panel(axes[1], per_seed, "conv2", "radial-FFT α (conv2)")
    fig.suptitle(f"Phase 8 — radial-FFT α on conv leaves (K={K_SEED}; k=5 under-recovery caveat)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_final_acc_box(per_seed: list[dict], save_path: Path) -> None:
    data = [[r[ARM_ACC_KEY[a]] for r in per_seed] for a in ARMS]
    cols = [ARM_COLOURS[a] for a in ARMS]
    labels = [ARM_LABELS[a].replace("FWS-parallel-no-G_H", "FWS-par") for a in ARMS]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False, labels=labels)
    for patch, c in zip(bp["boxes"], cols, strict=True):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    for ml, c in zip(bp["medians"], cols, strict=True):
        ml.set_color(c); ml.set_linewidth(2.0)
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(data, cols, strict=True), start=1):
        v = np.asarray(vals, dtype=float)
        jitter = rng.uniform(-0.10, 0.10, size=v.size)
        ax.scatter(np.full_like(v, i) + jitter, v, color=c, s=40, zorder=3, edgecolor="white", linewidths=0.6)
    ax.set_ylabel("final test accuracy")
    ax.set_title(f"Phase 8 — final test accuracy (K={K_SEED}, {EPOCHS} epochs)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_jacobian_spectrum(per_seed: list[dict], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ks = np.arange(1, SIGMA_TOP_K + 1)
    for r in per_seed:
        ax.plot(ks, r["sigma_z_hyper"], color=FWS_HYPER_COLOUR, alpha=0.55, linewidth=1.0)
        ax.plot(ks, r["sigma_gh_hyper"], color=FWS_HYPER_COLOUR, alpha=0.35, linewidth=1.0, linestyle="--")
        ax.plot(ks, r["sigma_z_par"], color=FWS_PAR_COLOUR, alpha=0.55, linewidth=1.0)
    ax.plot([], [], color=FWS_HYPER_COLOUR, linewidth=2.0, label="hyper: σ(∂render/∂z)")
    ax.plot([], [], color=FWS_HYPER_COLOUR, linewidth=2.0, linestyle="--", label="hyper: σ(∂render/∂G_H)")
    ax.plot([], [], color=FWS_PAR_COLOUR, linewidth=2.0, label="parallel: σ(∂render/∂z)")
    ax.set(xlabel="singular-value rank (1 = largest)", ylabel="σ",
           title=f"Phase 8 — Jacobian σ-spectrum at final, top-{SIGMA_TOP_K} (K={K_SEED})")
    ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


# --- Curation plots ----------------------------------------------------------
def _select_reps(per_seed: list[dict], key: str) -> dict[str, dict]:
    s = sorted(per_seed, key=lambda r: r[key], reverse=True)
    return {"best": s[0], "median": s[len(s) // 2], "worst": s[-1]}


def plot_conv_kernels(per_seed: list[dict], arm: str, save_path: Path) -> None:
    rep = _select_reps(per_seed, ARM_ACC_KEY[arm])
    row_labels = ["best", "median", "worst"]
    fig, axes = plt.subplots(3, CONV1_OUT, figsize=(CONV1_OUT * 1.0, 3.4))
    for row, label in enumerate(row_labels):
        r = rep[label]
        c1 = r[ARM_W_KEY[arm]]["conv1_w"]                # (8, 3, 5, 5)
        for k in range(CONV1_OUT):
            f = c1[k].transpose(1, 2, 0)                # (5, 5, 3)
            mn, mx = float(f.min()), float(f.max())
            f_n = (f - mn) / (mx - mn + 1e-9)
            axes[row, k].imshow(f_n, interpolation="nearest")
            axes[row, k].set_xticks([]); axes[row, k].set_yticks([])
            if k == 0:
                axes[row, k].set_ylabel(f"{label}\nseed {r['seed']}\nacc {r[ARM_ACC_KEY[arm]]:.3f}",
                                        fontsize=7)
    fig.suptitle(f"Phase 8 — {ARM_LABELS[arm]} conv1 kernels (8 × 5×5 RGB, per-filter normalised)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(save_path, dpi=150); plt.close(fig)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    return cm


def plot_arm_curation(per_seed: list[dict], arm: str, test_y: np.ndarray, raw_test_x: np.ndarray,
                      save_path: Path) -> None:
    rep = _select_reps(per_seed, ARM_ACC_KEY[arm])
    row_labels = ["best", "median", "worst"]
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(test_y.shape[0], size=12, replace=False)

    fig = plt.figure(figsize=(15.0, 11.0))
    gs = fig.add_gridspec(3, 15, wspace=0.4, hspace=0.5)

    for row, label in enumerate(row_labels):
        r = rep[label]
        preds = r[ARM_PREDS_KEY[arm]]
        ax_cm = fig.add_subplot(gs[row, 0:3])
        cm = _confusion_matrix(test_y, preds)
        ax_cm.imshow(cm, cmap="Blues", aspect="auto")
        ax_cm.set_title(f"{label} (seed {r['seed']}, acc {r[ARM_ACC_KEY[arm]]:.3f})\nconfusion",
                        fontsize=8)
        ax_cm.set_xticks(range(NUM_CLASSES)); ax_cm.set_xticklabels(CIFAR_NAMES, rotation=90, fontsize=6)
        ax_cm.set_yticks(range(NUM_CLASSES)); ax_cm.set_yticklabels(CIFAR_NAMES, fontsize=6)
        for k, si in enumerate(sample_idx):
            col = 3 + k
            ax = fig.add_subplot(gs[row, col])
            img = raw_test_x[si].transpose(1, 2, 0)
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            correct = bool(preds[si] == test_y[si])
            colour = WONG[3] if correct else WONG[6]
            ax.set_title(f"T:{CIFAR_NAMES[test_y[si]]}\nP:{CIFAR_NAMES[preds[si]]}",
                         fontsize=5, color=colour)
            for sp in ax.spines.values():
                sp.set_edgecolor(colour); sp.set_linewidth(1.2)
    fig.suptitle(f"Phase 8 — {ARM_LABELS[arm]} curation (best/median/worst by final test acc, K={K_SEED})",
                 fontsize=11)
    fig.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_fws_g_leaf_panel(per_seed: list[dict], save_path: Path) -> None:
    rep = _select_reps(per_seed, "final_fws_acc")
    chosen = rep["median"]
    g_leaf = chosen["g_leaf_params"]

    H = G_LEAF_HIDDEN_DIM
    fig, axes = plt.subplots(N_LEAVES, 3, figsize=(7.5, 1.0 * N_LEAVES + 0.5))
    for row, leaf_name in enumerate(LEAF_ORDER):
        p = g_leaf[leaf_name]; rank = LEAF_RANKS[leaf_name]
        panels = [(f"W_in ({H}×{rank})", p["W_in"]),
                  (f"W_h ({H}×{H})", p["W_h"]),
                  (f"W_out (1×{H})", p["W_out"])]
        for col, (kn, arr) in enumerate(panels):
            M = arr if arr.ndim == 2 else arr.reshape(1, -1)
            vmax = float(np.percentile(np.abs(M), 95))
            vmax = vmax if vmax > 0 else (float(np.abs(M).max()) or 1.0)
            axes[row, col].imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            axes[row, col].set_title(f"{leaf_name} — {kn}", fontsize=7)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    fig.suptitle(f"Phase 8 — per-leaf G_leaf params (FWS-hyper median seed {chosen['seed']})",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(save_path, dpi=130); plt.close(fig)


def plot_g_leaf_cosine(per_seed: list[dict], save_path: Path) -> None:
    rep = _select_reps(per_seed, "final_fws_acc")
    chosen = rep["median"]
    M = chosen["g_leaf_cosines"]
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(M, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(N_LEAVES)); ax.set_xticklabels(LEAF_ORDER, rotation=60, fontsize=7)
    ax.set_yticks(range(N_LEAVES)); ax.set_yticklabels(LEAF_ORDER, fontsize=7)
    ax.set_title(f"FWS-hyper: pairwise G_leaf cosine (median seed {chosen['seed']})", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


# --- Research log helpers ----------------------------------------------------
def median_iqr_min_max(xs: np.ndarray) -> tuple[float, float, float, float]:
    return (float(np.median(xs)),
            float(np.quantile(xs, 0.75) - np.quantile(xs, 0.25)),
            float(xs.min()),
            float(xs.max()))


# --- Main --------------------------------------------------------------------
def main() -> None:
    stage = os.environ.get("PHASE8_STAGE", "all")

    print("=" * 72)
    print(f"Phase 8 — CIFAR-10 fiducial cell  (stage={stage})")
    print("=" * 72)
    print("Loading CIFAR-10 ...")
    train_x, train_y, test_x, test_y = load_cifar10()
    raw_root = _download_cifar()
    raw_test_x = _load_pickle_batch(raw_root / "test_batch")[0]
    print(f"  train: {train_x.shape} {train_y.shape} | test: {test_x.shape} {test_y.shape}")

    init_G_d, init_z_d, init_par_d, init_par_z_d, init_W_d, init_op_d = make_inits(0)
    fws_n = count_params({"G": init_G_d, "z": init_z_d})
    par_n = count_params({"P": init_par_d, "z": init_par_z_d})
    w_n = count_params(init_W_d)
    op_n = count_params(init_op_d)
    print(f"\nG_leaf param size per rank: {G_LEAF_PARAM_SIZE}  (max={MAX_G_LEAF_PARAM_SIZE})")
    print(f"G_H hidden={G_H_HIDDEN_DIM}, z={DIM_Z}, leaf_emb=({N_LEAVES},{DIM_LEAF_EMB})")
    print(f"FWS-hyper trainable          : {fws_n}")
    print(f"FWS-parallel-no-G_H trainable: {par_n}")
    print(f"W matched (mainnet)          : {w_n}")
    print(f"W overparam (factored)       : {op_n}")
    print(f"K_seed={K_SEED}, epochs={EPOCHS}, batch={BATCH_SIZE}, falsifier_steps={FALSIFIER_STEPS}")
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Stage 0: falsifier --------------------------------------------------
    stage0_records: dict[str, list[tuple[int, float]]] | None = None
    proceed_to_k3 = True
    stage0_text = ""
    if stage in ("falsifier", "all"):
        t0 = time.time()
        stage0_records = run_stage0_falsifier(train_x, train_y, seed=0)
        proceed_to_k3, stage0_text = stage0_verdict(stage0_records)
        print("\n" + stage0_text)
        print(f"Stage 0 wall: {time.time() - t0:.1f}s")
        plot_stage0(stage0_records, FIGURES_DIR / "2026-06-07-phase8-stage0-falsifier.png")

    # ---- Stage 1: K=3 K=seed --------------------------------------------------
    per_seed: list[dict] = []
    nan_or_crash: list[str] = []
    k3_wall = 0.0
    if stage in ("k3", "all"):
        if not proceed_to_k3 and stage == "all":
            print("\nSTOP — Stage 0 verdict says init recovery, not FWS prior. K=3 skipped.")
        else:
            t0 = time.time()
            for seed in range(K_SEED):
                t_seed = time.time()
                print(f"\n--- seed {seed} ---", flush=True)
                try:
                    row = run_seed(seed, train_x, train_y, test_x, test_y)
                    per_seed.append(row)
                    if row["any_nan"]:
                        nan_or_crash.append(f"seed={seed}: NaN in a loss trajectory")
                    print(f"  seed {seed} done: hyper={row['final_fws_acc']:.4f} "
                          f"par={row['final_par_acc']:.4f} w={row['final_w_acc']:.4f} "
                          f"op={row['final_op_acc']:.4f}  ({time.time() - t_seed:.1f}s)", flush=True)
                except Exception as e:  # noqa: BLE001
                    nan_or_crash.append(f"seed={seed}: CRASH — {type(e).__name__}: {e}")
                    print(f"  seed {seed} CRASH — {type(e).__name__}: {e}", flush=True)
            k3_wall = time.time() - t0

    # ---- Summary + plots + report -------------------------------------------
    if per_seed:
        fws_a = median_iqr_min_max(np.array([r["final_fws_acc"] for r in per_seed]))
        par_a = median_iqr_min_max(np.array([r["final_par_acc"] for r in per_seed]))
        w_a = median_iqr_min_max(np.array([r["final_w_acc"] for r in per_seed]))
        op_a = median_iqr_min_max(np.array([r["final_op_acc"] for r in per_seed]))
        print("\nFinal test accuracy (verbatim):")
        print(f"  FWS-hyper           : median={fws_a[0]:.4f} IQR={fws_a[1]:.4f} min={fws_a[2]:.4f} max={fws_a[3]:.4f}")
        print(f"  FWS-parallel-no-G_H : median={par_a[0]:.4f} IQR={par_a[1]:.4f} min={par_a[2]:.4f} max={par_a[3]:.4f}")
        print(f"  W matched           : median={w_a[0]:.4f} IQR={w_a[1]:.4f} min={w_a[2]:.4f} max={w_a[3]:.4f}")
        print(f"  W overparam         : median={op_a[0]:.4f} IQR={op_a[1]:.4f} min={op_a[2]:.4f} max={op_a[3]:.4f}")

        plot_loss_trajectories(per_seed, FIGURES_DIR / "2026-06-07-phase8-loss-trajectories.png")
        plot_acc_trajectories(per_seed, FIGURES_DIR / "2026-06-07-phase8-acc-trajectories.png")
        plot_htsr_box(per_seed, FIGURES_DIR / "2026-06-07-phase8-htsr-alpha-boxplot.png")
        plot_fft_box(per_seed, FIGURES_DIR / "2026-06-07-phase8-radial-fft-alpha-boxplot.png")
        plot_final_acc_box(per_seed, FIGURES_DIR / "2026-06-07-phase8-final-acc-boxplot.png")
        plot_jacobian_spectrum(per_seed, FIGURES_DIR / "2026-06-07-phase8-jacobian-spectrum.png")
        for arm in ARMS:
            plot_conv_kernels(per_seed, arm,
                              FIGURES_DIR / f"2026-06-07-phase8-conv-kernels-{arm}.png")
            plot_arm_curation(per_seed, arm, test_y, raw_test_x,
                              FIGURES_DIR / f"2026-06-07-phase8-curations-{arm}.png")
        plot_fws_g_leaf_panel(per_seed, FIGURES_DIR / "2026-06-07-phase8-fws-hyper-leaves.png")
        plot_g_leaf_cosine(per_seed, FIGURES_DIR / "2026-06-07-phase8-g_leaf-cosine.png")

    nan_summary = "\n".join(f"- {line}" for line in nan_or_crash) if nan_or_crash else "- none."
    stage0_md = ""
    if stage0_records is not None:
        rows = []
        for kind in SCALE_KINDS:
            recs = stage0_records[kind]
            row = f"| {kind} | " + " | ".join(f"{v:.4g}" for _, v in recs) + " |"
            rows.append(row)
        steps_hdr = " | ".join(f"step {s}" for s in FALSIFIER_CHECKPOINTS)
        stage0_md = (
            f"\n## Stage 0 — G_H W_out init-scale falsifier (BLOCKING)\n\n"
            f"For each ``G_H`` ``W_out`` init scale we ran {FALSIFIER_STEPS} FWS-hyper\n"
            f"outer steps (seed 0) and probed ``σ_min(∂render/∂z)`` at the\n"
            f"checkpoints {FALSIFIER_CHECKPOINTS}.\n\n"
            f"| scale_kind | {steps_hdr} |\n"
            f"|---|" + "|".join(["---"] * len(FALSIFIER_CHECKPOINTS)) + "|\n"
            + "\n".join(rows)
            + f"\n\n**Verdict (decision rule: ≥1 OoM spread at final step → proceed):**\n\n"
            + f"```\n{stage0_text}\n```\n"
            + "\n![Stage 0 falsifier](figures/2026-06-07-phase8-stage0-falsifier.png)\n"
        )

    per_seed_rows = "\n".join(
        f"| {r['seed']} | {r['final_fws_acc']:.4f} | {r['final_par_acc']:.4f} | "
        f"{r['final_w_acc']:.4f} | {r['final_op_acc']:.4f} | "
        f"{r['fws_alphas']['conv1'][0]:.3g} | {r['fws_alphas']['conv2'][0]:.3g} | "
        f"{r['fws_alphas']['fc1'][0]:.3g} | {r['fws_alphas']['fc2'][0]:.3g} |"
        for r in per_seed
    )

    k3_md = ""
    if per_seed:
        k3_md = f"""
## Stage 1 — K={K_SEED} fiducial cell

### Final test accuracy (verbatim)

| arm | median | IQR | min | max |
|---|---|---|---|---|
| FWS-hyper            | {fws_a[0]:.4f} | {fws_a[1]:.4f} | {fws_a[2]:.4f} | {fws_a[3]:.4f} |
| FWS-parallel-no-G_H  | {par_a[0]:.4f} | {par_a[1]:.4f} | {par_a[2]:.4f} | {par_a[3]:.4f} |
| W matched            | {w_a[0]:.4f} | {w_a[1]:.4f} | {w_a[2]:.4f} | {w_a[3]:.4f} |
| W overparam          | {op_a[0]:.4f} | {op_a[1]:.4f} | {op_a[2]:.4f} | {op_a[3]:.4f} |

### Per-seed table

| seed | FWS-hyper acc | FWS-par acc | W matched acc | W overparam acc | α conv1 (hyper) | α conv2 (hyper) | α fc1 (hyper) | α fc2 (hyper) |
|---|---|---|---|---|---|---|---|---|
{per_seed_rows}

![loss trajectories](figures/2026-06-07-phase8-loss-trajectories.png)

![test accuracy trajectories](figures/2026-06-07-phase8-acc-trajectories.png)

![HT-SR α (fc leaves)](figures/2026-06-07-phase8-htsr-alpha-boxplot.png)

![radial-FFT α (conv leaves)](figures/2026-06-07-phase8-radial-fft-alpha-boxplot.png)

![final test accuracy](figures/2026-06-07-phase8-final-acc-boxplot.png)

![Jacobian σ-spectrum at convergence](figures/2026-06-07-phase8-jacobian-spectrum.png)

### Curated outputs

Per-arm best/median/worst by final test accuracy (pre-registered, not eyeballed).

![FWS-hyper conv1 kernels](figures/2026-06-07-phase8-conv-kernels-fws_hyper.png)

![FWS-parallel-no-G_H conv1 kernels](figures/2026-06-07-phase8-conv-kernels-fws_parallel.png)

![W matched conv1 kernels](figures/2026-06-07-phase8-conv-kernels-w_matched.png)

![W overparam conv1 kernels](figures/2026-06-07-phase8-conv-kernels-w_overparam.png)

![FWS-hyper curation](figures/2026-06-07-phase8-curations-fws_hyper.png)

![FWS-parallel-no-G_H curation](figures/2026-06-07-phase8-curations-fws_parallel.png)

![W matched curation](figures/2026-06-07-phase8-curations-w_matched.png)

![W overparam curation](figures/2026-06-07-phase8-curations-w_overparam.png)

![Per-leaf G_leaf params (FWS-hyper, median seed)](figures/2026-06-07-phase8-fws-hyper-leaves.png)

![FWS-hyper pairwise G_leaf cosine](figures/2026-06-07-phase8-g_leaf-cosine.png)
"""

    md = f"""# Phase 8 — CIFAR-10 fiducial cell — 2026-06-07

## Background

Phases 1–6 ran on a synthetic 2-layer SiLU teacher–student task. Phase 7
introduced the per-leaf hyper-renderer; phases 8–9 lifted it to MNIST and
then CIFAR-10. Per user direction this **renumbered Phase 8** is the
design-doc *fiducial cell*: CIFAR-10 + WideKernelCNN-SiLU with two
architecture corrections and three engineering-team-derived controls.

## Architecture corrections (vs phase 9)

1. **Natural-rank coord scheme** — ``G_leaf`` takes a coordinate of
   dimension equal to the leaf's rank: 1 for biases, 2 for fc weights, 4
   for conv kernels. There are three distinct ``G_leaf`` templates rather
   than a single rank-4-with-zero-padding template.

2. **No separate linear readout on G_leaf** — the last layer is
   ``sin(W_out @ hidden + b_out)`` with ``W_out: (1, hidden_dim)``
   producing the scalar output directly. No post-trunk linear projection.

## Mainnet (WideKernelCNN-SiLU)

- conv1: ``({CONV1_OUT}, {IN_CHANNELS}, {KERNEL_SIZE}, {KERNEL_SIZE})`` + bias ``({CONV1_OUT},)``
- SiLU + maxpool 2×2 → ``({CONV1_OUT}, 14, 14)``
- conv2: ``({CONV2_OUT}, {CONV1_OUT}, {KERNEL_SIZE}, {KERNEL_SIZE})`` + bias ``({CONV2_OUT},)``
- SiLU + maxpool 2×2 → ``({CONV2_OUT}, 5, 5)``  (flatten → {FC1_IN_DIM})
- fc1: ``({FC1_HIDDEN}, {FC1_IN_DIM})`` + bias ``({FC1_HIDDEN},)``  SiLU
- fc2: ``({NUM_CLASSES}, {FC1_HIDDEN})`` + bias ``({NUM_CLASSES},)``  (logits)

Total mainnet trainable parameters: **{w_n}**. Kernel size $k=5$ is large
enough to engage the radial-FFT α diagnostic, with the documented
under-recovery bias at small $k$ flagged in the caveats.

## Arms

| arm | renderer | trainable params |
|---|---|---|
| FWS-hyper            | $G_H$ → per-leaf $G_{{\\text{{leaf}}}}$ from sliced flat output | {fws_n} |
| FWS-parallel-no-$G_H$ | 3 independent $G_{{\\text{{leaf}}}}$ (rank 1, 2, 4), shared $z$ FiLMs each | {par_n} |
| W matched            | direct WideKernelCNN-SiLU | {w_n} |
| W overparam          | factored conv (rank-$P$ depthwise+pointwise) + factored fc | {op_n} |

### G_leaf details

Depth-2 sine-only MLP, hidden $H = {G_LEAF_HIDDEN_DIM}$, $\\omega_0 =
{OMEGA_0}$ on both layers, ``W_out: (1, H)``. Forward pass at coord $c$
is ``sin(W_out · sin(W_h · sin(omega_0 W_in · c + b_in) + b_h) + b_out)``
(scalar). Per-rank parameter sizes: rank 1 → {G_LEAF_PARAM_SIZE[1]},
rank 2 → {G_LEAF_PARAM_SIZE[2]}, rank 4 → {G_LEAF_PARAM_SIZE[4]}.

### G_H details

2-layer SiLU MLP with hidden dim ``{G_H_HIDDEN_DIM}``; input is ``[z,
leaf_emb[id]]`` (``DIM_Z={DIM_Z}``, ``DIM_LEAF_EMB={DIM_LEAF_EMB}``,
``N_LEAVES={N_LEAVES}``); output is a flat vector of length ``{MAX_G_LEAF_PARAM_SIZE}``
(``max_l G_LEAF_PARAM_SIZE[l]``). Per leaf, take the first
``G_LEAF_PARAM_SIZE[leaf_rank]`` entries and unpack into the
rank-appropriate ``G_leaf`` template.

### FWS-parallel-no-G_H details

Three independent ``G_leaf`` instances (one per distinct rank ∈ {{1, 2,
4}}), each initialised under the SIREN-paper scheme. Shared
$z \\in \\mathbb{{R}}^{{{DIM_Z}}}$ is mapped to per-rank
``(gamma, beta)`` via a linear layer of shape ``(2 H, DIM_Z)``, applied at
the hidden layer of the rank-matched ``G_leaf``. No $G_H$ — the per-leaf
parameters are directly trainable.

**Note on param counts**: FWS-hyper (~{fws_n}) ≫ FWS-parallel (~{par_n}).
We deliberately keep ``G_LEAF_HIDDEN_DIM = {G_LEAF_HIDDEN_DIM}`` constant
across both FWS arms to make the *architecture* the controlled axis
between them; the question is "is ``G_H`` load-bearing", and conflating
that with an enlarged ``G_leaf`` would defeat the ablation. The relevant
compute-matched comparator for FWS-hyper is the **W matched** arm
(matching mainnet param count), not FWS-parallel.

## Coord normalisation

Anisotropic ``[-1, 1]`` per leaf rank-axis. Rank-1 biases get a 1-D
linspace coord; rank-2 fc weights get an ``(o, i)`` grid; rank-4 conv
kernels get an ``(oc, ic, kh, kw)`` grid. Coords carry the leaf's natural
shape; the leaf identity flows through the learned leaf embedding into
``G_H``.

## Setup

- Dataset: CIFAR-10 (50 000 train / 10 000 test, 32×32 RGB), cached to
  ``~/.cache/cifar10``, per-channel mean/std normalised using train
  statistics. Source: ``cs.toronto.edu/~kriz/cifar-10-python.tar.gz``.
- Loss: softmax cross-entropy on logits.
- Optimisers: Adam(``{G_LR}``) on $G_H$; Adam(``{Z_LR}``) on $z$
  (``optax.multi_transform``); Adam(``{PARALLEL_LR}``) on the parallel
  arm; Adam(``{W_LR}``) on $W$ matched; Adam(``{OP_LR}``) on $W$ overparam.
- Schedule: ``epochs = {EPOCHS}``, ``batch_size = {BATCH_SIZE}``.
- $K_{{\\text{{seed}}}} = {K_SEED}$ (smoke, not K=10 confirmation).
{stage0_md}{k3_md}
## NaN / crash report

{nan_summary}

## Honest caveats

- **Fiducial cell, K=3 smoke** — not K=10 confirmation. Per-seed
  variance not characterised.
- **5 epochs only.** CIFAR-10 typically trains 50–200 epochs; absolute
  accuracies are not benchmark-comparable. The relative ordering across
  arms at fixed-budget early training is the legitimate signal.
- **Single architecture per arm.** No conv-topology sweep.
- **W overparam factorisation choice.** Depth-wise + point-wise rank-$P$
  for conv; rank-$P$ stacked factors for fc. Alternative factorisations
  (over-complete, full-rank stacked, BTT, Tucker) untested.
- **Radial-FFT α at k=5 has known under-recovery bias** ([Trello
  LYBZKFDi](https://trello.com/c/LYBZKFDi)). Read conv-leaf α values
  alongside their R² fit quality, not at face value.
- **FWS-parallel arm has a much smaller param count than FWS-hyper.** We
  deliberately matched the *architecture* (G_leaf shape) instead of the
  param count to keep "is G_H load-bearing" the controlled axis. The
  compute-matched comparator for FWS-hyper is W matched.
- **σ probes are top-{SIGMA_TOP_K} power iteration**, not the global
  spectrum; ``σ_min`` is the smallest of the top-{SIGMA_TOP_K}.
- **Hessian top eigenvalue** computed on a {EVAL_HESS_BATCH}-image
  subsample.
- **Wall-clock budget**: stage 0 ≈ varied; stage 1 K=3 ≈ {k3_wall:.1f}s.

## What's next

Triaged depending on Stage 0 verdict:

- If Stage 0 said *init recovery*: report the failure verbatim. Either
  (a) try architectures that further separate from phase 6's failure
  mode (multiplicative coupling between $G_H$ and $G_{{\\text{{leaf}}}}$,
  or learned coord encodings), or (b) move to a different mainnet that
  exercises the FWS prior differently.
- If Stage 0 said *FWS prior doing geometric work* and Stage 1 ran:
  K=10 confirmation at the same budget, then a longer-training run
  (50 epochs) to see whether the early-training ordering survives
  convergence.
"""
    RESEARCH_FILE.write_text(md)
    print(f"\nWrote research log to {RESEARCH_FILE}")


if __name__ == "__main__":
    main()
