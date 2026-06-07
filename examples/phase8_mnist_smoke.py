"""Phase 8: MNIST integration smoke with per-leaf hyper-renderer.

Phases 4-7 were a synthetic 2-layer SiLU teacher-student regression task.
Phase 8 graduates to real data: MNIST digit classification with the per-leaf
hyper-renderer architecture from phase 7. This is an *integration smoke* (K=3,
5 epochs) — first time the pipes connect MNIST → mainnet → cross-entropy →
hyper-renderer training → measurement → plotting. K=10 confirmation is a
later phase if the pipes connect cleanly.

Architecture summary (FWS-hyper arm):

- mainnet: 2-layer SiLU classifier ``x (784,) → W1 (128,784) → silu → W2 (10,128) → logits``.
- ``G_H``: hyper-renderer mapping ``(z, leaf_id) → G_leaf params``. Sized so
  the total FWS-hyper trainable param count matches the mainnet (~100K), not
  the 8× balloon used in the synthetic regime.
- ``G_leaf``: depth-2 SIREN over 1-D within-leaf coord, hidden 32, omega_0 = 30.
- ``z``: 64-D modulation latent, flows into ``G_H``.

The W arms (matched + overparam) train the mainnet directly (W matched) or via
a 2-factor low-rank decomposition (W overparam, P chosen so total ≈ 5× matched).

Run::

    PHASE8_K_SEED=1 uv run python examples/phase8_mnist_smoke.py  # single-seed pilot
    uv run python examples/phase8_mnist_smoke.py                  # K=3
"""

from __future__ import annotations

import gzip
import os
import struct
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
RESEARCH_FILE = RESEARCH_DIR / "2026-06-07-phase8-mnist-smoke.md"
MNIST_CACHE = Path.home() / ".cache" / "mnist"
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"

WONG_PALETTE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
FWS_COLOUR = WONG_PALETTE[5]
W_MATCHED_COLOUR = WONG_PALETTE[6]
W_OVERPARAM_COLOUR = WONG_PALETTE[3]

# --- Mainnet topology -------------------------------------------------------
IN_DIM = 784
HIDDEN_DIM_MAIN = 128
OUT_DIM = 10
LEAF_ORDER: tuple[str, ...] = ("W1", "b1", "W2", "b2")
N_LEAVES = len(LEAF_ORDER)
LEAF_SHAPES: dict[str, tuple[int, ...]] = {
    "W1": (HIDDEN_DIM_MAIN, IN_DIM),
    "b1": (HIDDEN_DIM_MAIN,),
    "W2": (OUT_DIM, HIDDEN_DIM_MAIN),
    "b2": (OUT_DIM,),
}
LEAF_SIZES: dict[str, int] = {k: int(np.prod(s)) for k, s in LEAF_SHAPES.items()}
TOTAL_W_PARAMS = sum(LEAF_SIZES.values())  # 101_770

# --- Hyper-renderer hyperparameters (sized for ~100K total FWS-hyper params) ---
DIM_Z = 64
DIM_LEAF_EMB = 8
G_H_HIDDEN_DIM = 80
G_LEAF_COORD_DIM = 1
G_LEAF_HIDDEN_DIM = 32
OMEGA_FIRST = 30.0
OMEGA_HIDDEN = 30.0
G_LR = 1e-3
Z_LR = 1e-3

# --- Over-parameterised arm: P chosen so total ≈ 5× matched ~ 500K --------
# Total = P*(IN_DIM + HIDDEN_DIM + HIDDEN_DIM + OUT_DIM) + biases
#       = P * (784 + 128 + 128 + 10) + 138 = P*1050 + 138 ≈ 500K  ⇒  P ≈ 476
OVERPARAM_P = 476

# --- Training schedule ------------------------------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 5
N_TRAIN = 50_000
N_TEST = 10_000
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE  # 390
NUM_OUTER_STEPS = STEPS_PER_EPOCH * NUM_EPOCHS  # 1950
LOSS_LOG_STRIDE = 100  # log every 100 steps
NUM_CHECKPOINTS = NUM_EPOCHS  # end-of-epoch checkpoints
K_SEED = int(os.environ.get("PHASE8_K_SEED", 3))

# Spectral probes
SIGMA_TOP_K = 10
SIGMA_POWER_ITERS = 60
HESSIAN_POWER_ITERS = 40


# --- MNIST loader (stdlib only, no torchvision) ------------------------------
def _download(name: str) -> Path:
    MNIST_CACHE.mkdir(parents=True, exist_ok=True)
    out = MNIST_CACHE / name
    if not out.exists():
        url = f"{MNIST_MIRROR}/{name}"
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, out)
    return out


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, magic
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols).astype(np.float32) / 255.0


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, magic
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int32)


def load_mnist() -> dict[str, np.ndarray]:
    """Returns dict with x_train, y_train, x_test, y_test as numpy arrays.

    Trims the 60K MNIST train set to the first 50K (brief spec). Pixel values
    are normalised to [0, 1] floats. Labels are int32 in [0, 10).
    """
    x_train_full = _read_idx_images(_download("train-images-idx3-ubyte.gz"))
    y_train_full = _read_idx_labels(_download("train-labels-idx1-ubyte.gz"))
    x_test = _read_idx_images(_download("t10k-images-idx3-ubyte.gz"))
    y_test = _read_idx_labels(_download("t10k-labels-idx1-ubyte.gz"))
    return {
        "x_train": x_train_full[:N_TRAIN],
        "y_train": y_train_full[:N_TRAIN],
        "x_test": x_test[:N_TEST],
        "y_test": y_test[:N_TEST],
    }


# --- Mainnet (the student classifier) ---------------------------------------
def mainnet_forward(params: dict[str, Array], x: Float[Array, " in_dim"]) -> Float[Array, " out_dim"]:
    h = jax.nn.silu(params["W1"] @ x + params["b1"])
    return params["W2"] @ h + params["b2"]


def cross_entropy_loss(params: dict[str, Array], batch: dict) -> Float[Array, ""]:
    logits = jax.vmap(lambda x: mainnet_forward(params, x))(batch["x"])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(logits.shape[0]), batch["y"]])


def accuracy(params: dict[str, Array], batch: dict) -> float:
    logits = jax.vmap(lambda x: mainnet_forward(params, x))(batch["x"])
    preds = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(preds == batch["y"]))


def predictions(params: dict[str, Array], batch: dict) -> np.ndarray:
    logits = jax.vmap(lambda x: mainnet_forward(params, x))(batch["x"])
    return np.asarray(jnp.argmax(logits, axis=-1))


# --- G_leaf: produced-parameter SIREN ----------------------------------------
_G_LEAF_W1_SIZE = G_LEAF_HIDDEN_DIM * G_LEAF_COORD_DIM
_G_LEAF_B1_SIZE = G_LEAF_HIDDEN_DIM
_G_LEAF_W2_SIZE = G_LEAF_HIDDEN_DIM * G_LEAF_HIDDEN_DIM
_G_LEAF_B2_SIZE = G_LEAF_HIDDEN_DIM
_G_LEAF_WOUT_SIZE = G_LEAF_HIDDEN_DIM
_G_LEAF_BOUT_SIZE = 1
G_LEAF_PARAM_SIZE = (
    _G_LEAF_W1_SIZE + _G_LEAF_B1_SIZE + _G_LEAF_W2_SIZE
    + _G_LEAF_B2_SIZE + _G_LEAF_WOUT_SIZE + _G_LEAF_BOUT_SIZE
)


def _slice_g_leaf_params(flat: Float[Array, " G_LEAF_PARAM_SIZE"]) -> dict[str, Array]:
    offset = 0

    def take(size: int, shape: tuple[int, ...]) -> Array:
        nonlocal offset
        chunk = flat[offset : offset + size].reshape(shape)
        offset += size
        return chunk

    W1 = take(_G_LEAF_W1_SIZE, (G_LEAF_HIDDEN_DIM, G_LEAF_COORD_DIM))
    b1 = take(_G_LEAF_B1_SIZE, (G_LEAF_HIDDEN_DIM,))
    W2 = take(_G_LEAF_W2_SIZE, (G_LEAF_HIDDEN_DIM, G_LEAF_HIDDEN_DIM))
    b2 = take(_G_LEAF_B2_SIZE, (G_LEAF_HIDDEN_DIM,))
    w_out = take(_G_LEAF_WOUT_SIZE, (G_LEAF_HIDDEN_DIM,))
    b_out = take(_G_LEAF_BOUT_SIZE, (1,))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "w_out": w_out, "b_out": b_out}


def g_leaf_forward(coord: Float[Array, " G_LEAF_COORD_DIM"], params: dict[str, Array]) -> Float[Array, ""]:
    pre1 = params["W1"] @ coord + params["b1"]
    h1 = jnp.sin(OMEGA_FIRST * pre1)
    pre2 = params["W2"] @ h1 + params["b2"]
    h2 = jnp.sin(OMEGA_HIDDEN * pre2)
    return jnp.dot(params["w_out"], h2) + params["b_out"][0]


# --- G_H: the hyper-renderer (identical structure to phase 7) ----------------
class HyperRenderer(eqx.Module):
    leaf_embedding: Float[Array, "n_leaves dim_leaf_emb"]
    W_in: Float[Array, "hidden in_total"]
    b_in: Float[Array, "hidden"]
    W_out: Float[Array, "g_leaf_param_size hidden"]
    b_out: Float[Array, "g_leaf_param_size"]

    def __init__(self, *, key: Array) -> None:
        k_emb, k_w_in, k_b_in, k_w_out, k_b_out = jax.random.split(key, 5)
        in_total = DIM_Z + DIM_LEAF_EMB
        emb_bound = 1.0 / jnp.sqrt(jnp.array(DIM_LEAF_EMB, dtype=jnp.float32))
        self.leaf_embedding = jax.random.uniform(
            k_emb, (N_LEAVES, DIM_LEAF_EMB), minval=-emb_bound, maxval=emb_bound
        )
        bound_in = jnp.sqrt(jnp.array(6.0 / in_total, dtype=jnp.float32))
        self.W_in = jax.random.uniform(
            k_w_in, (G_H_HIDDEN_DIM, in_total), minval=-bound_in, maxval=bound_in
        )
        self.b_in = jax.random.uniform(
            k_b_in, (G_H_HIDDEN_DIM,), minval=-bound_in, maxval=bound_in
        )
        bound_out = 0.1 / jnp.sqrt(jnp.array(G_H_HIDDEN_DIM, dtype=jnp.float32))
        self.W_out = jax.random.uniform(
            k_w_out, (G_LEAF_PARAM_SIZE, G_H_HIDDEN_DIM), minval=-bound_out, maxval=bound_out
        )
        self.b_out = jax.random.uniform(
            k_b_out, (G_LEAF_PARAM_SIZE,), minval=-bound_out, maxval=bound_out
        )

    def produce(self, z: Float[Array, " dim_z"], leaf_id: int) -> dict[str, Array]:
        emb = self.leaf_embedding[leaf_id]
        inp = jnp.concatenate([z, emb])
        h = jax.nn.silu(self.W_in @ inp + self.b_in)
        flat = self.W_out @ h + self.b_out
        return _slice_g_leaf_params(flat)


def _within_leaf_coord(size: int) -> Float[Array, "size 1"]:
    if size == 1:
        return jnp.zeros((1, 1))
    return jnp.linspace(-1.0, 1.0, size).reshape(size, 1)


LEAF_COORDS: dict[str, Array] = {name: _within_leaf_coord(LEAF_SIZES[name]) for name in LEAF_ORDER}


def hyper_render_fn(G_H: HyperRenderer, z: Float[Array, " dim_z"]) -> dict[str, Array]:
    P = {name: jnp.zeros(shape) for name, shape in LEAF_SHAPES.items()}

    def f(path, shape, dtype, params):
        G_H_in, z_in = params
        leaf_name = path[0].key
        leaf_id = LEAF_ORDER.index(leaf_name)
        leaf_params = G_H_in.produce(z_in, leaf_id)
        coords = LEAF_COORDS[leaf_name]
        values = jax.vmap(lambda c: g_leaf_forward(c, leaf_params))(coords)
        return values.reshape(shape).astype(dtype)

    return loom.render(P, f, (G_H, z))


# --- Over-parameterised arm (low-rank factoring) ----------------------------
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
        self.A_W1_2 = jax.random.normal(k1b, (HIDDEN_DIM_MAIN, P)) * scale
        self.A_W2_1 = jax.random.normal(k2a, (P, HIDDEN_DIM_MAIN)) * scale
        self.A_W2_2 = jax.random.normal(k2b, (OUT_DIM, P)) * scale
        self.b1 = jnp.zeros((HIDDEN_DIM_MAIN,))
        self.b2 = jnp.zeros((OUT_DIM,))

    def materialise(self) -> dict[str, Array]:
        return {
            "W1": self.A_W1_2 @ self.A_W1_1,
            "b1": self.b1,
            "W2": self.A_W2_2 @ self.A_W2_1,
            "b2": self.b2,
        }


# --- Training segment makers (JIT'd scan over a batch sequence) -------------
def _global_l2_norm(grad_tree: PyTree) -> Float[Array, ""]:
    flat, _ = ravel_pytree(grad_tree)
    return jnp.linalg.norm(flat)


def make_fws_segment(G_lr: float, z_lr: float):
    fws_optimiser = optax.multi_transform(
        {"G": optax.adam(G_lr), "z": optax.adam(z_lr)},
        {"G": "G", "z": "z"},
    )

    def loss_fn(combined, batch):
        W_tree = hyper_render_fn(combined["G"], combined["z"])
        return cross_entropy_loss(W_tree, batch)

    def step(carry, batch):
        params, state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        grad_norm = _global_l2_norm(grads)
        updates, new_state = fws_optimiser.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_state), (loss, grad_norm)

    def run(combined, state, batches):
        (final_params, final_state), (loss_traj, grad_traj) = jax.lax.scan(
            step, (combined, state), xs=batches
        )
        return final_params, final_state, loss_traj, grad_traj

    return fws_optimiser, jax.jit(run)


def make_direct_segment(loss_fn_W, optimiser):
    def step(carry, batch):
        params, state = carry
        loss, grads = jax.value_and_grad(loss_fn_W)(params, batch)
        grad_norm = _global_l2_norm(grads)
        updates, new_state = optimiser.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_state), (loss, grad_norm)

    def run(params, state, batches):
        (final_params, final_state), (loss_traj, grad_traj) = jax.lax.scan(
            step, (params, state), xs=batches
        )
        return final_params, final_state, loss_traj, grad_traj

    return jax.jit(run)


# --- Spectral probes -------------------------------------------------------
def sigma_at_z(G_H: HyperRenderer, z: Array, *, seed: int = 7) -> Array:
    operator = lambda z_var: hyper_render_fn(G_H, z_var)  # noqa: E731
    return singular_spectrum(operator, z, k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS, key=jax.random.key(seed))


def sigma_at_gh(G_H: HyperRenderer, z: Array, *, seed: int = 13) -> Array:
    operator = lambda G_var: hyper_render_fn(G_var, z)  # noqa: E731
    return singular_spectrum(operator, G_H, k=SIGMA_TOP_K, num_iterations=SIGMA_POWER_ITERS, key=jax.random.key(seed))


def hessian_top_at(params: PyTree, batch: dict, *, seed: int = 11) -> Array:
    grad_fn = jax.grad(lambda p: cross_entropy_loss(p, batch))
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


def count_params(tree: PyTree) -> int:
    flat, _ = ravel_pytree(tree)
    return int(flat.size)


# --- Per-seed run -----------------------------------------------------------
def make_inits(seed: int):
    key = jax.random.key(seed)
    k_g, k_z, k_w, k_op = jax.random.split(key, 4)
    init_G = HyperRenderer(key=k_g)
    init_z = jax.random.normal(k_z, (DIM_Z,)) * 0.1
    k_w1, k_w2 = jax.random.split(k_w)
    # He-style init for the SiLU mainnet at matched scale
    init_W: dict[str, Array] = {
        "W1": jax.random.normal(k_w1, LEAF_SHAPES["W1"]) * jnp.sqrt(2.0 / IN_DIM),
        "b1": jnp.zeros(LEAF_SHAPES["b1"]),
        "W2": jax.random.normal(k_w2, LEAF_SHAPES["W2"]) * jnp.sqrt(2.0 / HIDDEN_DIM_MAIN),
        "b2": jnp.zeros(LEAF_SHAPES["b2"]),
    }
    init_op = OverparamMLP(P=OVERPARAM_P, key=k_op)
    return init_G, init_z, init_W, init_op


def make_batches(x: np.ndarray, y: np.ndarray, *, key: Array) -> dict[str, Array]:
    """Shuffle (x, y) and return as a (n_batches, batch_size, ...) JAX array dict.

    Drops the trailing remainder so the scan length is known statically.
    """
    n = x.shape[0]
    perm = np.asarray(jax.random.permutation(key, n))
    x_sh = x[perm]
    y_sh = y[perm]
    n_batches = n // BATCH_SIZE
    x_b = jnp.asarray(x_sh[: n_batches * BATCH_SIZE].reshape(n_batches, BATCH_SIZE, -1))
    y_b = jnp.asarray(y_sh[: n_batches * BATCH_SIZE].reshape(n_batches, BATCH_SIZE))
    return {"x": x_b, "y": y_b}


def run_seed(seed: int, mnist: dict[str, np.ndarray], log_prefix: str = "") -> dict:
    init_G, init_z, init_W, init_op = make_inits(seed)
    test_batch_full = {
        "x": jnp.asarray(mnist["x_test"]),
        "y": jnp.asarray(mnist["y_test"]),
    }

    # FWS-hyper
    fws_combined = {"G": init_G, "z": init_z}
    fws_opt, fws_run = make_fws_segment(G_LR, Z_LR)
    fws_state = fws_opt.init(fws_combined)

    # W matched
    w_opt = optax.adam(1e-3)
    w_run = make_direct_segment(cross_entropy_loss, w_opt)
    w_state = w_opt.init(init_W)
    w_params = init_W

    # W overparam
    op_opt = optax.adam(1e-3)

    def op_loss_fn(op_params, batch):
        return cross_entropy_loss(op_params.materialise(), batch)

    op_run = make_direct_segment(op_loss_fn, op_opt)
    op_state = op_opt.init(init_op)
    op_params = init_op

    # End-of-epoch FWS diagnostics
    alphas_W1 = np.full(NUM_EPOCHS + 1, np.nan)
    alphas_W2 = np.full(NUM_EPOCHS + 1, np.nan)
    r2s_W1 = np.full(NUM_EPOCHS + 1, np.nan)
    r2s_W2 = np.full(NUM_EPOCHS + 1, np.nan)
    sigma_min_z = np.full(NUM_EPOCHS + 1, np.nan)
    sigma_max_z = np.full(NUM_EPOCHS + 1, np.nan)
    sigma_min_gh = np.full(NUM_EPOCHS + 1, np.nan)
    sigma_max_gh = np.full(NUM_EPOCHS + 1, np.nan)
    # Test-set metrics per arm, per checkpoint
    test_acc = {arm: np.full(NUM_EPOCHS + 1, np.nan) for arm in ("fws_hyper", "w_matched", "w_overparam")}
    test_xent = {arm: np.full(NUM_EPOCHS + 1, np.nan) for arm in ("fws_hyper", "w_matched", "w_overparam")}

    def measure_fws(ckpt_idx: int, G_now: HyperRenderer, z_now: Array, W_tree: dict[str, Array]) -> None:
        a1, r1 = ht_sr_alpha(W_tree["W1"])
        a2, r2 = ht_sr_alpha(W_tree["W2"])
        alphas_W1[ckpt_idx] = a1
        alphas_W2[ckpt_idx] = a2
        r2s_W1[ckpt_idx] = r1
        r2s_W2[ckpt_idx] = r2
        sigma_z = np.asarray(sigma_at_z(G_now, z_now))
        sigma_min_z[ckpt_idx] = float(sigma_z[-1])
        sigma_max_z[ckpt_idx] = float(sigma_z[0])
        sigma_g = np.asarray(sigma_at_gh(G_now, z_now))
        sigma_min_gh[ckpt_idx] = float(sigma_g[-1])
        sigma_max_gh[ckpt_idx] = float(sigma_g[0])

    def measure_test(ckpt_idx: int, fws_W: dict, w_W: dict, op_W: dict) -> None:
        test_acc["fws_hyper"][ckpt_idx] = accuracy(fws_W, test_batch_full)
        test_acc["w_matched"][ckpt_idx] = accuracy(w_W, test_batch_full)
        test_acc["w_overparam"][ckpt_idx] = accuracy(op_W, test_batch_full)
        test_xent["fws_hyper"][ckpt_idx] = float(cross_entropy_loss(fws_W, test_batch_full))
        test_xent["w_matched"][ckpt_idx] = float(cross_entropy_loss(w_W, test_batch_full))
        test_xent["w_overparam"][ckpt_idx] = float(cross_entropy_loss(op_W, test_batch_full))

    # checkpoint 0 (untrained)
    W_now = hyper_render_fn(fws_combined["G"], fws_combined["z"])
    measure_fws(0, fws_combined["G"], fws_combined["z"], W_now)
    measure_test(0, W_now, w_params, op_params.materialise())

    fws_loss_segments: list[np.ndarray] = []
    w_loss_segments: list[np.ndarray] = []
    op_loss_segments: list[np.ndarray] = []

    t_start = time.time()
    for epoch in range(NUM_EPOCHS):
        key_epoch = jax.random.fold_in(jax.random.key(seed + 1000), epoch)
        batches = make_batches(mnist["x_train"], mnist["y_train"], key=key_epoch)

        fws_combined, fws_state, fws_loss_seg, _ = fws_run(fws_combined, fws_state, batches)
        fws_loss_segments.append(np.asarray(fws_loss_seg))

        w_params, w_state, w_loss_seg, _ = w_run(w_params, w_state, batches)
        w_loss_segments.append(np.asarray(w_loss_seg))

        op_params, op_state, op_loss_seg, _ = op_run(op_params, op_state, batches)
        op_loss_segments.append(np.asarray(op_loss_seg))

        W_now = hyper_render_fn(fws_combined["G"], fws_combined["z"])
        measure_fws(epoch + 1, fws_combined["G"], fws_combined["z"], W_now)
        measure_test(epoch + 1, W_now, w_params, op_params.materialise())

        print(
            f"{log_prefix}  epoch {epoch + 1}/{NUM_EPOCHS}: "
            f"fws_test_acc={test_acc['fws_hyper'][epoch + 1]:.4f}  "
            f"w_test_acc={test_acc['w_matched'][epoch + 1]:.4f}  "
            f"op_test_acc={test_acc['w_overparam'][epoch + 1]:.4f}  "
            f"({time.time() - t_start:.1f}s)"
        )

    fws_loss_traj = np.concatenate(fws_loss_segments)
    w_loss_traj = np.concatenate(w_loss_segments)
    op_loss_traj = np.concatenate(op_loss_segments)

    # Final HT-SR α on each arm's terminal W1 / W2
    fws_W_final = hyper_render_fn(fws_combined["G"], fws_combined["z"])
    a1_fws, r1_fws = ht_sr_alpha(fws_W_final["W1"])
    a2_fws, r2_fws = ht_sr_alpha(fws_W_final["W2"])
    a1_w, r1_w = ht_sr_alpha(w_params["W1"])
    a2_w, r2_w = ht_sr_alpha(w_params["W2"])
    op_W_final = op_params.materialise()
    a1_op, r1_op = ht_sr_alpha(op_W_final["W1"])
    a2_op, r2_op = ht_sr_alpha(op_W_final["W2"])

    # Hessian top eigenvalue at convergence, on a single fixed test batch
    hess_batch = {"x": test_batch_full["x"][:512], "y": test_batch_full["y"][:512]}
    hess_fws = float(np.asarray(hessian_top_at(fws_W_final, hess_batch))[0])
    hess_w = float(np.asarray(hessian_top_at(w_params, hess_batch))[0])
    hess_op = float(np.asarray(hessian_top_at(op_W_final, hess_batch))[0])

    # Confusion matrices + sample predictions
    fws_preds = predictions(fws_W_final, test_batch_full)
    w_preds = predictions(w_params, test_batch_full)
    op_preds = predictions(op_W_final, test_batch_full)
    y_true_np = np.asarray(test_batch_full["y"])

    def _confusion(preds: np.ndarray) -> np.ndarray:
        cm = np.zeros((OUT_DIM, OUT_DIM), dtype=np.int32)
        for t, p in zip(y_true_np, preds):
            cm[int(t), int(p)] += 1
        return cm

    any_nan = bool(
        np.isnan(fws_loss_traj).any() or np.isnan(w_loss_traj).any() or np.isnan(op_loss_traj).any()
    )

    # Per-leaf G_leaf params (for curation panel)
    g_leaf_params_per_leaf = {
        name: {k: np.asarray(v) for k, v in fws_combined["G"].produce(fws_combined["z"], LEAF_ORDER.index(name)).items()}
        for name in LEAF_ORDER
    }

    return {
        "seed": seed,
        "fws_test_acc": float(test_acc["fws_hyper"][-1]),
        "w_test_acc": float(test_acc["w_matched"][-1]),
        "op_test_acc": float(test_acc["w_overparam"][-1]),
        "fws_test_xent": float(test_xent["fws_hyper"][-1]),
        "w_test_xent": float(test_xent["w_matched"][-1]),
        "op_test_xent": float(test_xent["w_overparam"][-1]),
        "fws_loss_traj": fws_loss_traj,
        "w_loss_traj": w_loss_traj,
        "op_loss_traj": op_loss_traj,
        "test_acc": test_acc,
        "test_xent": test_xent,
        "alphas_W1": alphas_W1,
        "alphas_W2": alphas_W2,
        "r2s_W1": r2s_W1,
        "r2s_W2": r2s_W2,
        "alpha_W1_fws": a1_fws, "r2_W1_fws": r1_fws,
        "alpha_W2_fws": a2_fws, "r2_W2_fws": r2_fws,
        "alpha_W1_w": a1_w, "r2_W1_w": r1_w,
        "alpha_W2_w": a2_w, "r2_W2_w": r2_w,
        "alpha_W1_op": a1_op, "r2_W1_op": r1_op,
        "alpha_W2_op": a2_op, "r2_W2_op": r2_op,
        "sigma_min_z": sigma_min_z,
        "sigma_max_z": sigma_max_z,
        "sigma_min_gh": sigma_min_gh,
        "sigma_max_gh": sigma_max_gh,
        "hess_fws": hess_fws, "hess_w": hess_w, "hess_op": hess_op,
        "any_nan": any_nan,
        "fws_W": {k: np.asarray(v) for k, v in fws_W_final.items()},
        "w_W": {k: np.asarray(v) for k, v in w_params.items()},
        "op_W": {k: np.asarray(v) for k, v in op_W_final.items()},
        "g_leaf_params": g_leaf_params_per_leaf,
        "y_true": y_true_np,
        "fws_preds": fws_preds, "w_preds": w_preds, "op_preds": op_preds,
        "x_test": np.asarray(mnist["x_test"]),
        "cm_fws": _confusion(fws_preds), "cm_w": _confusion(w_preds), "cm_op": _confusion(op_preds),
    }


# --- Plotting -------------------------------------------------------------
def _select_representative_seeds(per_seed: list[dict], key: str, *, higher_is_better: bool = True) -> dict[str, dict]:
    sorted_seeds = sorted(per_seed, key=lambda r: r[key], reverse=higher_is_better)
    return {
        "best": sorted_seeds[0],
        "median": sorted_seeds[len(sorted_seeds) // 2],
        "worst": sorted_seeds[-1],
    }


def plot_loss_trajectories(per_seed: list[dict], save_path: Path):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for r, arm_key, colour, label in [
        (per_seed, "fws_loss_traj", FWS_COLOUR, "FWS-hyper"),
        (per_seed, "w_loss_traj", W_MATCHED_COLOUR, "W matched"),
        (per_seed, "op_loss_traj", W_OVERPARAM_COLOUR, "W overparam"),
    ]:
        stack = np.stack([s[arm_key] for s in r], axis=0)
        steps = np.arange(stack.shape[1])
        for s in r:
            ax.plot(steps, s[arm_key], color=colour, alpha=0.25, linewidth=0.8)
        ax.plot(steps, np.median(stack, axis=0), color=colour, linewidth=1.8, label=label)
    ax.set(yscale="log", xlabel="outer step", ylabel="train cross-entropy",
           title=f"Phase 8: MNIST training-loss trajectories (K={K_SEED})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_test_acc_trajectory(per_seed: list[dict], save_path: Path):
    epochs = np.arange(NUM_EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for arm_key, colour, label in [
        ("fws_hyper", FWS_COLOUR, "FWS-hyper"),
        ("w_matched", W_MATCHED_COLOUR, "W matched"),
        ("w_overparam", W_OVERPARAM_COLOUR, "W overparam"),
    ]:
        stack = np.stack([r["test_acc"][arm_key] for r in per_seed], axis=0)
        for r in per_seed:
            ax.plot(epochs, r["test_acc"][arm_key], color=colour, alpha=0.30, linewidth=1.0)
        ax.plot(epochs, np.median(stack, axis=0), color=colour, linewidth=2.0, label=label)
    ax.set(xlabel="epoch", ylabel="test accuracy",
           title=f"Phase 8: MNIST test accuracy by epoch (K={K_SEED})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_alpha_trajectory(per_seed: list[dict], save_path: Path):
    epochs = np.arange(NUM_EPOCHS + 1)
    alphas_stack = np.stack([r["alphas_W1"] for r in per_seed], axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for r in per_seed:
        ax.plot(epochs, r["alphas_W1"], color=FWS_COLOUR, alpha=0.30, linewidth=1.0)
    ax.plot(epochs, np.nanmedian(alphas_stack, axis=0), color=FWS_COLOUR, linewidth=2.2, label="median across seeds")
    ax.axhspan(1.4, 1.7, color=WONG_PALETTE[3], alpha=0.18, label="trained-net α band [1.4, 1.7]")
    ax.set(xlabel="epoch", ylabel="HT-SR α (W1, rendered)",
           title=f"Phase 8: FWS-hyper α trajectory on W1 (K={K_SEED})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_final_acc_box(per_seed: list[dict], save_path: Path):
    fws = np.array([r["fws_test_acc"] for r in per_seed])
    w = np.array([r["w_test_acc"] for r in per_seed])
    op = np.array([r["op_test_acc"] for r in per_seed])
    data = [fws, w, op]
    colours = [FWS_COLOUR, W_MATCHED_COLOUR, W_OVERPARAM_COLOUR]
    labels = ["FWS-hyper", "W matched", "W overparam"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, c in zip(bp["boxes"], colours, strict=True):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(c)
    for median_line, c in zip(bp["medians"], colours, strict=True):
        median_line.set_color(c)
        median_line.set_linewidth(2.0)
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(data, colours, strict=True), start=1):
        jitter = rng.uniform(-0.10, 0.10, size=vals.size)
        ax.scatter(np.full_like(vals, i) + jitter, vals, color=c, s=40, zorder=3, edgecolor="white", linewidths=0.6)
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("final test accuracy")
    ax.set_title(f"Phase 8: final test accuracy distribution (K={K_SEED})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _heatmap(ax, M: np.ndarray, *, title: str, cmap: str = "RdBu_r"):
    vmax = float(np.percentile(np.abs(M), 95)) or float(np.abs(M).max()) or 1.0
    ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _confusion_heatmap(ax, cm: np.ndarray, *, title: str):
    ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("predicted", fontsize=7)
    ax.set_ylabel("true", fontsize=7)
    ax.set_xticks(range(OUT_DIM))
    ax.set_yticks(range(OUT_DIM))
    ax.tick_params(axis="both", labelsize=6)


def _sample_classifications(ax, x_test: np.ndarray, y_true: np.ndarray, preds: np.ndarray,
                            *, title: str, n: int = 12, rng_seed: int = 0):
    """Render n random test images in a horizontal strip with true/pred annotations."""
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(x_test.shape[0], size=n, replace=False)
    # build a (28, 28*n) strip
    strip = np.concatenate([x_test[i].reshape(28, 28) for i in idx], axis=1)
    ax.imshow(strip, cmap="gray_r", aspect="auto")
    for j, i in enumerate(idx):
        colour = "green" if int(y_true[i]) == int(preds[i]) else "red"
        ax.text(28 * j + 14, -2, f"{y_true[i]}→{preds[i]}",
                ha="center", va="bottom", fontsize=6, color=colour)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def _w1_filter_grid(ax, W1: np.ndarray, *, title: str, n_show: int = 16):
    """Show the first n_show rows of W1 as 28×28 input filters in a 4×4 grid."""
    rows = W1[:n_show]
    side = int(np.sqrt(n_show))
    cell = 28
    grid = np.zeros((side * cell, side * cell), dtype=np.float32)
    vmax = float(np.percentile(np.abs(rows), 95)) or float(np.abs(rows).max()) or 1.0
    for k in range(n_show):
        r, c = divmod(k, side)
        img = rows[k].reshape(28, 28)
        grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = img
    ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_arm_curation(per_seed: list[dict], arm: str, save_path: Path):
    arm_acc_key = {"fws_hyper": "fws_test_acc", "w_matched": "w_test_acc", "w_overparam": "op_test_acc"}[arm]
    arm_W_key = {"fws_hyper": "fws_W", "w_matched": "w_W", "w_overparam": "op_W"}[arm]
    arm_preds_key = {"fws_hyper": "fws_preds", "w_matched": "w_preds", "w_overparam": "op_preds"}[arm]
    arm_cm_key = {"fws_hyper": "cm_fws", "w_matched": "cm_w", "w_overparam": "cm_op"}[arm]
    arm_label = {"fws_hyper": "FWS-hyper", "w_matched": "W matched", "w_overparam": "W overparam"}[arm]

    # Higher accuracy is better.
    rep = _select_representative_seeds(per_seed, arm_acc_key, higher_is_better=True)
    row_labels = ["best", "median", "worst"]

    fig, axes = plt.subplots(3, 5, figsize=(15.0, 8.0))
    for row_idx, row_label in enumerate(row_labels):
        r = rep[row_label]
        W = r[arm_W_key]
        prefix = f"{row_label} (seed {r['seed']}, acc {r[arm_acc_key]:.3f})"
        _w1_filter_grid(axes[row_idx, 0], W["W1"], title=f"{prefix}\nW1 first-16 filters")
        _heatmap(axes[row_idx, 1], W["W2"], title="W2 (10×128)")
        _confusion_heatmap(axes[row_idx, 2], r[arm_cm_key], title="confusion matrix")
        _sample_classifications(
            axes[row_idx, 3], r["x_test"], r["y_true"], r[arm_preds_key],
            title="12 random test classifications", rng_seed=r["seed"],
        )
        ax_traj = axes[row_idx, 4]
        loss_key = {"fws_hyper": "fws_loss_traj", "w_matched": "w_loss_traj", "w_overparam": "op_loss_traj"}[arm]
        ax_traj.plot(np.arange(r[loss_key].size), r[loss_key], color=WONG_PALETTE[5], linewidth=0.8)
        ax_traj.set_yscale("log")
        ax_traj.set_title("train-XE trajectory", fontsize=8)
        ax_traj.tick_params(axis="both", labelsize=6)
        ax_traj.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Phase 8: {arm_label} arm — best / median / worst (by test accuracy, K={K_SEED})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_fws_g_leaf_curation(per_seed: list[dict], save_path: Path):
    rep = _select_representative_seeds(per_seed, "fws_test_acc", higher_is_better=True)
    chosen = rep["median"]
    g_leaf = chosen["g_leaf_params"]
    fig, axes = plt.subplots(N_LEAVES, 3, figsize=(10.0, 8.5))
    for row_idx, leaf_name in enumerate(LEAF_ORDER):
        p = g_leaf[leaf_name]
        _heatmap(axes[row_idx, 0], p["W1"], title=f"{leaf_name} — G_leaf W1 (32,1)")
        _heatmap(axes[row_idx, 1], p["W2"], title=f"{leaf_name} — G_leaf W2 (32,32)")
        _heatmap(axes[row_idx, 2], p["w_out"].reshape(1, -1), title=f"{leaf_name} — G_leaf w_out (1,32)")
    fig.suptitle(
        f"Phase 8: per-leaf G_leaf parameters (median-test-acc FWS-hyper seed = {chosen['seed']})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# --- Main ------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Phase 8 — MNIST integration smoke")
    print("=" * 72)

    init_G_demo, init_z_demo, init_W_demo, init_op_demo = make_inits(0)
    fws_param_count = count_params({"G": init_G_demo, "z": init_z_demo})
    w_param_count = count_params(init_W_demo)
    op_param_count = count_params(init_op_demo)

    print(f"Mainnet: in={IN_DIM}, hidden={HIDDEN_DIM_MAIN}, out={OUT_DIM} → {TOTAL_W_PARAMS} W params")
    print(f"FWS-hyper G_LEAF_PARAM_SIZE={G_LEAF_PARAM_SIZE}, G_H_HIDDEN_DIM={G_H_HIDDEN_DIM}")
    print(f"FWS-hyper trainable params: {fws_param_count}")
    print(f"W matched trainable params: {w_param_count}")
    print(f"W overparam trainable params: {op_param_count}  (P={OVERPARAM_P})")
    print(f"K_seed = {K_SEED}, epochs = {NUM_EPOCHS}, batch_size = {BATCH_SIZE}, steps/epoch = {STEPS_PER_EPOCH}")

    print("\nLoading MNIST (cache: %s)..." % MNIST_CACHE)
    mnist = load_mnist()
    print(f"  x_train {mnist['x_train'].shape}  y_train {mnist['y_train'].shape}")
    print(f"  x_test  {mnist['x_test'].shape}  y_test  {mnist['y_test'].shape}")

    per_seed: list[dict] = []
    nan_or_crash: list[str] = []
    seed_wall_times: list[float] = []
    t0 = time.time()
    for seed in range(K_SEED):
        t_seed = time.time()
        try:
            row = run_seed(seed, mnist, log_prefix=f"[seed {seed}]")
            per_seed.append(row)
            if row["any_nan"]:
                nan_or_crash.append(f"seed={seed}: NaN in a loss trajectory")
            wall = time.time() - t_seed
            seed_wall_times.append(wall)
            print(
                f"  seed {seed}: fws_acc={row['fws_test_acc']:.4f}  w_acc={row['w_test_acc']:.4f}  "
                f"op_acc={row['op_test_acc']:.4f}  alpha_W1_fws={row['alpha_W1_fws']:.4g}  "
                f"any_nan={row['any_nan']}  ({wall:.1f}s)"
            )
            if seed == 0:
                print(f"  --- first seed complete, total elapsed {time.time() - t0:.1f}s (pipes verified) ---")
        except Exception as e:  # noqa: BLE001
            nan_or_crash.append(f"seed={seed}: CRASH — {type(e).__name__}: {e}")
            print(f"  seed {seed}: CRASH — {type(e).__name__}: {e}")

    total_wall = time.time() - t0
    print(f"\nAll seeds complete: {total_wall:.1f}s total wall, median {np.median(seed_wall_times):.1f}s/seed.")

    fws_accs = np.array([r["fws_test_acc"] for r in per_seed])
    w_accs = np.array([r["w_test_acc"] for r in per_seed])
    op_accs = np.array([r["op_test_acc"] for r in per_seed])
    fws_xents = np.array([r["fws_test_xent"] for r in per_seed])
    w_xents = np.array([r["w_test_xent"] for r in per_seed])
    op_xents = np.array([r["op_test_xent"] for r in per_seed])

    print()
    print("Final test accuracy (median ± range across K seeds):")
    print(f"  FWS-hyper:    median={np.median(fws_accs):.4f}  min={fws_accs.min():.4f}  max={fws_accs.max():.4f}")
    print(f"  W matched:    median={np.median(w_accs):.4f}  min={w_accs.min():.4f}  max={w_accs.max():.4f}")
    print(f"  W overparam:  median={np.median(op_accs):.4f}  min={op_accs.min():.4f}  max={op_accs.max():.4f}")

    a_fws = np.array([r["alpha_W1_fws"] for r in per_seed])
    a_w = np.array([r["alpha_W1_w"] for r in per_seed])
    a_op = np.array([r["alpha_W1_op"] for r in per_seed])
    r2_fws = np.array([r["r2_W1_fws"] for r in per_seed])
    r2_w = np.array([r["r2_W1_w"] for r in per_seed])
    r2_op = np.array([r["r2_W1_op"] for r in per_seed])

    print()
    print("Final HT-SR α on W1 (median ± range across K seeds):")
    print(f"  FWS-hyper:    α median={np.median(a_fws):.4g}  R² median={np.median(r2_fws):.4g}")
    print(f"  W matched:    α median={np.median(a_w):.4g}  R² median={np.median(r2_w):.4g}")
    print(f"  W overparam:  α median={np.median(a_op):.4g}  R² median={np.median(r2_op):.4g}")

    # --- Plots ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_loss = FIGURES_DIR / "2026-06-07-phase8-loss-trajectories.png"
    fig_acc = FIGURES_DIR / "2026-06-07-phase8-test-acc-trajectory.png"
    fig_alpha = FIGURES_DIR / "2026-06-07-phase8-alpha-trajectory.png"
    fig_box = FIGURES_DIR / "2026-06-07-phase8-final-acc-box.png"
    plot_loss_trajectories(per_seed, fig_loss)
    plot_test_acc_trajectory(per_seed, fig_acc)
    plot_alpha_trajectory(per_seed, fig_alpha)
    plot_final_acc_box(per_seed, fig_box)
    plot_arm_curation(per_seed, "fws_hyper", FIGURES_DIR / "2026-06-07-phase8-curations-fws_hyper.png")
    plot_arm_curation(per_seed, "w_matched", FIGURES_DIR / "2026-06-07-phase8-curations-w_matched.png")
    plot_arm_curation(per_seed, "w_overparam", FIGURES_DIR / "2026-06-07-phase8-curations-w_overparam.png")
    plot_fws_g_leaf_curation(per_seed, FIGURES_DIR / "2026-06-07-phase8-curations-fws-hyper-leaves.png")
    print(f"\nWrote plots to {FIGURES_DIR}")

    # --- Research log ---
    rep_fws = _select_representative_seeds(per_seed, "fws_test_acc", higher_is_better=True)
    rep_w = _select_representative_seeds(per_seed, "w_test_acc", higher_is_better=True)
    rep_op = _select_representative_seeds(per_seed, "op_test_acc", higher_is_better=True)

    per_seed_rows = "\n".join(
        f"| {r['seed']} | {r['fws_test_acc']:.4f} | {r['w_test_acc']:.4f} | {r['op_test_acc']:.4f} | "
        f"{r['fws_test_xent']:.4g} | {r['w_test_xent']:.4g} | {r['op_test_xent']:.4g} | "
        f"{r['alpha_W1_fws']:.4g} | {r['alpha_W1_w']:.4g} | {r['alpha_W1_op']:.4g} | "
        f"{r['hess_fws']:.4g} | {r['hess_w']:.4g} | {r['hess_op']:.4g} | "
        f"{str(r['any_nan'])} |"
        for r in per_seed
    )
    nan_summary = "\n".join(f"- {line}" for line in nan_or_crash) if nan_or_crash else "- none."

    acc_table = (
        "| arm | params | median acc | min acc | max acc | median test XE |\n"
        "|---|---|---|---|---|---|\n"
        f"| FWS-hyper | {fws_param_count} | {np.median(fws_accs):.4f} | {fws_accs.min():.4f} | {fws_accs.max():.4f} | {np.median(fws_xents):.4g} |\n"
        f"| W matched | {w_param_count} | {np.median(w_accs):.4f} | {w_accs.min():.4f} | {w_accs.max():.4f} | {np.median(w_xents):.4g} |\n"
        f"| W overparam | {op_param_count} | {np.median(op_accs):.4f} | {op_accs.min():.4f} | {op_accs.max():.4f} | {np.median(op_xents):.4g} |"
    )
    alpha_table = (
        "| arm | median α(W1) | median R²(W1) | median α(W2) | median R²(W2) |\n"
        "|---|---|---|---|---|\n"
        f"| FWS-hyper | {np.median(a_fws):.4g} | {np.median(r2_fws):.4g} | {np.median([r['alpha_W2_fws'] for r in per_seed]):.4g} | {np.median([r['r2_W2_fws'] for r in per_seed]):.4g} |\n"
        f"| W matched | {np.median(a_w):.4g} | {np.median(r2_w):.4g} | {np.median([r['alpha_W2_w'] for r in per_seed]):.4g} | {np.median([r['r2_W2_w'] for r in per_seed]):.4g} |\n"
        f"| W overparam | {np.median(a_op):.4g} | {np.median(r2_op):.4g} | {np.median([r['alpha_W2_op'] for r in per_seed]):.4g} | {np.median([r['r2_W2_op'] for r in per_seed]):.4g} |"
    )

    md = f"""# Phase 8 — MNIST integration smoke — 2026-06-07

## Background

Phases 4–7 ran the per-leaf hyper-renderer programme on a synthetic
2-layer SiLU teacher-student regression task. The user direction for
phase 8 was to **graduate to real data** — first-time MNIST integration
smoke for the FWS-hyper architecture from phase 7, with cross-entropy
loss, mini-batch SGD, and a 100K-param student classifier.

This is an *integration smoke*: $K = {K_SEED}$ at 5 epochs each, the
minimum needed to verify the pipes connect. K=10 statistical confirmation
is a later phase if and only if the pipes are clean here.

## Setup

### Data

MNIST 28×28 grayscale digits, fetched once from
`{MNIST_MIRROR}` (PyTorch's `ossci-datasets` mirror) and cached locally
at `{MNIST_CACHE}`. Pixels normalised to $[0, 1]$, flattened to a 784-D
vector. Train set is the first {N_TRAIN} of MNIST's 60K-image train split
(per brief spec); test set is the canonical {N_TEST}-image test split.
Loader is stdlib `urllib` + `gzip` + `struct` — no `torchvision`,
no `tensorflow_datasets` (neither were installed in the project venv;
the stdlib loader keeps the example self-contained).

### Mainnet (the student classifier)

2-layer SiLU MLP: `x (784,) → W1 (128, 784) → silu → W2 (10, 128) → logits`.
Total trainable scalars: ${TOTAL_W_PARAMS}$
(``W_1`` = {LEAF_SIZES['W1']}, ``b_1`` = {LEAF_SIZES['b1']},
``W_2`` = {LEAF_SIZES['W2']}, ``b_2`` = {LEAF_SIZES['b2']}).
Both W arms reuse this exact shape; the FWS-hyper arm renders into it
via ``hyper_render_fn``.

### Three arms

- **FWS-hyper** ({fws_param_count} trainable params): per-leaf
  hyper-renderer from phase 7, sized so the trainable count *matches*
  the mainnet rather than ballooning to the 8× ratio used in the
  synthetic regime. ``G_H`` is a 2-layer SiLU MLP with hidden dim
  ``{G_H_HIDDEN_DIM}``, output ``G_LEAF_PARAM_SIZE = {G_LEAF_PARAM_SIZE}``.
  ``G_leaf`` is a depth-2 SIREN over a 1-D within-leaf coord with
  hidden dim ``{G_LEAF_HIDDEN_DIM}`` and explicit
  $\\omega_0 = {OMEGA_FIRST}$. ``z ∈ R^{DIM_Z}``, flows into ``G_H``.
- **W matched** ({w_param_count}): direct training of the mainnet
  with Adam$(10^{{-3}})$.
- **W overparam** ({op_param_count}): factored MLP
  ``W_k = A_{{2,k}} A_{{1,k}}`` with rank $P = {OVERPARAM_P}$;
  biases 1-D. Total params ≈ 5× matched (the synthetic-regime
  W-overparam ratio was ~150×; at MNIST scale a 5× ratio is the
  natural architectural-overhead choice — see caveats).

### Training schedule

- Optimiser: Adam$(10^{{-3}})$ on every parameter group
  (``G_H``-params, ``z``, ``W_*``, ``A_*``).
- Mini-batch SGD with batch size {BATCH_SIZE}, {NUM_EPOCHS} epochs
  = {STEPS_PER_EPOCH} steps/epoch × {NUM_EPOCHS} = {NUM_OUTER_STEPS}
  outer steps. Per-epoch reshuffle (`jax.random.permutation`); trailing
  batch dropped so the per-epoch scan length is static.
- Each epoch is a single `jax.lax.scan` over the pre-shuffled batches.
- $K_{{\\text{{seed}}}} = {K_SEED}$.

### Per-seed measurements

- Train cross-entropy + grad-norm at every outer step (logged in the
  trajectory plot).
- Test accuracy + cross-entropy on the full {N_TEST}-image test set at
  end of each epoch (5 checkpoints + the untrained baseline).
- For FWS-hyper at each end-of-epoch checkpoint: HT-SR $\\alpha$ + R²
  on the rendered ``W_1`` and ``W_2`` separately; $\\sigma$-spectrum
  (top-{SIGMA_TOP_K} via power iteration with {SIGMA_POWER_ITERS}
  iterations) of $\\partial \\text{{render}}/\\partial z$ and
  $\\partial \\text{{render}}/\\partial G_H\\_params$.
- At end of training, for all three arms: final test accuracy,
  HT-SR $\\alpha$ + R² on $W_1$ and $W_2$, Hessian top eigenvalue
  via `landscape_archaeology.singular_spectrum` applied to
  `jax.grad(cross_entropy_loss)` on a single fixed 512-image
  test slice (full-test Hessian power iteration is prohibitive
  for a smoke).

## Results

### Final test accuracy

{acc_table}

### Final HT-SR α on rendered / trained W1 + W2 (median across K seeds)

{alpha_table}

### Per-seed table

| seed | FWS-h acc | W-m acc | W-op acc | FWS-h XE | W-m XE | W-op XE | α(W1) FWS | α(W1) W-m | α(W1) W-op | hess FWS | hess W-m | hess W-op | any NaN |
|------|-----------|---------|----------|----------|--------|---------|-----------|-----------|------------|----------|----------|-----------|---------|
{per_seed_rows}

### Plots

![training loss trajectories (all arms)](figures/2026-06-07-phase8-loss-trajectories.png)

![test accuracy by epoch](figures/2026-06-07-phase8-test-acc-trajectory.png)

![FWS-hyper α(W1) trajectory](figures/2026-06-07-phase8-alpha-trajectory.png)

![final test accuracy distribution](figures/2026-06-07-phase8-final-acc-box.png)

## Curated outputs (representative seeds)

Per-arm best / median / worst seeds picked by final test accuracy rank
(pre-registered rule, not eyeballed). Each row shows the first 16
rows of $W_1$ reshaped as $28 \\times 28$ input filters, the full $W_2$
heatmap, the confusion matrix, 12 random test classifications, and the
per-step training-XE trajectory.

| arm | best seed | best acc | median seed | median acc | worst seed | worst acc |
|---|---|---|---|---|---|---|
| FWS-hyper | {rep_fws['best']['seed']} | {rep_fws['best']['fws_test_acc']:.4f} | {rep_fws['median']['seed']} | {rep_fws['median']['fws_test_acc']:.4f} | {rep_fws['worst']['seed']} | {rep_fws['worst']['fws_test_acc']:.4f} |
| W matched | {rep_w['best']['seed']} | {rep_w['best']['w_test_acc']:.4f} | {rep_w['median']['seed']} | {rep_w['median']['w_test_acc']:.4f} | {rep_w['worst']['seed']} | {rep_w['worst']['w_test_acc']:.4f} |
| W overparam | {rep_op['best']['seed']} | {rep_op['best']['op_test_acc']:.4f} | {rep_op['median']['seed']} | {rep_op['median']['op_test_acc']:.4f} | {rep_op['worst']['seed']} | {rep_op['worst']['op_test_acc']:.4f} |

![FWS-hyper arm curation](figures/2026-06-07-phase8-curations-fws_hyper.png)

![W matched arm curation](figures/2026-06-07-phase8-curations-w_matched.png)

![W overparam arm curation](figures/2026-06-07-phase8-curations-w_overparam.png)

### Per-leaf G_leaf eyeball check (FWS-hyper, median seed)

Visualises the four ``G_leaf`` parameter sets ``G_H`` emits for the four
leaves on the median-test-accuracy FWS-hyper seed. Asks: are the four
emitted parameter sets structurally distinct, or do they collapse?

![Per-leaf G_leaf parameters](figures/2026-06-07-phase8-curations-fws-hyper-leaves.png)

## NaN / crash report

{nan_summary}

## Honest caveats

- **Integration smoke, not statistical confirmation.** $K = {K_SEED}$
  with 5 training epochs — enough to verify pipes (dataloader, model,
  cross-entropy, hyper-rendered grad, optimiser, measurement, plotting),
  not enough to license claims about which arm wins. No paired Wilcoxon
  / bootstrap-BCa is reported here; the next phase upgrades to that
  regime if the pipes are clean.
- **5 epochs is light training.** Real MNIST classifiers commonly train
  20–100+ epochs and exceed 99% test accuracy; the comparison signal at
  5 epochs is on a different part of the learning curve. This is a
  conscious smoke choice, not a claim about asymptotic behaviour.
- **W-overparam ratio = 5× matched.** Phase 6 used $P = 395$ giving a
  ~150× ratio at the synthetic 58-param scale, where matched and
  over-parameterised arms differ by orders of magnitude. At MNIST scale
  the mainnet is already ~100K params, and a comparable ratio would
  approach 15M trainable params (architecturally indistinguishable from
  a width-1024 MLP and far outside the brief's spirit). The 5× ratio
  reflects the natural architectural-overhead vs FWS-overhead trade-off
  at this scale.
- **MNIST is a small dataset where classifiers commonly hit 99%+.**
  Even modest architectures saturate at very high test accuracy, which
  compresses the FWS-vs-W signal. The contrast is therefore on
  *learning dynamics* (loss curves, $\\alpha$ trajectories) more than
  on terminal accuracy.
- **Hessian probe uses a 512-image slice**, not the full test set,
  because $W$-overparam has ~500K params and full-test top-eigenvalue
  power iteration on a 100K+ param model is dominant in wall time.
  The slice is fixed across arms within a seed so the comparison is
  apples-to-apples within seed.
- **MNIST loader is stdlib-only** (`urllib` + `gzip` + `struct`).
  Switching to `tensorflow_datasets` or `torchvision` is an optional
  follow-up; nothing about the science depends on it.
- $\\alpha$ on $W_2$ is fit on a 10-eigenvalue spectrum (since $W_2$
  is $10 \\times 128$), which is the lower bound of HT-SR fit
  reliability — read $R^2$ alongside the $\\alpha$ value.

## What's next

If the pipes connect cleanly here, phase 9 (CIFAR-10 with a conv
mainnet) is the next dataset step, and a $K = 10$ MNIST follow-up with
20+ epochs and a paired-Wilcoxon + bootstrap-BCa contract becomes
warranted. If the FWS-hyper arm fails (NaN, collapse, or accuracy stuck
at chance), the priority instead becomes diagnosing whether the
100K-mainnet-scale ``G_H`` budget is too small or whether the 1-D
within-leaf coord scheme breaks down on the 784-D ``W_1`` row dim.
"""

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_FILE.write_text(md)
    print(f"Wrote research log to {RESEARCH_FILE}")


if __name__ == "__main__":
    main()
