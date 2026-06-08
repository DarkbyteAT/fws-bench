"""WideKernelCNN-SiLU mainnet topology, leaf layout, forward + loss.

Same mainnet across phases 8 / 9 / 10 of the FWS programme: a 2-conv
2-fc CIFAR-10 classifier with kernel size :math:`k=5` (large enough to
engage the radial-FFT α diagnostic) and ~30 K trainable parameters
(small enough that K=3 × 4 arms × 5 epochs is CPU-tractable).

The leaf order, shapes, and ranks are the single source of truth for
every arm: ``FWS-hyper`` slices ``G_H``'s flat output per leaf, the
direct ``W`` arms initialise these tensors directly, and the
diagnostics in :mod:`._common.diagnostics` index by leaf name.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


# --- Mainnet topology --------------------------------------------------------
IN_CHANNELS = 3
NUM_CLASSES = 10
CONV1_OUT = 8
CONV2_OUT = 16
KERNEL_SIZE = 5
FC1_HIDDEN = 64

# 32 -> conv k=5 -> 28 -> pool2 -> 14 -> conv k=5 -> 10 -> pool2 -> 5 -> flat 400
FC1_IN_DIM = 5 * 5 * CONV2_OUT

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
TOTAL_W_PARAMS: int = sum(LEAF_SIZES.values())

LEAF_RANKS: dict[str, int] = {n: len(LEAF_SHAPES[n]) for n in LEAF_ORDER}
DISTINCT_RANKS: tuple[int, ...] = (1, 2, 4)


# --- Forward pass + loss -----------------------------------------------------
def _conv2d(x: Array, w: Array, b: Array) -> Array:
    out = jax.lax.conv_general_dilated(
        x[None], w, window_strides=(1, 1), padding="VALID",
    )[0]
    return out + b[:, None, None]


def _maxpool2(x: Array) -> Array:
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2), (1, 2, 2), "VALID")


def cnn_forward(params: dict[str, Array], x: Float[Array, "3 32 32"]) -> Float[Array, " 10"]:
    """WideKernelCNN-SiLU forward pass on a single image."""
    h = jax.nn.silu(_conv2d(x, params["conv1_w"], params["conv1_b"]))
    h = _maxpool2(h)
    h = jax.nn.silu(_conv2d(h, params["conv2_w"], params["conv2_b"]))
    h = _maxpool2(h)
    h = h.reshape(-1)
    h = jax.nn.silu(params["fc1_w"] @ h + params["fc1_b"])
    return params["fc2_w"] @ h + params["fc2_b"]


def cross_entropy_loss(params: dict[str, Array], batch: dict) -> Float[Array, ""]:
    """Mean softmax cross-entropy over a CIFAR-10 batch."""
    logits = jax.vmap(lambda x: cnn_forward(params, x))(batch["x"])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(batch["y"], NUM_CLASSES)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


# --- W matched: direct CNN init ---------------------------------------------
def init_cnn_params(key: Array) -> dict[str, Array]:
    """Xavier-uniform init for the WideKernelCNN-SiLU mainnet."""
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


# --- W overparam: factored mainnet (~4× matched) ----------------------------
OVERPARAM_CONV_P = 32       # bottleneck channels for factored conv
OVERPARAM_FC1_P = 200
OVERPARAM_FC2_P = 64


class OverparamCNN(eqx.Module):
    """Factored (rank-P depthwise + pointwise) conv + factored fc CNN.

    Materialises to a tensor tree with the same leaf shapes as the matched
    arm via :meth:`materialise`; the materialised tree is then consumed by
    :func:`cnn_forward` and :func:`cross_entropy_loss`.
    """

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
        """Build the LEAF_SHAPES-keyed W tree from the factored params."""
        conv1_w = jnp.einsum("opHW,pi->oiHW", self.c1_depth, self.c1_point)
        conv2_w = jnp.einsum("opHW,pi->oiHW", self.c2_depth, self.c2_point)
        return {
            "conv1_w": conv1_w, "conv1_b": self.c1_bias,
            "conv2_w": conv2_w, "conv2_b": self.c2_bias,
            "fc1_w": self.fc1_A2 @ self.fc1_A1, "fc1_b": self.fc1_bias,
            "fc2_w": self.fc2_A2 @ self.fc2_A1, "fc2_b": self.fc2_bias,
        }
