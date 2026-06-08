"""CIFAR-mainnet-specific diagnostic helpers on rendered W trees.

The generic spectrum-fit functions ``ht_sr_alpha`` and ``radial_fft_alpha``
live in :mod:`landscape_archaeology` (the diagnostics library); the
Jacobian / Hessian σ-spectrum primitive is
:func:`landscape_archaeology.singular_spectrum`. This module is only the
thin CIFAR-mainnet-specific glue that knows which leaves are conv
weights and which are fc weights, plus pytree-level utilities used by
the orchestration loop.

What lives here:

- :func:`per_leaf_alphas` — apply :func:`landscape_archaeology.ht_sr_alpha`
  to fc1 / fc2 and :func:`landscape_archaeology.radial_fft_alpha` to
  conv1 / conv2.
- :func:`g_leaf_cosine_matrix` — pairwise cosine between the FWS-hyper
  ``G_H``'s per-leaf ``G_leaf`` parameter vectors. Used for the leaf-
  identity plot in the curated outputs.
- :func:`count_params`, :func:`global_l2_norm` — pytree convenience
  helpers used by the example-side reporting code.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree
from landscape_archaeology import ht_sr_alpha, radial_fft_alpha

from . import mainnet


def per_leaf_alphas(W_tree: dict[str, Array]) -> dict[str, tuple[float, float]]:
    """Compute α per fc-weight leaf (HT-SR) and per conv-weight leaf (radial-FFT).

    Returns a dict keyed by ``"conv1"``, ``"conv2"``, ``"fc1"``, ``"fc2"`` —
    leaf-class names dropping the ``_w`` suffix. Maps each entry to the
    ``(alpha, r2)`` pair returned by the respective ``landscape_archaeology``
    diagnostic.
    """
    return {
        "conv1": radial_fft_alpha(W_tree["conv1_w"]),
        "conv2": radial_fft_alpha(W_tree["conv2_w"]),
        "fc1":   ht_sr_alpha(W_tree["fc1_w"]),
        "fc2":   ht_sr_alpha(W_tree["fc2_w"]),
    }


def g_leaf_cosine_matrix(
    produce_flat: Callable[[Array, int], Array],
    z: Array,
    *,
    leaf_order: tuple[str, ...] = mainnet.LEAF_ORDER,
    leaf_ranks: dict[str, int] | None = None,
    g_leaf_param_size: dict[int, int] | None = None,
    max_g_leaf_param_size: int | None = None,
) -> np.ndarray:
    """Pairwise cosine between flat ``G_leaf`` parameter vectors.

    For each leaf, ``produce_flat(z, leaf_id)`` returns G_H's full output
    vector; we take the prefix corresponding to that leaf's rank-specific
    parameter size and zero-pad to the longest length, then compute the
    pairwise cosine. Used to inspect whether the hyper-renderer produces
    leaf-distinct or leaf-collapsed parameters at convergence.
    """
    from . import arms as _arms  # local import to avoid arms→diagnostics cycle

    if leaf_ranks is None:
        leaf_ranks = mainnet.LEAF_RANKS
    if g_leaf_param_size is None:
        g_leaf_param_size = _arms.G_LEAF_PARAM_SIZE
    if max_g_leaf_param_size is None:
        max_g_leaf_param_size = _arms.MAX_G_LEAF_PARAM_SIZE

    flats: list[np.ndarray] = []
    for name in leaf_order:
        leaf_id = leaf_order.index(name)
        rank = leaf_ranks[name]
        flat = np.asarray(produce_flat(z, leaf_id))[:g_leaf_param_size[rank]]
        padded = np.zeros(max_g_leaf_param_size, dtype=flat.dtype)
        padded[:flat.shape[0]] = flat
        flats.append(padded)
    M = np.stack(flats, axis=0)
    n = M @ M.T
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return n / (norms @ norms.T + 1e-30)


def count_params(tree: PyTree) -> int:
    """Total scalar count of all arrays in the pytree."""
    flat, _ = ravel_pytree(tree)
    return int(flat.size)


def global_l2_norm(grad_tree: PyTree) -> Array:
    """Global L2 norm across a pytree of gradients."""
    flat, _ = ravel_pytree(grad_tree)
    return jnp.linalg.norm(flat)
