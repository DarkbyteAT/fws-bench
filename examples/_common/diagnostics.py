"""Spectral and statistical diagnostics on rendered W trees.

Three families:

- **Per-leaf α**: :func:`ht_sr_alpha` for fully-connected weight matrices
  (heavy-tailed self-regularisation slope), :func:`radial_fft_alpha` for
  conv kernels (radial-FFT power-law slope). Both return ``(alpha, r2)``.
- **Jacobian σ-spectrum**: :func:`sigma_spectrum_op` wraps
  ``landscape_archaeology.singular_spectrum`` around a renderer-as-operator.
- **Hessian top-eigs**: :func:`hessian_top` returns the top-k Hessian
  eigenvalues of the loss at the materialised W tree.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree
from landscape_archaeology import singular_spectrum

from . import mainnet


# --- HT-SR α on fc weight matrices -----------------------------------------
def ht_sr_alpha(W: Array) -> tuple[float, float]:
    """Heavy-tailed self-regularisation slope of ``W``'s eigenvalue spectrum.

    Computes eigenvalues of ``W^T W`` (or ``W W^T`` when more efficient),
    fits a log-log line to (rank, eigenvalue), and returns ``(alpha, r2)``
    where ``alpha = -slope`` and ``r2`` is the coefficient of determination
    of the fit. Returns NaN if the matrix is rank-deficient enough that
    fewer than two non-trivial eigenvalues remain.
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


# --- Radial-FFT α on conv kernels ------------------------------------------
def radial_fft_alpha(conv_w: Array) -> tuple[float, float]:
    """Power-law slope of a conv kernel's radial-frequency spectrum.

    Computes the 2-D FFT of each ``(in_ch, out_ch)`` slice, pools squared
    magnitude over channel dims, radially averages, and fits a log-log
    line to ``(radial_freq, power)``. Returns ``(alpha, r2)`` where
    ``alpha = -slope`` and ``r2`` is the fit's coefficient of determination.

    Known caveat: small kernel sizes (``k <= 5``) under-recover the true
    α because the radial grid has few non-zero bins.
    """
    W_np = np.asarray(conv_w)
    out_ch, in_ch, kH, kW = W_np.shape
    del out_ch, in_ch
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


def per_leaf_alphas(W_tree: dict[str, Array]) -> dict[str, tuple[float, float]]:
    """Compute α per fc-weight leaf (HT-SR) and per conv-weight leaf (radial-FFT).

    Returns a dict keyed by ``"conv1"``, ``"conv2"``, ``"fc1"``, ``"fc2"`` —
    leaf-class names dropping the ``_w`` suffix.
    """
    return {
        "conv1": radial_fft_alpha(W_tree["conv1_w"]),
        "conv2": radial_fft_alpha(W_tree["conv2_w"]),
        "fc1":   ht_sr_alpha(W_tree["fc1_w"]),
        "fc2":   ht_sr_alpha(W_tree["fc2_w"]),
    }


# --- Jacobian σ-spectrum and Hessian top-eig --------------------------------
def sigma_spectrum_op(
    op: Callable[[Any], Any],
    point: Any,
    *,
    k: int = 10,
    num_iterations: int = 60,
    seed: int = 7,
) -> Array:
    """Top-``k`` singular values of ``op``'s Jacobian at ``point``.

    Thin wrapper around ``landscape_archaeology.singular_spectrum`` so the
    seed-as-int / num_iterations defaults are unified across phases.
    """
    return singular_spectrum(op, point, k=k, num_iterations=num_iterations,
                             key=jax.random.key(seed))


def hessian_top(
    params: PyTree,
    batch: dict,
    *,
    k: int = 10,
    num_iterations: int = 40,
    seed: int = 11,
) -> Array:
    """Top-``k`` Hessian eigenvalues of ``cross_entropy_loss`` at ``params``.

    Implemented as a singular spectrum on the gradient operator
    ``grad(loss)`` which equals the eigenspectrum of the symmetric
    Hessian.
    """
    grad_fn = jax.grad(lambda p: mainnet.cross_entropy_loss(p, batch))
    return singular_spectrum(grad_fn, params, k=k, num_iterations=num_iterations,
                             key=jax.random.key(seed))


# --- G_leaf cosine matrix (FWS-hyper only) ----------------------------------
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


# --- Param counting --------------------------------------------------------
def count_params(tree: PyTree) -> int:
    """Total scalar count of all arrays in the pytree."""
    flat, _ = ravel_pytree(tree)
    return int(flat.size)


def global_l2_norm(grad_tree: PyTree) -> Array:
    """Global L2 norm across a pytree of gradients."""
    flat, _ = ravel_pytree(grad_tree)
    return jnp.linalg.norm(flat)
