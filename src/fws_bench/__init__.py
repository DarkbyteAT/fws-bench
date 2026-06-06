"""Experimental orchestration for the FWS programme.

fws-bench houses the harness that runs paired z-space-vs-W-space training,
sweeps the (joint / alternating / meta) optimisation regime axis, partitions
the optax optimiser state between G and z parameter groups, and runs the
activation / renderer / per-group-optimiser ablations described in
docs/experiments/hypernet-ablation-programme.md of the fws repo.

v0.1.0 implements ``Regime.JOINT`` fully — a single forward pass per outer
step updates both G and z via a partitioned ``optax.multi_transform``, in
parallel with a W-direct arm trained on the same task. ``ALTERNATING`` and
``META`` are scaffolded and raise ``NotImplementedError``; their
implementations land in v0.2.0+ once JOINT-arm signal informs the design.
"""

from collections.abc import Callable
from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PyTree


class Regime(Enum):
    """Optimisation regime for the (G, z) parameter group split.

    JOINT: single forward pass, both groups updated per outer step.
    ALTERNATING: fix G, train z for N_z steps; fix z, train G for N_G; repeat.
    META: outer loop optimises G via inner-loop z-adaptation from a prior.
    """

    JOINT = "joint"
    ALTERNATING = "alternating"
    META = "meta"


class TrainResult(NamedTuple):
    """Result of a single training arm.

    Attributes:
        final_params: parameter pytree at the end of training. For the FWS arm
            this is the combined ``{"G": ..., "z": ...}`` pytree the
            partitioned optimiser was driven over; for the W arm this is the
            same pytree structure as ``init_W_params``.
        loss_trajectory: scalar task loss recorded once per outer step.
        grad_norm_trajectory: L2 norm of the parameter-space gradient recorded
            once per outer step.
    """

    final_params: PyTree
    loss_trajectory: Float[Array, " num_steps"]
    grad_norm_trajectory: Float[Array, " num_steps"]


class PairedTrainResult(NamedTuple):
    """Result of paired (FWS arm, W-direct arm) training.

    Attributes:
        fws_arm: trajectory + final params of the FWS arm (G and z trained via
            ``render_fn`` and a partitioned ``optax.multi_transform``).
        w_arm: trajectory + final params of the W-direct arm (W trained
            directly against the same task loss).
        regime: the regime used to schedule the FWS arm's updates.
        num_outer_steps: total outer-loop step count both arms ran for.
    """

    fws_arm: TrainResult
    w_arm: TrainResult
    regime: Regime
    num_outer_steps: int


def _global_l2_norm(grad_tree: PyTree) -> Float[Array, ""]:
    """Global L2 norm of a gradient pytree (flatten, then ``jnp.linalg.norm``)."""
    flat, _ = ravel_pytree(grad_tree)
    return jnp.linalg.norm(flat)


def _train_fws_arm_joint(
    *,
    init_G_params: PyTree,
    init_z_params: PyTree,
    render_fn: Callable[[PyTree, PyTree], PyTree],
    task_loss_fn: Callable[[PyTree, PyTree], Float[Array, ""]],
    task_batch: PyTree,
    G_optimiser: optax.GradientTransformation,
    z_optimiser: optax.GradientTransformation,
    num_outer_steps: int,
) -> TrainResult:
    """Run the FWS arm under the JOINT regime.

    A single ``jax.grad`` call per outer step produces gradients for both G
    and z; ``optax.multi_transform`` routes the G-leaves through
    ``G_optimiser`` and the z-leaves through ``z_optimiser`` while keeping
    their states disjoint.
    """
    combined_params = {"G": init_G_params, "z": init_z_params}
    param_labels = {"G": "G", "z": "z"}
    fws_optimiser = optax.multi_transform(
        {"G": G_optimiser, "z": z_optimiser},
        param_labels,
    )
    opt_state = fws_optimiser.init(combined_params)

    def loss_fn(combined: PyTree, batch: PyTree) -> Float[Array, ""]:
        W = render_fn(combined["G"], combined["z"])
        return task_loss_fn(W, batch)

    def step(carry, _):
        params, state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, task_batch)
        grad_norm = _global_l2_norm(grads)
        updates, new_state = fws_optimiser.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_state), (loss, grad_norm)

    (final_params, _), (loss_trajectory, grad_norm_trajectory) = jax.lax.scan(
        step, (combined_params, opt_state), xs=None, length=num_outer_steps
    )
    return TrainResult(
        final_params=final_params,
        loss_trajectory=loss_trajectory,
        grad_norm_trajectory=grad_norm_trajectory,
    )


def _train_w_arm(
    *,
    init_W_params: PyTree,
    task_loss_fn: Callable[[PyTree, PyTree], Float[Array, ""]],
    task_batch: PyTree,
    W_optimiser: optax.GradientTransformation,
    num_outer_steps: int,
) -> TrainResult:
    """Run the W-direct arm: SGD on ``task_loss_fn`` over ``init_W_params``."""
    opt_state = W_optimiser.init(init_W_params)

    def loss_fn(W: PyTree, batch: PyTree) -> Float[Array, ""]:
        return task_loss_fn(W, batch)

    def step(carry, _):
        params, state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, task_batch)
        grad_norm = _global_l2_norm(grads)
        updates, new_state = W_optimiser.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_state), (loss, grad_norm)

    (final_params, _), (loss_trajectory, grad_norm_trajectory) = jax.lax.scan(
        step, (init_W_params, opt_state), xs=None, length=num_outer_steps
    )
    return TrainResult(
        final_params=final_params,
        loss_trajectory=loss_trajectory,
        grad_norm_trajectory=grad_norm_trajectory,
    )


def paired_train(
    *,
    init_G_params: PyTree,
    init_z_params: PyTree,
    render_fn: Callable[[PyTree, PyTree], PyTree],
    init_W_params: PyTree,
    task_loss_fn: Callable[[PyTree, PyTree], Float[Array, ""]],
    task_batch: PyTree,
    regime: Regime = Regime.JOINT,
    G_optimiser: optax.GradientTransformation | None = None,
    z_optimiser: optax.GradientTransformation | None = None,
    W_optimiser: optax.GradientTransformation | None = None,
    num_outer_steps: int = 1000,
) -> PairedTrainResult:
    """Train W directly (W-arm) and z through ``render(G, z)`` (FWS-arm) on a shared task.

    Both arms see the same ``task_loss_fn`` and ``task_batch`` and run for
    ``num_outer_steps`` outer iterations, recording scalar task loss and the
    L2 norm of the parameter-space gradient at each step. The FWS arm
    partitions the combined ``{"G": ..., "z": ...}`` pytree across
    ``G_optimiser`` and ``z_optimiser`` via ``optax.multi_transform``, so the
    two parameter groups keep separate optimiser state. The W arm trains W
    directly against the same loss; matching the compute budget makes the two
    arms' trajectories comparable along the regime axis.

    Args:
        init_G_params: initial G parameters (e.g. an INR basis pytree).
        init_z_params: initial z modulation parameters.
        render_fn: ``(G, z) -> W``. The reparameterisation through which the
            FWS arm trains z (and, in the JOINT regime, G).
        init_W_params: initial W parameters for the direct-W arm. Same shape
            as ``render_fn(init_G_params, init_z_params)`` for paired
            comparison, though the function does not enforce this.
        task_loss_fn: ``(W, batch) -> scalar``. Shared by both arms.
        task_batch: the training batch (or full dataset) passed unchanged each
            step. A single fixed batch is sufficient for v0.1.0 smoke runs.
        regime: which regime to use for the FWS arm. Only ``JOINT`` is
            implemented in v0.1.0.
        G_optimiser: optax transform for the G partition. Defaults to
            ``optax.adam(1e-3)`` when ``None``.
        z_optimiser: optax transform for the z partition. Defaults to
            ``optax.adam(1e-3)`` when ``None``.
        W_optimiser: optax transform for the direct-W arm. Defaults to
            ``optax.adam(1e-3)`` when ``None``.
        num_outer_steps: outer-loop budget for both arms.

    Returns:
        A ``PairedTrainResult`` carrying both arms' final params and
        per-step ``loss`` / ``grad_norm`` trajectories.

    Raises:
        NotImplementedError: when ``regime`` is ``ALTERNATING`` or ``META``;
            those regimes ship in v0.2.0+. See the fws-bench Trello board for
            the prototype cards.
    """
    if regime is not Regime.JOINT:
        raise NotImplementedError(
            f"fws-bench v0.1.0 implements Regime.JOINT only; "
            f"Regime.{regime.value.upper()} is scaffolded for v0.2.0+. "
            f"See https://trello.com/b/dYJqhfUV/fws-bench for the prototype cards"
        )

    G_optimiser = G_optimiser if G_optimiser is not None else optax.adam(1e-3)
    z_optimiser = z_optimiser if z_optimiser is not None else optax.adam(1e-3)
    W_optimiser = W_optimiser if W_optimiser is not None else optax.adam(1e-3)

    fws_arm = _train_fws_arm_joint(
        init_G_params=init_G_params,
        init_z_params=init_z_params,
        render_fn=render_fn,
        task_loss_fn=task_loss_fn,
        task_batch=task_batch,
        G_optimiser=G_optimiser,
        z_optimiser=z_optimiser,
        num_outer_steps=num_outer_steps,
    )
    w_arm = _train_w_arm(
        init_W_params=init_W_params,
        task_loss_fn=task_loss_fn,
        task_batch=task_batch,
        W_optimiser=W_optimiser,
        num_outer_steps=num_outer_steps,
    )
    return PairedTrainResult(
        fws_arm=fws_arm,
        w_arm=w_arm,
        regime=regime,
        num_outer_steps=num_outer_steps,
    )


__all__ = ["PairedTrainResult", "Regime", "TrainResult", "paired_train"]
__version__ = "0.1.0"
