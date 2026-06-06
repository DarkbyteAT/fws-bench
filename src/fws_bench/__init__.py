"""Experimental orchestration for the FWS programme.

fws-bench houses the harness that runs paired z-space-vs-W-space training,
sweeps the (joint / alternating / meta) optimisation regime axis, partitions
the optax optimiser state between G and z parameter groups, and runs the
activation / renderer / per-group-optimiser ablations described in
docs/experiments/hypernet-ablation-programme.md of the fws repo.

This is the v0.0.0 scaffold; the public surface below is placeholder and
returns NotImplementedError. The first real cell - paired training across a
single (regime, optimiser, mainnet-activation) point - lands as the Trello
board's "First z-space vs W-space paired-training prototype" card.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

from jaxtyping import PyTree


class Regime(Enum):
    """Optimisation regime for the (G, z) parameter group split.

    JOINT: single forward pass, both groups updated per outer step.
    ALTERNATING: fix G, train z for N_z steps; fix z, train G for N_G; repeat.
    META: outer loop optimises G via inner-loop z-adaptation from a prior.
    """

    JOINT = "joint"
    ALTERNATING = "alternating"
    META = "meta"


def paired_train(
    render_fn: Callable[[PyTree], PyTree],
    task_loss_fn: Callable[[PyTree, Any], Any],
    *,
    regime: Regime = Regime.JOINT,
    num_outer_steps: int = 1000,
) -> dict[str, Any]:
    """Train W directly (W-arm) and z through render(G, z) (FWS-arm) on the same task.

    Args:
        render_fn: the reparameterisation function (e.g. loom.render bound to G and P).
        task_loss_fn: scalar loss against a task data batch.
        regime: which optimisation regime to use for the FWS-arm.
        num_outer_steps: total outer-loop step budget per arm.

    Returns:
        A dict with keys "fws_arm" and "w_arm", each containing training curves,
        rendered-weight diagnostics, and final test metrics.
    """
    raise NotImplementedError(
        "fws-bench v0.0.0 is a scaffold; see https://trello.com/b/dYJqhfUV/fws-bench for the first prototype card"
    )


__all__ = ["Regime", "paired_train"]
__version__ = "0.0.0"
