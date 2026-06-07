"""Stage-0 ``G_H`` output-init scale falsifier.

A blocking K=1 gate run before any K=3+ paired training. For each
candidate ``G_H`` output-layer init scale, train the FWS-hyper arm under
the same task loss for ``num_steps`` outer steps and probe
``σ_min(∂ render / ∂ z)`` at a set of checkpoints. At the final
checkpoint, compute the log10 spread of ``σ_min`` across scales: if it
is below ``threshold_oom`` decades, the architecture is doing init
recovery (not the FWS prior) — STOP.

This module owns the *orchestration* of the falsifier: the per-scale
training loop, the checkpoint probing, the spread calculation, and the
decision rule. It is operator-agnostic — the σ-probe is passed in as a
callback (``sigma_at_z``) so the same harness drives phase-8's
SiLU-MLP, phase-9's recursive SIREN, and phase-10's polar-projection
variants without ceremony.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from fws_bench.multiarm import Arm, make_step


@dataclass(frozen=True)
class StageZeroVerdict:
    """Outcome of the Stage-0 falsifier.

    Attributes:
        records: ``{scale_kind: [(step, sigma_min), ...]}`` — the per-scale
            probe trajectory.
        final_sigmas: ``{scale_kind: sigma_min_at_final_step}``.
        log10_spread: ``max(log10 σ_min) - min(log10 σ_min)`` at the final
            probe step, across ``scale_kinds``.
        proceed: True iff ``log10_spread >= threshold_oom``. False ⇒
            init recovery — Stage 1 K=3 must be skipped.
        text: human-readable verdict string, printed by the harness and
            included in the research log.
        checkpoints: probe steps used (last entry equals ``num_steps``).
    """

    records: dict[str, list[tuple[int, float]]]
    final_sigmas: dict[str, float]
    log10_spread: float
    proceed: bool
    text: str
    checkpoints: tuple[int, ...]


def stage0_falsifier(
    *,
    arm_factory: Callable[[str], Arm],
    scale_kinds: tuple[str, ...],
    train_x: np.ndarray,
    train_y: np.ndarray,
    sigma_at_z: Callable[[Any], Array],
    num_steps: int = 5000,
    checkpoints: tuple[int, ...] | None = None,
    threshold_oom: float = 1.0,
    seed: int = 0,
    batch_size: int = 128,
) -> StageZeroVerdict:
    """Run K=1 FWS-hyper training for each ``scale_kind`` and probe σ_min.

    Args:
        arm_factory: ``scale_kind -> Arm``. Constructs an FWS-hyper arm
            whose ``G_H`` uses the named output-layer init scale. Same
            shape across phases; only the ``G_H`` architecture varies.
        scale_kinds: tuple of named init-scale choices to compare.
        train_x: training inputs of shape ``(n_train, ...)``. The harness
            shuffles indices per epoch but does not evaluate on test data —
            the kill rule is structural, on the σ-probe.
        train_y: integer training labels of shape ``(n_train,)``.
        sigma_at_z: ``arm_state -> singular spectrum array``. Wraps a
            phase-specific call to
            :func:`landscape_archaeology.singular_spectrum` so the
            harness does not need to know the arm's state pytree shape.
            The smallest entry of the returned spectrum is taken as
            ``σ_min``.
        num_steps: outer-step count per scale_kind.
        checkpoints: probe steps. Defaults to ``(0, 100, 1000, num_steps)``.
        threshold_oom: minimum log10 spread (in decades) at the final
            checkpoint for the verdict to be PROCEED. Default 1.0 OoM.
        seed: PRNG / shuffle seed; same value used for every scale_kind so
            the comparison is fair.
        batch_size: minibatch size.

    Returns:
        A :class:`StageZeroVerdict` carrying the per-scale records, the
        final spread, the proceed/stop decision, and a printable
        verdict text.
    """
    if checkpoints is None:
        checkpoints = (0, 100, 1000, num_steps)
    print("=" * 72)
    print("Stage 0 — G_H output-init scale falsifier (BLOCKING)")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    n_train = train_x.shape[0]
    records: dict[str, list[tuple[int, float]]] = {}

    for scale_kind in scale_kinds:
        print(f"\n  scale_kind={scale_kind}  steps={num_steps}", flush=True)
        arm = arm_factory(scale_kind)
        key = jax.random.key(seed)
        state = arm.init(key)
        opt_state = arm.optimiser.init(state)
        step = make_step(arm)

        recs: list[tuple[int, float]] = []
        idx = np.arange(n_train)
        rng.shuffle(idx)
        cursor = 0
        last_print = time.time()

        for step_i in range(num_steps + 1):
            if step_i in checkpoints:
                sig = np.asarray(sigma_at_z(state))
                sigma_min = float(sig[-1])
                recs.append((step_i, sigma_min))
                print(f"    step {step_i:>5d}: sigma_min(d render / d z) = {sigma_min:.6g}",
                      flush=True)
                last_print = time.time()
            if step_i == num_steps:
                break
            if cursor + batch_size > n_train:
                rng.shuffle(idx)
                cursor = 0
            bi = idx[cursor:cursor + batch_size]
            cursor += batch_size
            batch = {"x": jnp.asarray(train_x[bi]), "y": jnp.asarray(train_y[bi])}
            state, opt_state, _, _ = step(state, opt_state, batch)
            if time.time() - last_print > 60:
                print(f"    progress: step {step_i} (between checkpoints, no probe)",
                      flush=True)
                last_print = time.time()

        records[scale_kind] = recs

    final = {k: v[-1][1] for k, v in records.items()}
    vals = np.array(list(final.values()))
    log_vals = np.log10(np.maximum(vals, np.finfo(vals.dtype).tiny))
    log_spread = float(log_vals.max() - log_vals.min())
    proceed = log_spread >= threshold_oom
    text = (
        f"sigma_min at step {num_steps}: "
        + ", ".join(f"{k}={final[k]:.4g}" for k in scale_kinds)
        + f"  (log10 spread = {log_spread:.3f})\n"
        + (f"DECISION: spread >= {threshold_oom:.1f} OoM — FWS prior doing geometric work; PROCEED to K=3."
           if proceed else
           f"DECISION: spread < {threshold_oom:.1f} OoM — init recovery, not FWS prior; STOP, do not run K=3.")
    )
    print("\n" + text, flush=True)
    return StageZeroVerdict(
        records=records,
        final_sigmas=final,
        log10_spread=log_spread,
        proceed=proceed,
        text=text,
        checkpoints=tuple(checkpoints),
    )


__all__ = ["StageZeroVerdict", "stage0_falsifier"]
