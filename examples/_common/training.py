"""Stage-0 falsifier + paired 4-arm training orchestration.

Two entry points:

- :func:`stage0_falsifier` runs the FWS-hyper arm under K=1 for each of a
  list of ``G_H`` output-init scale choices and records
  ``sigma_min(d render / d z)`` at the per-phase checkpoints. The verdict
  is decision-ruled at :func:`stage0_verdict`: log10 spread at the final
  step must be ≥ 1.0 OoM for Stage 1 to proceed.

- :func:`paired_train_4arm` runs all four arms (FWS-hyper, FWS-parallel,
  W matched, W overparam) in lockstep over the same batch indices for
  ``K_seed`` seeds and returns a list of per-seed result dicts.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array

from . import arms as arms_module
from . import diagnostics, mainnet
from .arms import Arm


# --- Shared schedule defaults ----------------------------------------------
BATCH_SIZE: int = 128
LOG_EVERY: int = 100


def _make_step(arm: Arm) -> Callable:
    """Return a jitted ``(state, opt_state, batch) -> (state, opt_state, loss, gnorm)`` step."""

    @jax.jit
    def step(state, opt_state, batch):
        loss, grads = jax.value_and_grad(arm.loss_fn)(state, batch)
        gnorm = diagnostics.global_l2_norm(grads)
        updates, new_opt = arm.optimiser.update(grads, opt_state, state)
        new_state = optax.apply_updates(state, updates)
        return new_state, new_opt, loss, gnorm

    return step


def _eval_test_loss_acc(
    params: dict[str, Array],
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    chunk: int = 1024,
) -> tuple[float, float]:
    n = test_x.shape[0]
    total_loss = 0.0
    total_correct = 0
    for i in range(0, n, chunk):
        bx = jnp.asarray(test_x[i:i + chunk])
        by = jnp.asarray(test_y[i:i + chunk])
        logits = jax.vmap(lambda x: mainnet.cnn_forward(params, x))(bx)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(by, mainnet.NUM_CLASSES)
        loss = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
        preds = jnp.argmax(logits, axis=-1)
        total_loss += float(loss) * bx.shape[0]
        total_correct += int(jnp.sum(preds == by))
    return total_loss / n, total_correct / n


def _all_test_preds(
    params: dict[str, Array],
    test_x: np.ndarray,
    *,
    chunk: int = 1024,
) -> np.ndarray:
    n = test_x.shape[0]
    preds = np.empty(n, dtype=np.int32)
    for i in range(0, n, chunk):
        bx = jnp.asarray(test_x[i:i + chunk])
        logits = jax.vmap(lambda x: mainnet.cnn_forward(params, x))(bx)
        preds[i:i + chunk] = np.asarray(jnp.argmax(logits, axis=-1))
    return preds


# --- Stage 0: G_H output-init scale falsifier -------------------------------
@dataclass(frozen=True)
class Stage0Verdict:
    """The outcome of the Stage-0 falsifier.

    ``records[scale_kind]`` is the list of ``(step, sigma_min)`` probes
    for that scale. ``log10_spread`` is the max - min of ``log10(sigma_min)``
    at the final probe step. ``proceed`` is True iff ``log10_spread >=
    threshold_oom`` (default 1.0 OoM).
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
    batch_size: int = BATCH_SIZE,
) -> Stage0Verdict:
    """Run K=1 FWS-hyper training for each ``scale_kind`` and probe σ_min.

    ``arm_factory(scale_kind) -> Arm`` constructs an FWS-hyper arm whose
    ``G_H`` uses the named output-layer init scale. ``sigma_at_z(state)``
    returns the full singular spectrum of ``d render / d z`` at the
    current FWS-hyper state; we record its smallest entry at each of the
    ``checkpoints`` step indices.

    ``threshold_oom`` is the decision rule: if log10 spread of σ_min
    across scales at the final checkpoint is below this many decades,
    the architecture is doing init recovery (not the FWS prior) and
    ``proceed`` is False.
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
        step = _make_step(arm)

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
    return Stage0Verdict(
        records=records,
        final_sigmas=final,
        log10_spread=log_spread,
        proceed=proceed,
        text=text,
        checkpoints=tuple(checkpoints),
    )


# --- Stage 1: paired 4-arm training -----------------------------------------
def paired_train_4arm(
    *,
    arms_list: list[Arm],
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    num_epochs: int = 5,
    K_seed: int = 3,
    batch_size: int = BATCH_SIZE,
    log_every: int = LOG_EVERY,
    per_seed_diagnostics: Callable[[int, dict, dict], dict] | None = None,
) -> tuple[list[dict], list[str], float]:
    """Run all arms on the same batch sequence for ``K_seed`` seeds.

    Returns ``(per_seed_results, nan_or_crash_log, wall_clock_seconds)``.
    Each result dict carries the arm-keyed loss / ckpt trajectories,
    final accuracy, rendered W tree, test predictions, and per-leaf α
    values; ``per_seed_diagnostics(seed, final_states, final_Ws) -> dict``
    is merged in for arm-arbitrary phase-specific extras (e.g. G_leaf
    cosines, σ-spectra at convergence, Hessian top-eigs).
    """
    per_seed: list[dict] = []
    nan_or_crash: list[str] = []
    t0 = time.time()

    for seed in range(K_seed):
        t_seed = time.time()
        print(f"\n--- seed {seed} ---", flush=True)
        try:
            row = _run_seed(
                seed=seed,
                arms_list=arms_list,
                train_x=train_x, train_y=train_y,
                test_x=test_x, test_y=test_y,
                num_epochs=num_epochs,
                batch_size=batch_size,
                log_every=log_every,
                per_seed_diagnostics=per_seed_diagnostics,
            )
            per_seed.append(row)
            if row["any_nan"]:
                nan_or_crash.append(f"seed={seed}: NaN in a loss trajectory")
            line = " ".join(f"{a.short}={row[f'final_{a.short}_acc']:.4f}" for a in arms_list)
            print(f"  seed {seed} done: {line}  ({time.time() - t_seed:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            nan_or_crash.append(f"seed={seed}: CRASH — {type(e).__name__}: {e}")
            print(f"  seed {seed} CRASH — {type(e).__name__}: {e}", flush=True)

    return per_seed, nan_or_crash, time.time() - t0


def _run_seed(
    *,
    seed: int,
    arms_list: list[Arm],
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    num_epochs: int,
    batch_size: int,
    log_every: int,
    per_seed_diagnostics: Callable[[int, dict, dict], dict] | None,
) -> dict:
    """Run all four arms in lockstep for one seed; return a result dict."""
    key = jax.random.key(seed)
    keys = jax.random.split(key, len(arms_list))

    states: dict[str, Any] = {}
    opt_states: dict[str, Any] = {}
    steps: dict[str, Callable] = {}
    for a, k in zip(arms_list, keys, strict=True):
        states[a.short] = a.init(k)
        opt_states[a.short] = a.optimiser.init(states[a.short])
        steps[a.short] = _make_step(a)

    losses: dict[str, list[tuple[int, float, float]]] = {a.short: [] for a in arms_list}
    ckpts: dict[str, list[tuple[int, float, float]]] = {a.short: [] for a in arms_list}

    rng = np.random.default_rng(seed)
    n_train = train_x.shape[0]
    steps_per_epoch = n_train // batch_size

    global_step = 0
    for epoch in range(num_epochs):
        idx = np.arange(n_train)
        rng.shuffle(idx)
        for s in range(steps_per_epoch):
            bi = idx[s * batch_size:(s + 1) * batch_size]
            batch = {"x": jnp.asarray(train_x[bi]), "y": jnp.asarray(train_y[bi])}
            for a in arms_list:
                states[a.short], opt_states[a.short], loss, gn = steps[a.short](
                    states[a.short], opt_states[a.short], batch)
                if global_step % log_every == 0:
                    losses[a.short].append((global_step, float(loss), float(gn)))
            global_step += 1

        for a in arms_list:
            W = a.render_W(states[a.short])
            tl, ta = _eval_test_loss_acc(W, test_x, test_y)
            ckpts[a.short].append((epoch + 1, tl, ta))
        line = " ".join(f"{a.short}={ckpts[a.short][-1][2]:.4f}" for a in arms_list)
        print(f"    [seed {seed}] epoch {epoch + 1}/{num_epochs}: {line}", flush=True)

    final_Ws: dict[str, dict[str, Array]] = {a.short: a.render_W(states[a.short]) for a in arms_list}
    alphas: dict[str, dict] = {a.short: diagnostics.per_leaf_alphas(final_Ws[a.short]) for a in arms_list}
    final_preds: dict[str, np.ndarray] = {a.short: _all_test_preds(final_Ws[a.short], test_x) for a in arms_list}

    any_nan = any(
        not np.isfinite(t[1])
        for arr in losses.values()
        for t in arr
    )

    row: dict = {
        "seed": seed,
        "any_nan": any_nan,
    }
    for a in arms_list:
        row[f"{a.short}_losses"] = np.array(losses[a.short])
        row[f"{a.short}_ckpts"] = np.array(ckpts[a.short])
        row[f"{a.short}_alphas"] = alphas[a.short]
        row[f"{a.short}_test_preds"] = final_preds[a.short]
        row[f"{a.short}_W"] = {k: np.asarray(v) for k, v in final_Ws[a.short].items()}
        row[f"final_{a.short}_acc"] = float(ckpts[a.short][-1][2])

    if per_seed_diagnostics is not None:
        extras = per_seed_diagnostics(seed, states, final_Ws)
        row.update(extras)
    return row


# --- Convenience: silence unused import noise -------------------------------
_ = arms_module  # the package-level import is for re-export convenience
