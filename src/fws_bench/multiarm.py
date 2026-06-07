"""N-arm paired-training harness for ablation runs.

Extends :func:`fws_bench.paired_train` (two arms — one FWS, one
direct-W) to an arbitrary list of arms run in lockstep on the same batch
sequence. The motivating use case is the phase 8/9/10 four-arm grid
(FWS-hyper, FWS-parallel, W matched, W overparam) but nothing in this
module commits to any particular arm shape: each :class:`Arm` carries
its own ``init``, ``loss_fn``, ``render_W``, and ``optimiser``, and the
harness drives them through identical batch sequences across ``K_seed``
seeds.

What is and is not in this module:

- **In**: the Arm value type, the jitted per-arm step factory, the
  lockstep K_seed training loop, NaN / crash tracking, the per-seed
  results dict aggregation.

- **Out**: the dataset, the mainnet forward function, the test-accuracy
  evaluator, the per-seed phase-specific diagnostics. All of these are
  passed in as callbacks so the harness stays task-agnostic.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree


@dataclass(frozen=True)
class Arm:
    """One arm of an N-arm paired training contract.

    Fields:

    - ``name``  — human-readable label (e.g. ``"FWS-hyper"``).
    - ``short`` — slug used in figure filenames and result-dict keys.
    - ``color`` — palette hex string for downstream plots.
    - ``init`` — ``PRNGKey -> trainable_state``. Any pytree; for FWS arms
      it is typically ``{"G": G_H, "z": z}``, for direct-W arms it is a
      leaf-dict, for factored arms it is an Equinox module.
    - ``loss_fn`` — ``(state, batch) -> scalar``.
    - ``render_W`` — ``state -> rendered W tree``. Lets downstream
      diagnostics index by leaf without knowing the arm shape. Identity
      for direct-W arms.
    - ``optimiser`` — pre-built optax gradient transformation. For
      partitioned arms (e.g. FWS-hyper with separate G / z optimisers)
      use :func:`optax.multi_transform` to build the partition before
      constructing the Arm.
    """

    name: str
    short: str
    color: str
    init: Callable[[Array], Any]
    loss_fn: Callable[[Any, PyTree], Array]
    render_W: Callable[[Any], PyTree]
    optimiser: optax.GradientTransformation


def _global_l2_norm(grad_tree: PyTree) -> Array:
    flat, _ = ravel_pytree(grad_tree)
    return jnp.linalg.norm(flat)


def make_step(arm: Arm) -> Callable:
    """Build a JIT'd ``(state, opt_state, batch) -> (new_state, new_opt, loss, gnorm)`` step.

    Captures the arm's ``loss_fn`` and ``optimiser`` into the closure so
    the same step function can be reused across all seeds for that arm.
    """

    @jax.jit
    def step(state, opt_state, batch):
        loss, grads = jax.value_and_grad(arm.loss_fn)(state, batch)
        gnorm = _global_l2_norm(grads)
        updates, new_opt = arm.optimiser.update(grads, opt_state, state)
        new_state = optax.apply_updates(state, updates)
        return new_state, new_opt, loss, gnorm

    return step


# Type aliases for the callback shapes the harness consumes.
EvalFn = Callable[[PyTree], tuple[float, float]]
"""``rendered_W -> (test_loss, test_acc)``."""

PredictFn = Callable[[PyTree], np.ndarray]
"""``rendered_W -> int test-prediction array (length n_test)``."""

PerSeedDiagnosticsFn = Callable[[int, dict[str, Any], dict[str, PyTree]], dict]
"""``(seed, arm_states, rendered_W_per_arm) -> dict``.

Returned dict is merged into the seed's result row; lets phase code
inject σ-spectra, Hessian top-eigs, etc.
"""


def paired_train_4arm(
    *,
    arms: list[Arm],
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_fn: EvalFn,
    predict_fn: PredictFn,
    num_epochs: int = 5,
    K_seed: int = 3,
    batch_size: int = 128,
    log_every: int = 100,
    per_seed_diagnostics: PerSeedDiagnosticsFn | None = None,
) -> tuple[list[dict], list[str], float]:
    """Run every arm on the same batch sequence for ``K_seed`` seeds.

    Each seed runs all arms in lockstep — same shuffle order, same batch
    indices, same number of outer steps — so the resulting trajectories
    are paired and directly comparable.

    Args:
        arms: arms to train in parallel. Each :class:`Arm` brings its own
            init / loss / render / optimiser. Order is preserved in result
            keys: row keys are ``{arm.short}_losses``, ``{arm.short}_ckpts``,
            ``final_{arm.short}_acc``, etc.
        train_x: training inputs of shape ``(n_train, ...)``. The harness
            shuffles ``train_x.shape[0]`` indices per epoch.
        train_y: integer training labels of shape ``(n_train,)``.
        eval_fn: ``rendered_W -> (test_loss, test_acc)``. Called once per
            arm per epoch to populate the per-arm ``_ckpts`` array. The
            harness does not know what ``test_loss`` or ``test_acc`` mean.
        predict_fn: ``rendered_W -> int test-prediction array``. Called
            once per arm at the end of training for the seed; used by
            downstream curated-prediction plots.
        num_epochs: outer epochs per seed.
        K_seed: number of seeds; results are stacked across seeds.
        batch_size: minibatch size for the inner training loop.
        log_every: record ``(step, loss, gnorm)`` every Nth step.
        per_seed_diagnostics: optional ``(seed, arm_states, rendered_W_per_arm)
            -> dict``. Result dict is merged into the seed's row before
            being appended to the output list.

    Returns:
        ``(per_seed_results, nan_or_crash_log, wall_clock_seconds)``.

        ``per_seed_results`` is a list of result dicts (one per seed), each
        carrying per-arm loss/ckpt trajectories, final rendered W tree,
        test predictions, final accuracy, plus any extras from
        ``per_seed_diagnostics``.

        ``nan_or_crash_log`` carries one string per seed that hit a NaN
        in its loss trajectory or raised inside ``_run_seed``.

        ``wall_clock_seconds`` is the total wall time across all seeds.
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
                arms=arms,
                train_x=train_x, train_y=train_y,
                eval_fn=eval_fn, predict_fn=predict_fn,
                num_epochs=num_epochs,
                batch_size=batch_size,
                log_every=log_every,
                per_seed_diagnostics=per_seed_diagnostics,
            )
            per_seed.append(row)
            if row["any_nan"]:
                nan_or_crash.append(f"seed={seed}: NaN in a loss trajectory")
            line = " ".join(f"{a.short}={row[f'final_{a.short}_acc']:.4f}" for a in arms)
            print(f"  seed {seed} done: {line}  ({time.time() - t_seed:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            nan_or_crash.append(f"seed={seed}: CRASH — {type(e).__name__}: {e}")
            print(f"  seed {seed} CRASH — {type(e).__name__}: {e}", flush=True)

    return per_seed, nan_or_crash, time.time() - t0


def _run_seed(
    *,
    seed: int,
    arms: list[Arm],
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_fn: EvalFn,
    predict_fn: PredictFn,
    num_epochs: int,
    batch_size: int,
    log_every: int,
    per_seed_diagnostics: PerSeedDiagnosticsFn | None,
) -> dict:
    key = jax.random.key(seed)
    keys = jax.random.split(key, len(arms))

    states: dict[str, Any] = {}
    opt_states: dict[str, Any] = {}
    steps: dict[str, Callable] = {}
    for a, k in zip(arms, keys, strict=True):
        states[a.short] = a.init(k)
        opt_states[a.short] = a.optimiser.init(states[a.short])
        steps[a.short] = make_step(a)

    losses: dict[str, list[tuple[int, float, float]]] = {a.short: [] for a in arms}
    ckpts: dict[str, list[tuple[int, float, float]]] = {a.short: [] for a in arms}

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
            for a in arms:
                states[a.short], opt_states[a.short], loss, gn = steps[a.short](
                    states[a.short], opt_states[a.short], batch)
                if global_step % log_every == 0:
                    losses[a.short].append((global_step, float(loss), float(gn)))
            global_step += 1

        for a in arms:
            W = a.render_W(states[a.short])
            tl, ta = eval_fn(W)
            ckpts[a.short].append((epoch + 1, tl, ta))
        line = " ".join(f"{a.short}={ckpts[a.short][-1][2]:.4f}" for a in arms)
        print(f"    [seed {seed}] epoch {epoch + 1}/{num_epochs}: {line}", flush=True)

    final_Ws: dict[str, PyTree] = {a.short: a.render_W(states[a.short]) for a in arms}
    final_preds: dict[str, np.ndarray] = {a.short: predict_fn(final_Ws[a.short]) for a in arms}

    any_nan = any(
        not np.isfinite(t[1])
        for arr in losses.values()
        for t in arr
    )

    row: dict = {
        "seed": seed,
        "any_nan": any_nan,
    }
    for a in arms:
        row[f"{a.short}_losses"] = np.array(losses[a.short])
        row[f"{a.short}_ckpts"] = np.array(ckpts[a.short])
        row[f"{a.short}_test_preds"] = final_preds[a.short]
        row[f"{a.short}_W"] = {k: np.asarray(v) for k, v in final_Ws[a.short].items()}
        row[f"final_{a.short}_acc"] = float(ckpts[a.short][-1][2])

    if per_seed_diagnostics is not None:
        extras = per_seed_diagnostics(seed, states, final_Ws)
        row.update(extras)
    return row


__all__ = ["Arm", "EvalFn", "PerSeedDiagnosticsFn", "PredictFn", "make_step", "paired_train_4arm"]
