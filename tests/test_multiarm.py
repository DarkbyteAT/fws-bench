"""Behaviour tests for the N-arm paired training harness."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import fws_bench


pytestmark = pytest.mark.unit


# --- Toy arm fixtures --------------------------------------------------------
def _linear_arm(name: str, short: str, lr: float = 1e-2) -> fws_bench.Arm:
    """A toy 1-layer linear-regression arm over a single weight matrix.

    state: ``{"W": (4, 4) matrix}``. loss: mean squared error against a fixed
    target broadcast batch. render_W: identity (state already has the W tree
    shape). Lets us exercise the harness without depending on CIFAR-10 or
    any specific mainnet.
    """

    def init(key):
        return {"W": jax.random.normal(key, (4, 4))}

    def loss_fn(state, batch):
        pred = batch["x"] @ state["W"]
        return jnp.mean((pred - batch["y"]) ** 2)

    return fws_bench.Arm(
        name=name,
        short=short,
        color="#000000",
        init=init,
        loss_fn=loss_fn,
        render_W=lambda state: state,
        optimiser=optax.adam(lr),
    )


def _toy_dataset(n: int = 256, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    y = (x @ np.eye(4)).astype(np.float32)  # trivially learnable target
    return x, y


# --- Arm shape ---------------------------------------------------------------
def test_arm_is_a_frozen_dataclass_with_seven_fields():
    # Given: the Arm value type.
    # When: we instantiate one.
    arm = _linear_arm("Toy", "toy")

    # Then: it carries exactly the seven contract fields.
    assert arm.name == "Toy"
    assert arm.short == "toy"
    assert arm.color == "#000000"
    assert callable(arm.init)
    assert callable(arm.loss_fn)
    assert callable(arm.render_W)
    assert isinstance(arm.optimiser, optax.GradientTransformation)


def test_arm_is_frozen():
    # Given: an Arm.
    arm = _linear_arm("Toy", "toy")

    # When: we try to mutate a field.
    # Then: the frozen dataclass rejects the mutation with FrozenInstanceError.
    with pytest.raises(FrozenInstanceError):
        arm.name = "Mutated"  # type: ignore[misc]


# --- paired_train_4arm -------------------------------------------------------
def test_paired_train_drives_n_arms_in_lockstep():
    # Given: two toy arms + a small dataset + a no-op eval / predict.
    arm_a = _linear_arm("A", "a")
    arm_b = _linear_arm("B", "b")
    x, y = _toy_dataset(n=64)

    def eval_fn(W):
        return 0.5, 0.7  # constant; the harness should still record per-epoch ckpts

    def predict_fn(W):
        return np.zeros(8, dtype=np.int32)

    # When: we run paired_train_4arm for K=2 seeds.
    per_seed, nan_log, wall = fws_bench.paired_train_4arm(
        arms=[arm_a, arm_b],
        train_x=x, train_y=y,
        eval_fn=eval_fn, predict_fn=predict_fn,
        num_epochs=2, K_seed=2, batch_size=16, log_every=1,
    )

    # Then: results have per-seed dicts, no NaN log, all arms present.
    assert len(per_seed) == 2
    assert nan_log == []
    assert wall > 0
    for row in per_seed:
        for short in ("a", "b"):
            assert f"{short}_losses" in row
            assert f"{short}_ckpts" in row
            assert f"final_{short}_acc" in row
            assert f"{short}_W" in row
            assert row[f"{short}_ckpts"].shape == (2, 3)  # (n_epochs, [epoch, loss, acc])
            assert row[f"{short}_test_preds"].shape == (8,)


def test_per_seed_diagnostics_callback_merged_into_row():
    # Given: a single arm and a per_seed_diagnostics callback that injects
    # a sentinel key into each seed's row.
    arm = _linear_arm("A", "a")
    x, y = _toy_dataset(n=32)

    def diagnostics(seed, states, final_Ws):
        return {"sentinel": seed * 10}

    # When: we run paired_train_4arm with the callback.
    per_seed, _, _ = fws_bench.paired_train_4arm(
        arms=[arm],
        train_x=x, train_y=y,
        eval_fn=lambda W: (0.0, 0.0), predict_fn=lambda W: np.zeros(4, dtype=np.int32),
        num_epochs=1, K_seed=2, batch_size=8,
        per_seed_diagnostics=diagnostics,
    )

    # Then: every seed's row carries the injected sentinel key.
    assert per_seed[0]["sentinel"] == 0
    assert per_seed[1]["sentinel"] == 10


def test_arm_diagnostics_at_convergence_namespaced_under_short():
    # Given: two arms, each with its own diagnostics_at_convergence callback
    # that returns leaf-keyed dicts. The harness should merge each return
    # under "{arm.short}.<key>" namespaces so two arms never collide on the
    # same dict key.
    arm_a = _linear_arm("A", "a")
    arm_b = _linear_arm("B", "b")

    def diag_a(state, eval_batch):
        return {"alphas": {"leaf": (1.0, 0.9)}, "norm": float(jnp.linalg.norm(state["W"]))}

    def diag_b(state, eval_batch):
        return {"alphas": {"leaf": (2.0, 0.5)}}

    arm_a = fws_bench.Arm(
        name=arm_a.name, short=arm_a.short, color=arm_a.color,
        init=arm_a.init, loss_fn=arm_a.loss_fn, render_W=arm_a.render_W,
        optimiser=arm_a.optimiser, diagnostics_at_convergence=diag_a,
    )
    arm_b = fws_bench.Arm(
        name=arm_b.name, short=arm_b.short, color=arm_b.color,
        init=arm_b.init, loss_fn=arm_b.loss_fn, render_W=arm_b.render_W,
        optimiser=arm_b.optimiser, diagnostics_at_convergence=diag_b,
    )
    x, y = _toy_dataset(n=32)

    # When: we run K=1 with the two arms wired up.
    per_seed, _, _ = fws_bench.paired_train_4arm(
        arms=[arm_a, arm_b],
        train_x=x, train_y=y,
        eval_fn=lambda W: (0.0, 0.0), predict_fn=lambda W: np.zeros(4, dtype=np.int32),
        num_epochs=1, K_seed=1, batch_size=8,
        diagnostics_batch_size=8,
    )

    # Then: each arm's diagnostic keys live under its own namespace.
    row = per_seed[0]
    assert row["a.alphas"]["leaf"] == (1.0, 0.9)
    assert row["b.alphas"]["leaf"] == (2.0, 0.5)
    assert "a.norm" in row
    assert "b.norm" not in row


def test_nan_in_loss_trajectory_logged():
    # Given: an arm whose loss is always NaN.
    def init(key):
        return {"W": jnp.array(jnp.nan)}

    def loss_fn(state, batch):
        return state["W"]  # propagates NaN

    nan_arm = fws_bench.Arm(
        name="NaN", short="nan", color="#000000", init=init,
        loss_fn=loss_fn, render_W=lambda s: {"W": s["W"]},
        optimiser=optax.adam(1e-3),
    )
    x, y = _toy_dataset(n=16)

    # When: we run.
    _, nan_log, _ = fws_bench.paired_train_4arm(
        arms=[nan_arm],
        train_x=x, train_y=y,
        eval_fn=lambda W: (0.0, 0.0), predict_fn=lambda W: np.zeros(4, dtype=np.int32),
        num_epochs=1, K_seed=1, batch_size=4, log_every=1,
    )

    # Then: a "NaN in a loss trajectory" entry appears.
    assert any("NaN" in entry for entry in nan_log)


# --- Public surface ---------------------------------------------------------
def test_public_surface_includes_arm_and_multi_arm_training():
    # Given: the package's public exports.
    # When: we inspect __all__.
    # Then: the new multi-arm surface is exposed.
    assert "Arm" in fws_bench.__all__
    assert "paired_train_4arm" in fws_bench.__all__
    assert hasattr(fws_bench, "Arm")
    assert hasattr(fws_bench, "paired_train_4arm")
