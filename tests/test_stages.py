"""Behaviour tests for the Stage-0 G_H output-init scale falsifier."""

import jax.numpy as jnp
import numpy as np
import optax
import pytest

import fws_bench


pytestmark = pytest.mark.unit


def _toy_arm_with_scale(scale: float) -> fws_bench.Arm:
    """Toy single-param arm whose σ probe trivially depends on the input scale.

    State is a single scalar trained against a fixed scalar target. We
    don't actually use the training; only the σ-probe callback matters
    for testing the orchestration loop.
    """

    def init(key):
        del key
        return {"w": jnp.array(scale)}

    def loss_fn(state, batch):
        del batch
        return state["w"] ** 2  # scalar — Stage 0 doesn't care, LR is 0 anyway

    return fws_bench.Arm(
        name=f"scale={scale}", short="toy", color="#000000",
        init=init, loss_fn=loss_fn,
        render_W=lambda s: {"w": s["w"]},
        optimiser=optax.adam(0.0),  # zero LR — state doesn't drift
    )


def _toy_dataset() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((32, 1), dtype=np.float32),
        np.ones((32,), dtype=np.float32),
    )


# --- Verdict shape ----------------------------------------------------------
def test_stage0_verdict_carries_records_spread_and_decision():
    # Given: three init scales differing by ≥1 OoM, a σ-probe that just
    # returns the trained scalar (so σ_min equals the scale).
    scale_values = {"small": 0.01, "medium": 1.0, "big": 100.0}
    train_x, train_y = _toy_dataset()

    def sigma_at_z(state):
        return jnp.array([float(state["w"])])

    # When: we run the falsifier with num_steps=0 so the values stay at init.
    verdict = fws_bench.stage0_falsifier(
        arm_factory=lambda sk: _toy_arm_with_scale(scale_values[sk]),
        scale_kinds=tuple(scale_values.keys()),
        train_x=train_x, train_y=train_y,
        sigma_at_z=sigma_at_z,
        num_steps=0,
        checkpoints=(0,),
        threshold_oom=1.0,
    )

    # Then: records, final_sigmas, spread, and decision are populated
    # consistently.
    assert set(verdict.records.keys()) == {"small", "medium", "big"}
    assert verdict.final_sigmas["small"] == pytest.approx(0.01, rel=1e-3)
    assert verdict.final_sigmas["big"] == pytest.approx(100.0, rel=1e-3)
    # log10(100) - log10(0.01) = 4 decades, > 1 OoM threshold
    assert verdict.log10_spread == pytest.approx(4.0, abs=1e-3)
    assert verdict.proceed is True
    assert "PROCEED" in verdict.text


def test_stage0_verdict_says_stop_when_spread_under_threshold():
    # Given: three scales all within an order of magnitude of each other.
    scale_values = {"a": 1.0, "b": 2.0, "c": 5.0}
    train_x, train_y = _toy_dataset()

    def sigma_at_z(state):
        return jnp.array([float(state["w"])])

    # When: we run the falsifier with threshold 1.0 OoM (spread will be < 1).
    verdict = fws_bench.stage0_falsifier(
        arm_factory=lambda sk: _toy_arm_with_scale(scale_values[sk]),
        scale_kinds=tuple(scale_values.keys()),
        train_x=train_x, train_y=train_y,
        sigma_at_z=sigma_at_z,
        num_steps=0,
        checkpoints=(0,),
        threshold_oom=1.0,
    )

    # Then: the verdict says STOP and proceed is False.
    assert verdict.proceed is False
    assert verdict.log10_spread < 1.0
    assert "STOP" in verdict.text


def test_stage0_default_checkpoints_match_num_steps():
    # Given: no explicit checkpoints argument.
    scale_values = {"only": 1.0}
    train_x, train_y = _toy_dataset()

    # When: we run with num_steps=100 — defaults should be (0, 100, 1000, 100).
    # The function explicitly handles num_steps below 1000 — checkpoints still
    # include 0, 100, 1000, num_steps; only those that fall inside are probed.
    verdict = fws_bench.stage0_falsifier(
        arm_factory=lambda sk: _toy_arm_with_scale(scale_values[sk]),
        scale_kinds=tuple(scale_values.keys()),
        train_x=train_x, train_y=train_y,
        sigma_at_z=lambda state: jnp.array([float(state["w"])]),
        num_steps=100,
    )

    # Then: the final checkpoint equals num_steps.
    assert verdict.checkpoints[-1] == 100


def test_stage0_threshold_oom_is_load_bearing():
    # Given: a fixed spread of exactly 1.0 decades.
    scale_values = {"low": 1.0, "high": 10.0}
    train_x, train_y = _toy_dataset()

    def sigma_at_z(state):
        return jnp.array([float(state["w"])])

    # When: we vary the threshold across the spread boundary.
    permissive = fws_bench.stage0_falsifier(
        arm_factory=lambda sk: _toy_arm_with_scale(scale_values[sk]),
        scale_kinds=tuple(scale_values.keys()),
        train_x=train_x, train_y=train_y,
        sigma_at_z=sigma_at_z,
        num_steps=0, checkpoints=(0,), threshold_oom=0.5,
    )
    strict = fws_bench.stage0_falsifier(
        arm_factory=lambda sk: _toy_arm_with_scale(scale_values[sk]),
        scale_kinds=tuple(scale_values.keys()),
        train_x=train_x, train_y=train_y,
        sigma_at_z=sigma_at_z,
        num_steps=0, checkpoints=(0,), threshold_oom=2.0,
    )

    # Then: the spread is the same but the verdicts flip with the threshold.
    assert permissive.log10_spread == pytest.approx(strict.log10_spread, abs=1e-3)
    assert permissive.proceed is True
    assert strict.proceed is False


# --- Public surface ---------------------------------------------------------
def test_public_surface_includes_stage0_falsifier_and_verdict():
    assert "stage0_falsifier" in fws_bench.__all__
    assert "StageZeroVerdict" in fws_bench.__all__
    assert hasattr(fws_bench, "stage0_falsifier")
    assert hasattr(fws_bench, "StageZeroVerdict")
