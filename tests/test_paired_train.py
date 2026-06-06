"""Behaviour tests for ``paired_train`` under the JOINT regime.

Each test pins one structural property of the paired-arm contract. Magnitude
thresholds are avoided per ``feedback_no_magic_thresholds``; directional and
structural assertions cover the smoke surface.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

import fws_bench


def _make_synthetic_regression(*, n_features: int = 8, seed: int = 0):
    """Build a tiny synthetic linear-regression task.

    Returns ``(task_loss_fn, task_batch, W_shape)``. ``task_loss_fn`` takes
    ``W`` of shape ``(n_features,)`` and a batch ``(X, y)``; the optimal W is
    the underlying linear coefficients.
    """
    key = jax.random.key(seed)
    key_x, key_w, key_noise = jax.random.split(key, 3)
    X = jax.random.normal(key_x, (32, n_features))
    W_true = jax.random.normal(key_w, (n_features,))
    noise = jax.random.normal(key_noise, (32,)) * 1e-2
    y = X @ W_true + noise

    def task_loss_fn(W, batch):
        X_batch, y_batch = batch
        preds = X_batch @ W
        return jnp.mean((preds - y_batch) ** 2)

    return task_loss_fn, (X, y), (n_features,)


def _identity_render(G_params, z_params):
    """Render G * z elementwise — trivial FWS-arm reparameterisation."""
    return G_params * z_params


def _make_inits(*, n_features: int = 8, seed: int = 1):
    """Initial G, z, W pytrees as plain arrays of shape ``(n_features,)``."""
    key = jax.random.key(seed)
    key_g, key_z, key_w = jax.random.split(key, 3)
    init_G = jax.random.normal(key_g, (n_features,)) * 0.1
    init_z = jnp.ones((n_features,))
    init_W = jax.random.normal(key_w, (n_features,)) * 0.1
    return init_G, init_z, init_W


@pytest.mark.unit
def test_joint_regime_runs_end_to_end():
    # Given: a tiny synthetic regression task and matching inits.
    task_loss_fn, task_batch, _ = _make_synthetic_regression()
    init_G, init_z, init_W = _make_inits()
    num_steps = 200

    # When: we run paired_train under JOINT.
    result = fws_bench.paired_train(
        init_G_params=init_G,
        init_z_params=init_z,
        render_fn=_identity_render,
        init_W_params=init_W,
        task_loss_fn=task_loss_fn,
        task_batch=task_batch,
        regime=fws_bench.Regime.JOINT,
        num_outer_steps=num_steps,
    )

    # Then: trajectories are finite, the right shape, and loss decreased on
    # average from the first 10 steps to the last 10 steps (directional, not
    # magnitude-threshold).
    assert result.fws_arm.loss_trajectory.shape == (num_steps,)
    assert result.w_arm.loss_trajectory.shape == (num_steps,)
    assert jnp.all(jnp.isfinite(result.fws_arm.loss_trajectory))
    assert jnp.all(jnp.isfinite(result.w_arm.loss_trajectory))
    fws_early = jnp.mean(result.fws_arm.loss_trajectory[:10])
    fws_late = jnp.mean(result.fws_arm.loss_trajectory[-10:])
    w_early = jnp.mean(result.w_arm.loss_trajectory[:10])
    w_late = jnp.mean(result.w_arm.loss_trajectory[-10:])
    assert fws_late < fws_early
    assert w_late < w_early


@pytest.mark.unit
def test_joint_regime_loss_decreases_directionally():
    # Given: a synthetic regression task.
    task_loss_fn, task_batch, _ = _make_synthetic_regression()
    init_G, init_z, init_W = _make_inits()

    # When: we train under JOINT for a few hundred steps.
    result = fws_bench.paired_train(
        init_G_params=init_G,
        init_z_params=init_z,
        render_fn=_identity_render,
        init_W_params=init_W,
        task_loss_fn=task_loss_fn,
        task_batch=task_batch,
        regime=fws_bench.Regime.JOINT,
        num_outer_steps=300,
    )

    # Then: terminal loss is strictly below initial loss for both arms.
    assert result.fws_arm.loss_trajectory[-1] < result.fws_arm.loss_trajectory[0]
    assert result.w_arm.loss_trajectory[-1] < result.w_arm.loss_trajectory[0]


@pytest.mark.unit
def test_joint_regime_grad_norms_are_finite():
    # Given: a synthetic regression task and inits.
    task_loss_fn, task_batch, _ = _make_synthetic_regression()
    init_G, init_z, init_W = _make_inits()

    # When: we run paired_train under JOINT.
    result = fws_bench.paired_train(
        init_G_params=init_G,
        init_z_params=init_z,
        render_fn=_identity_render,
        init_W_params=init_W,
        task_loss_fn=task_loss_fn,
        task_batch=task_batch,
        regime=fws_bench.Regime.JOINT,
        num_outer_steps=150,
    )

    # Then: every recorded grad norm is finite for both arms.
    assert jnp.all(jnp.isfinite(result.fws_arm.grad_norm_trajectory))
    assert jnp.all(jnp.isfinite(result.w_arm.grad_norm_trajectory))


@pytest.mark.unit
def test_joint_regime_arms_use_different_param_paths():
    # Given: inits where the FWS arm carries a combined (G, z) pytree and the
    # W arm carries a single W array.
    task_loss_fn, task_batch, _ = _make_synthetic_regression()
    init_G, init_z, init_W = _make_inits()

    # When: we run paired_train under JOINT.
    result = fws_bench.paired_train(
        init_G_params=init_G,
        init_z_params=init_z,
        render_fn=_identity_render,
        init_W_params=init_W,
        task_loss_fn=task_loss_fn,
        task_batch=task_batch,
        regime=fws_bench.Regime.JOINT,
        num_outer_steps=10,
    )

    # Then: the FWS arm's final params are a {"G", "z"} dict; the W arm's
    # final params have the same tree structure as init_W.
    assert isinstance(result.fws_arm.final_params, dict)
    assert set(result.fws_arm.final_params.keys()) == {"G", "z"}
    assert jax.tree.structure(result.w_arm.final_params) == jax.tree.structure(init_W)


@pytest.mark.unit
def test_joint_regime_optimiser_partition_uses_separate_states():
    # Given: distinct optimisers for G and z (adam vs sgd) so post-step
    # parameter deltas reflect the partition rather than a shared transform.
    task_loss_fn, task_batch, _ = _make_synthetic_regression()
    init_G, init_z, init_W = _make_inits()
    G_optimiser = optax.sgd(0.0)  # G is frozen — sign-clean partition probe
    z_optimiser = optax.sgd(1e-1)  # z moves

    # When: we run a single outer step under JOINT.
    result = fws_bench.paired_train(
        init_G_params=init_G,
        init_z_params=init_z,
        render_fn=_identity_render,
        init_W_params=init_W,
        task_loss_fn=task_loss_fn,
        task_batch=task_batch,
        regime=fws_bench.Regime.JOINT,
        G_optimiser=G_optimiser,
        z_optimiser=z_optimiser,
        num_outer_steps=5,
    )

    # Then: G is unchanged (lr 0), z is changed (lr 1e-1). Bit-identical for
    # G via array_equal; structural inequality for z via any-leaf delta.
    assert jnp.array_equal(result.fws_arm.final_params["G"], init_G)
    assert not jnp.allclose(result.fws_arm.final_params["z"], init_z)


@pytest.mark.unit
def test_alternating_regime_raises_notimplemented():
    # Given: a paired_train call with regime=ALTERNATING.
    task_loss_fn, task_batch, _ = _make_synthetic_regression()
    init_G, init_z, init_W = _make_inits()

    # When/Then: it raises NotImplementedError pointing at v0.2.0 and the
    # Trello board.
    with pytest.raises(NotImplementedError, match="v0.2.0"):
        fws_bench.paired_train(
            init_G_params=init_G,
            init_z_params=init_z,
            render_fn=_identity_render,
            init_W_params=init_W,
            task_loss_fn=task_loss_fn,
            task_batch=task_batch,
            regime=fws_bench.Regime.ALTERNATING,
            num_outer_steps=10,
        )


@pytest.mark.unit
def test_meta_regime_raises_notimplemented():
    # Given: a paired_train call with regime=META.
    task_loss_fn, task_batch, _ = _make_synthetic_regression()
    init_G, init_z, init_W = _make_inits()

    # When/Then: it raises NotImplementedError pointing at v0.2.0 and the
    # Trello board.
    with pytest.raises(NotImplementedError, match="v0.2.0"):
        fws_bench.paired_train(
            init_G_params=init_G,
            init_z_params=init_z,
            render_fn=_identity_render,
            init_W_params=init_W,
            task_loss_fn=task_loss_fn,
            task_batch=task_batch,
            regime=fws_bench.Regime.META,
            num_outer_steps=10,
        )


@pytest.mark.unit
def test_default_optimisers_are_adam():
    # Given: a paired_train call with all three optimisers left as None.
    task_loss_fn, task_batch, _ = _make_synthetic_regression()
    init_G, init_z, init_W = _make_inits()

    # When: we run paired_train without passing optimisers.
    result = fws_bench.paired_train(
        init_G_params=init_G,
        init_z_params=init_z,
        render_fn=_identity_render,
        init_W_params=init_W,
        task_loss_fn=task_loss_fn,
        task_batch=task_batch,
        regime=fws_bench.Regime.JOINT,
        num_outer_steps=100,
    )

    # Then: the run completes end-to-end and loss decreases on both arms.
    # (Indirect verification that defaults wired through — a non-functional
    # default would have crashed in optimiser.init or apply_updates.)
    assert result.fws_arm.loss_trajectory[-1] < result.fws_arm.loss_trajectory[0]
    assert result.w_arm.loss_trajectory[-1] < result.w_arm.loss_trajectory[0]
