"""Smoke test for the v0.0.0 scaffold public surface."""

from __future__ import annotations

import pytest

import fws_bench


pytestmark = pytest.mark.unit


def test_package_exposes_regime_and_paired_train():
    # Given: the public surface declared in fws_bench/__init__.py
    # When: we inspect the module's exported names
    # Then: both Regime and paired_train are exposed
    assert hasattr(fws_bench, "Regime")
    assert hasattr(fws_bench, "paired_train")
    assert "Regime" in fws_bench.__all__
    assert "paired_train" in fws_bench.__all__


def test_regime_enumerates_joint_alternating_meta():
    # Given: the Regime enum
    # When: we read out its members
    # Then: exactly the three regimes from docs/PHILOSOPHY.md are present
    values = {member.value for member in fws_bench.Regime}
    assert values == {"joint", "alternating", "meta"}


def test_paired_train_raises_not_implemented():
    # Given: the v0.0.0 scaffold of paired_train
    # When: we call it with any arguments
    # Then: it raises NotImplementedError pointing at the Trello prototype card
    with pytest.raises(NotImplementedError, match="scaffold"):
        fws_bench.paired_train(
            render_fn=lambda params: params,
            task_loss_fn=lambda params, batch: 0.0,
        )
