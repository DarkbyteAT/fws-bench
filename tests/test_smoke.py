"""Smoke test for the public surface."""

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


def test_package_exposes_train_result_value_types():
    # Given: the public surface declared in fws_bench/__init__.py
    # When: we inspect the module's exported names
    # Then: TrainResult and PairedTrainResult are exposed for downstream
    # callers that want to type-annotate trajectories
    assert hasattr(fws_bench, "TrainResult")
    assert hasattr(fws_bench, "PairedTrainResult")
    assert "TrainResult" in fws_bench.__all__
    assert "PairedTrainResult" in fws_bench.__all__
