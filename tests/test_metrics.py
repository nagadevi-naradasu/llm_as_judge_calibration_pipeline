"""Unit tests for the metrics engine (bootstrap CIs and invariants)."""

import numpy as np
import pytest

from src.metrics import (
    agreement,
    bootstrap_metric,
    compute_metrics,
    tnr,
    tpr,
)


def test_tpr_tnr_definitions_fail_is_positive():
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    # fails (0,0,0): caught = positions 0 and 4 -> 2/3
    assert tpr(y_true, y_pred) == pytest.approx(2 / 3)
    # passes (1,1): kept both -> 2/2
    assert tnr(y_true, y_pred) == pytest.approx(1.0)
    assert agreement(y_true, y_pred) == pytest.approx(4 / 5)


def test_bootstrap_ci_ordering_and_bounds():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = np.where(rng.random(200) < 0.9, y_true, 1 - y_true)
    mean, lo, hi = bootstrap_metric(y_true, y_pred, tpr, n_iterations=500, seed=1)
    assert lo <= mean <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_is_deterministic():
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 0])
    a = bootstrap_metric(y_true, y_pred, tpr, n_iterations=300, seed=7)
    b = bootstrap_metric(y_true, y_pred, tpr, n_iterations=300, seed=7)
    assert a == b


def test_perfect_predictions():
    y = np.array([0, 1] * 50)
    m = compute_metrics(y, y, n_iterations=200, seed=3)
    assert m["tpr_mean"] == pytest.approx(1.0)
    assert m["tnr_mean"] == pytest.approx(1.0)
    assert m["agreement"] == pytest.approx(1.0)
    assert m["kappa_mean"] == pytest.approx(1.0)


def test_always_pass_masks_imbalance():
    # 90% pass, judge always says pass: high agreement, zero error detection.
    y = np.array([1] * 90 + [0] * 10)
    p = np.ones_like(y)
    m = compute_metrics(y, p, n_iterations=200, seed=5)
    assert m["agreement"] == pytest.approx(0.9)
    assert m["tpr_mean"] == pytest.approx(0.0)
    assert m["agreement"] > min(m["tpr_mean"], m["tnr_mean"])


def test_ci_invariants_in_compute_metrics():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=120)
    p = np.where(rng.random(120) < 0.8, y, 1 - y)
    m = compute_metrics(y, p, n_iterations=400, seed=11)
    for key in ("tpr", "tnr"):
        assert m[f"{key}_ci_low"] <= m[f"{key}_mean"] <= m[f"{key}_ci_high"]
    assert m["kappa_ci_low"] <= m["kappa_mean"] <= m["kappa_ci_high"]
