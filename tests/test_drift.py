"""Unit tests for drift detection utilities (PSI, KS)."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the monitoring package importable.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src" / "churn_pipeline" / "monitoring"),
)

from drift import ks_statistic, psi, psi_categorical

# ── PSI (numeric) ───────────────────────────────────────────────────


class TestPSI:
    def test_identical_distributions_near_zero(self):
        data = np.random.RandomState(42).normal(0, 1, 1000)
        assert psi(data, data) < 0.01

    def test_shifted_distribution_detects_drift(self):
        rng = np.random.RandomState(42)
        baseline = rng.normal(0, 1, 5000)
        shifted = rng.normal(2, 1, 5000)  # mean shifted by 2 std
        result = psi(baseline, shifted)
        assert result > 0.25, f"PSI {result} should indicate significant drift"

    def test_moderate_shift(self):
        rng = np.random.RandomState(42)
        baseline = rng.normal(0, 1, 5000)
        shifted = rng.normal(0.5, 1, 5000)
        result = psi(baseline, shifted)
        assert 0.01 < result < 1.0

    def test_empty_arrays_return_zero(self):
        assert psi(np.array([]), np.array([1, 2, 3])) == 0.0
        assert psi(np.array([1, 2, 3]), np.array([])) == 0.0

    def test_nan_values_ignored(self):
        baseline = np.array([1.0, 2.0, 3.0, np.nan, 4.0, 5.0])
        current = np.array([1.5, 2.5, 3.5, np.nan, 4.5, 5.5])
        result = psi(baseline, current)
        assert result >= 0.0

    def test_constant_feature_returns_zero(self):
        baseline = np.ones(100)
        current = np.ones(100)
        assert psi(baseline, current) == 0.0

    def test_psi_is_non_negative(self):
        rng = np.random.RandomState(99)
        for _ in range(20):
            b = rng.exponential(1.0, 500)
            c = rng.exponential(1.5, 500)
            assert psi(b, c) >= 0.0


# ── PSI (categorical) ──────────────────────────────────────────────


class TestPSICategorical:
    def test_identical_categories_near_zero(self):
        data = np.array(["a", "b", "c", "a", "b", "c"] * 100)
        assert psi_categorical(data, data) < 0.01

    def test_shifted_categories_detects_drift(self):
        baseline = np.array(["a"] * 500 + ["b"] * 500)
        current = np.array(["a"] * 100 + ["b"] * 900)
        result = psi_categorical(baseline, current)
        assert result > 0.10

    def test_new_category_in_current(self):
        baseline = np.array(["a", "b"] * 100)
        current = np.array(["a", "b", "c"] * 100)
        result = psi_categorical(baseline, current)
        assert result > 0.0

    def test_empty_arrays_return_zero(self):
        assert psi_categorical(np.array([]), np.array(["a"])) == 0.0


# ── KS statistic ───────────────────────────────────────────────────


class TestKSStatistic:
    def test_identical_distributions_near_zero(self):
        data = np.random.RandomState(42).normal(0, 1, 1000)
        assert ks_statistic(data, data) < 0.01

    def test_completely_separated_distributions(self):
        baseline = np.arange(0, 100, dtype=float)
        current = np.arange(200, 300, dtype=float)
        assert ks_statistic(baseline, current) == pytest.approx(1.0)

    def test_shifted_distribution(self):
        rng = np.random.RandomState(42)
        baseline = rng.normal(0, 1, 5000)
        shifted = rng.normal(1, 1, 5000)
        result = ks_statistic(baseline, shifted)
        assert 0.1 < result < 0.9

    def test_empty_arrays_return_zero(self):
        assert ks_statistic(np.array([]), np.array([1, 2, 3])) == 0.0

    def test_ks_bounded_zero_to_one(self):
        rng = np.random.RandomState(7)
        for _ in range(20):
            b = rng.uniform(0, 10, 200)
            c = rng.uniform(5, 15, 200)
            result = ks_statistic(b, c)
            assert 0.0 <= result <= 1.0
