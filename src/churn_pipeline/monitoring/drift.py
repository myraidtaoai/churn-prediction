"""Drift detection utilities: PSI and KS statistics.

All functions operate on plain Python / numpy arrays so they can be
unit-tested without Spark.
"""

from __future__ import annotations

import numpy as np

# Small constant to avoid log(0) in PSI.
_EPS = 1e-6


def psi(
    baseline: np.ndarray,
    current: np.ndarray,
    buckets: int = 10,
) -> float:
    """Population Stability Index for a single numeric feature.

    Both arrays are binned into ``buckets`` quantile-based bins derived
    from the *baseline* distribution.  PSI measures how much the current
    distribution has shifted:

    - PSI < 0.10  → no significant shift
    - 0.10–0.25   → moderate shift
    - PSI > 0.25  → significant shift
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    # Quantile boundaries from baseline.
    edges = np.unique(np.percentile(baseline, np.linspace(0, 100, buckets + 1)))
    if len(edges) < 2:
        return 0.0

    base_counts = np.histogram(baseline, bins=edges)[0].astype(float)
    curr_counts = np.histogram(current, bins=edges)[0].astype(float)

    base_pct = base_counts / base_counts.sum() + _EPS
    curr_pct = curr_counts / curr_counts.sum() + _EPS

    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


def psi_categorical(
    baseline: np.ndarray,
    current: np.ndarray,
) -> float:
    """PSI for a categorical feature using category proportions."""
    baseline = np.asarray(baseline)
    current = np.asarray(current)

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    categories = set(np.unique(baseline)) | set(np.unique(current))

    base_total = float(len(baseline))
    curr_total = float(len(current))

    psi_value = 0.0
    for cat in categories:
        base_pct = float(np.sum(baseline == cat)) / base_total + _EPS
        curr_pct = float(np.sum(current == cat)) / curr_total + _EPS
        psi_value += (curr_pct - base_pct) * np.log(curr_pct / base_pct)

    return float(psi_value)


def ks_statistic(baseline: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (no scipy dependency).

    Returns the maximum absolute difference between the empirical CDFs.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    combined = np.sort(np.concatenate([baseline, current]))
    cdf_base = np.searchsorted(np.sort(baseline), combined, side="right") / len(
        baseline
    )
    cdf_curr = np.searchsorted(np.sort(current), combined, side="right") / len(current)

    return float(np.max(np.abs(cdf_base - cdf_curr)))
