"""Unit tests for model selection and production risk segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from model_utils import assign_risk_segments, choose_threshold


def test_choose_threshold_maximizes_validation_f1() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.40, 0.35, 0.80])

    threshold, best_f1 = choose_threshold(labels, probabilities)

    assert threshold == pytest.approx(0.35)
    assert best_f1 == pytest.approx(0.8)


def test_risk_segment_boundaries_match_production_scoring() -> None:
    segments = assign_risk_segments(
        np.array([0.0, 0.2999, 0.30, 0.4999, 0.50, 1.0]),
        classification_threshold=0.50,
    )

    assert segments.tolist() == ["Low", "Low", "Medium", "Medium", "High", "High"]


@pytest.mark.parametrize("threshold", [-0.1, 0.0, 1.1, np.nan])
def test_risk_segmentation_rejects_invalid_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="classification_threshold"):
        assign_risk_segments([0.5], threshold)


@pytest.mark.parametrize("probabilities", [[-0.1], [1.1], [np.nan], [np.inf]])
def test_risk_segmentation_rejects_invalid_probabilities(probabilities) -> None:
    with pytest.raises(ValueError, match="probability values"):
        assign_risk_segments(probabilities, 0.5)
