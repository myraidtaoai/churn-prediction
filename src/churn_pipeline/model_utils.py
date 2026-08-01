"""Pure, import-safe utilities shared by model training and scoring."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def choose_threshold(y_true, probability) -> tuple[float, float]:
    """Return the probability threshold that maximizes positive-class F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = (
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def assign_risk_segments(probability, classification_threshold: float) -> np.ndarray:
    """Assign High, Medium, or Low using the production scoring boundaries."""
    if (
        not np.isfinite(classification_threshold)
        or not 0 < classification_threshold <= 1
    ):
        raise ValueError("classification_threshold must be in the interval (0, 1].")
    values = np.asarray(probability, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("probability values must be finite and between 0 and 1.")
    return np.select(
        [
            values >= classification_threshold,
            values >= classification_threshold * 0.6,
        ],
        ["High", "Medium"],
        default="Low",
    )
