"""MLflow inference contract for churn probability scoring."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

INFERENCE_CONTRACT_TAG = "inference_contract"
INFERENCE_CONTRACT_VERSION = "churn_probability_v1"


class ChurnProbabilityModel(mlflow.pyfunc.PythonModel):
    """Return churn probabilities and thresholded predictions."""

    def __init__(self, estimator: Any, classification_threshold: float):
        self.estimator = estimator
        self.classification_threshold = float(classification_threshold)

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Score a batch using the wrapped probabilistic classifier."""
        probability = np.asarray(
            self.estimator.predict_proba(model_input)
        )[:, 1].astype(float)
        prediction = (
            probability >= self.classification_threshold
        ).astype("int64")

        return pd.DataFrame(
            {
                "churn_probability": probability,
                "churn_prediction": prediction,
            }
        )
