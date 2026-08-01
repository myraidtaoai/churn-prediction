import numpy as np
import pandas as pd
from inference import ChurnProbabilityModel
from mlflow.models import infer_signature


class FixedProbabilityEstimator:
    def predict_proba(self, model_input):
        probability = model_input["probability"].to_numpy(dtype=float)
        return np.column_stack([1.0 - probability, probability])


def test_probability_model_returns_probability_and_prediction():
    model = ChurnProbabilityModel(
        estimator=FixedProbabilityEstimator(),
        classification_threshold=0.60,
    )

    result = model.predict(
        context=None,
        model_input=pd.DataFrame({"probability": [0.20, 0.60, 0.90]}),
    )

    assert result.columns.tolist() == [
        "churn_probability",
        "churn_prediction",
    ]
    assert result["churn_probability"].tolist() == [0.20, 0.60, 0.90]
    assert result["churn_prediction"].tolist() == [0, 1, 1]


def test_probability_model_uses_configured_threshold():
    model = ChurnProbabilityModel(
        estimator=FixedProbabilityEstimator(),
        classification_threshold=0.75,
    )

    result = model.predict(
        context=None,
        model_input=pd.DataFrame({"probability": [0.70, 0.80]}),
    )

    assert result["churn_prediction"].tolist() == [0, 1]


def test_probability_model_signature_preserves_named_contract():
    model = ChurnProbabilityModel(
        estimator=FixedProbabilityEstimator(),
        classification_threshold=0.60,
    )
    model_input = pd.DataFrame({"probability": [0.20, 0.90]})

    signature = infer_signature(
        model_input,
        model.predict(context=None, model_input=model_input),
    )

    assert signature.inputs.input_names() == ["probability"]
    assert signature.outputs.input_names() == [
        "churn_probability",
        "churn_prediction",
    ]
