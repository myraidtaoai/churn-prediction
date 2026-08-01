from pathlib import Path

SCORING_SOURCE = (
    Path(__file__).parents[1] / "src" / "churn_pipeline" / "score.py"
)


def test_batch_scoring_uses_distributed_mlflow_inference():
    source = SCORING_SOURCE.read_text(encoding="utf-8")

    assert "mlflow.pyfunc.spark_udf" in source
    assert ".toPandas(" not in source


def test_batch_scoring_requires_versioned_inference_contract():
    source = SCORING_SOURCE.read_text(encoding="utf-8")

    assert "INFERENCE_CONTRACT_VERSION" in source
    assert "champion_contract != INFERENCE_CONTRACT_VERSION" in source
