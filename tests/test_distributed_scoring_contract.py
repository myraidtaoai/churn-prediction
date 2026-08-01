from pathlib import Path

SCORING_SOURCE = (
    Path(__file__).parents[1] / "src" / "churn_pipeline" / "score.py"
)
TRAINING_SOURCE = (
    Path(__file__).parents[1] / "src" / "churn_pipeline" / "train.py"
)


def test_batch_scoring_uses_distributed_mlflow_inference():
    source = SCORING_SOURCE.read_text(encoding="utf-8")

    assert "create_spark_udf_with_runtime_compat" in source
    assert ".toPandas(" not in source


def test_batch_scoring_requires_versioned_inference_contract():
    source = SCORING_SOURCE.read_text(encoding="utf-8")

    assert "INFERENCE_CONTRACT_VERSION" in source
    assert "champion_contract != INFERENCE_CONTRACT_VERSION" in source


def test_training_resolves_model_code_without_entrypoint_file_global():
    source = TRAINING_SOURCE.read_text(encoding="utf-8")

    assert "Path(__file__)" not in source
    assert "inspect.getfile(ChurnProbabilityModel)" in source
