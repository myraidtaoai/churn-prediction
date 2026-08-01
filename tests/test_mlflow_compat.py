from types import SimpleNamespace

import pytest
from mlflow_compat import (
    create_spark_udf_with_runtime_compat,
    normalize_uncut_databricks_runtime,
    repair_mlflow_serverless_runtime_cache,
)
from packaging.version import InvalidVersion


def test_uncut_databricks_runtime_is_normalized_for_mlflow():
    assert (
        normalize_uncut_databricks_runtime(
            "18.x-aarch64-photon-scala2"
        )
        == "18.999"
    )


@pytest.mark.parametrize(
    "version",
    ["18.2", "client.4.1", "not-a-version"],
)
def test_unrecognized_runtime_is_not_silently_rewritten(version):
    with pytest.raises(ValueError, match="Unsupported Databricks runtime"):
        normalize_uncut_databricks_runtime(version)


def test_mlflow_sandbox_cache_is_repaired(monkeypatch):
    sandbox_info = SimpleNamespace(
        runtime_version="18.x-aarch64-photon-scala2"
    )
    monkeypatch.setattr(
        "mlflow.utils.databricks_utils.get_dbconnect_udf_sandbox_info",
        lambda spark: sandbox_info,
    )

    repair_mlflow_serverless_runtime_cache(
        spark=object(),
    )

    assert sandbox_info.runtime_version == "18.999"


def test_spark_udf_creation_retries_after_runtime_repair(monkeypatch):
    sandbox_info = SimpleNamespace(
        runtime_version="18.x-aarch64-photon-scala2"
    )
    calls = []

    def create_udf(spark, model_uri, **options):
        calls.append((spark, model_uri, options))
        if len(calls) == 1:
            raise InvalidVersion(
                "Invalid version: '18.x-aarch64-photon-scala2'"
            )
        return "prediction-udf"

    monkeypatch.setattr(
        "mlflow.utils.databricks_utils.get_dbconnect_udf_sandbox_info",
        lambda spark: sandbox_info,
    )
    monkeypatch.setattr("mlflow.pyfunc.spark_udf", create_udf)

    result = create_spark_udf_with_runtime_compat(
        spark="spark-session",
        model_uri="models:/example/1",
        env_manager="local",
    )

    assert result == "prediction-udf"
    assert len(calls) == 2
    assert sandbox_info.runtime_version == "18.999"
