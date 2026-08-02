"""Compatibility helpers for MLflow on Databricks serverless runtimes."""

from __future__ import annotations

import re
from typing import Any

_UNCUT_DBR_VERSION = re.compile(r"^(?P<major>[0-9]+)\.x(?:[-.].*)?$")


def normalize_uncut_databricks_runtime(version: str) -> str:
    """Convert a DBR ``major.x`` image label to a comparable version."""
    match = _UNCUT_DBR_VERSION.fullmatch(version)
    if match is None:
        raise ValueError(f"Unsupported Databricks runtime version: {version!r}")

    return f"{match.group('major')}.999"


def repair_mlflow_serverless_runtime_cache(
    spark: Any,
) -> None:
    """Repair MLflow's cached DBR version after its strict parser fails.

    MLflow 2.22 passes the serverless image label to ``packaging.Version``.
    New uncut DBR images use labels such as
    ``18.x-aarch64-photon-scala2``, which are not PEP 440 versions.
    """
    from mlflow.utils.databricks_utils import (
        get_dbconnect_udf_sandbox_info,
    )

    sandbox_info = get_dbconnect_udf_sandbox_info(spark)
    sandbox_info.runtime_version = normalize_uncut_databricks_runtime(
        sandbox_info.runtime_version
    )


def create_spark_udf_with_runtime_compat(
    spark: Any,
    model_uri: str,
    **options: Any,
) -> Any:
    """Create an MLflow Spark UDF, retrying the known DBR 18.x failure."""
    import mlflow.pyfunc
    from packaging.version import InvalidVersion

    try:
        return mlflow.pyfunc.spark_udf(spark, model_uri, **options)
    except InvalidVersion:
        repair_mlflow_serverless_runtime_cache(spark)
        return mlflow.pyfunc.spark_udf(spark, model_uri, **options)
