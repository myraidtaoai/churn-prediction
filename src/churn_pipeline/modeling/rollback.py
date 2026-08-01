"""Roll the Champion alias back to a validated model version."""

from __future__ import annotations

import _path_helper  # noqa: F401 — adds churn_pipeline/ to sys.path

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import mlflow
from common import base_parser, get_spark, table
from inference import INFERENCE_CONTRACT_TAG, INFERENCE_CONTRACT_VERSION
from mlflow.exceptions import MlflowException

MetricsReader = Callable[[str], Any | None]
AuditWriter = Callable[[dict[str, Any]], None]


class RollbackValidationError(RuntimeError):
    """Raised before mutation when a rollback target is unsafe or incomplete."""


@dataclass(frozen=True)
class RollbackResult:
    previous_version: str
    target_version: str


def _get_model_version(client, model_name: str, version: str):
    try:
        return client.get_model_version(model_name, version)
    except MlflowException as exc:
        raise RollbackValidationError(
            f"Model version {version} does not exist for {model_name}."
        ) from exc


def _get_champion(client, model_name: str):
    try:
        return client.get_model_version_by_alias(model_name, "Champion")
    except MlflowException as exc:
        raise RollbackValidationError(
            "The registered model does not have a Champion alias to roll back."
        ) from exc


def _restore_alias(client, model_name: str, previous_version: str) -> None:
    client.set_registered_model_alias(
        model_name,
        "Champion",
        previous_version,
    )
    restored = client.get_model_version_by_alias(model_name, "Champion")
    if str(restored.version) != previous_version:
        raise RuntimeError(
            "Rollback compensation failed: the previous Champion alias "
            "could not be restored."
        )


def rollback_champion(
    *,
    client,
    model_name: str,
    target_version: str,
    read_metrics: MetricsReader,
    write_audit: AuditWriter,
    evaluated_at: datetime | None = None,
) -> RollbackResult:
    """Validate, move, verify, and audit a Champion rollback."""
    target_version = str(target_version).strip()
    if not target_version:
        raise RollbackValidationError(
            "A target model version is required. Set target_version in the "
            "Databricks Run now parameters or run the bundle with "
            "--params target_version=<model-version>."
        )

    target_model = _get_model_version(client, model_name, target_version)
    target_contract = (target_model.tags or {}).get(INFERENCE_CONTRACT_TAG, "")
    if target_contract != INFERENCE_CONTRACT_VERSION:
        raise RollbackValidationError(
            f"Model version {target_version} uses incompatible inference "
            f"contract {target_contract!r}."
        )

    target_metrics = read_metrics(target_version)
    if target_metrics is None:
        raise RollbackValidationError(
            f"Metrics were not found for rollback target version {target_version}."
        )

    current_champion = _get_champion(client, model_name)
    previous_version = str(current_champion.version)
    if previous_version == target_version:
        raise RollbackValidationError(
            f"Champion already points to model version {target_version}."
        )

    previous_contract = (current_champion.tags or {}).get(
        INFERENCE_CONTRACT_TAG,
        "",
    )
    previous_metrics = read_metrics(previous_version)

    client.set_registered_model_alias(
        model_name,
        "Champion",
        target_version,
    )

    try:
        confirmed = client.get_model_version_by_alias(model_name, "Champion")
        if str(confirmed.version) != target_version:
            raise RuntimeError("Champion rollback verification failed.")

        write_audit(
            {
                "model_name": model_name,
                "candidate_version": target_version,
                "previous_champion_version": previous_version,
                "candidate_inference_contract": target_contract,
                "previous_champion_inference_contract": previous_contract,
                "contract_upgrade": False,
                "candidate_pr_auc": float(target_metrics["test_pr_auc"]),
                "candidate_recall": float(target_metrics["test_recall"]),
                "champion_pr_auc": (
                    float(previous_metrics["test_pr_auc"])
                    if previous_metrics is not None
                    else None
                ),
                "champion_recall": (
                    float(previous_metrics["test_recall"])
                    if previous_metrics is not None
                    else None
                ),
                "decision": "ROLLBACK",
                "decision_reason": (
                    f"Champion alias rolled back from version {previous_version} "
                    f"to version {target_version}."
                ),
                "evaluated_at": evaluated_at or datetime.now(timezone.utc),
            }
        )
    except Exception as exc:
        _restore_alias(client, model_name, previous_version)
        raise RuntimeError(
            "Rollback did not complete; the previous Champion alias was restored."
        ) from exc

    return RollbackResult(
        previous_version=previous_version,
        target_version=target_version,
    )


def read_latest_metrics(spark, catalog: str, schema: str, version: str):
    """Read immutable metrics for one registered model version."""
    from pyspark.sql import functions as F

    return (
        spark.table(table(catalog, schema, "model_candidate_metrics"))
        .where(F.col("model_version") == str(version))
        .orderBy(F.col("trained_at").desc())
        .first()
    )


def write_rollback_audit(spark, catalog: str, schema: str, audit_row) -> None:
    """Append the verified rollback to the shared model audit history."""
    audit_schema = """
        model_name STRING,
        candidate_version STRING,
        previous_champion_version STRING,
        candidate_inference_contract STRING,
        previous_champion_inference_contract STRING,
        contract_upgrade BOOLEAN,
        candidate_pr_auc DOUBLE,
        candidate_recall DOUBLE,
        champion_pr_auc DOUBLE,
        champion_recall DOUBLE,
        decision STRING,
        decision_reason STRING,
        evaluated_at TIMESTAMP
    """
    (
        spark.createDataFrame([audit_row], schema=audit_schema)
        .write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(table(catalog, schema, "model_promotion_history"))
    )


def main(argv: list[str] | None = None) -> None:
    """Run the manual Databricks rollback task."""
    parser = base_parser("Rollback to a previous Champion model version.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--target-version", required=True)
    args = parser.parse_args(argv)

    spark = get_spark()
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()

    result = rollback_champion(
        client=client,
        model_name=args.model_name,
        target_version=args.target_version,
        read_metrics=lambda version: read_latest_metrics(
            spark,
            args.catalog,
            args.schema,
            version,
        ),
        write_audit=lambda row: write_rollback_audit(
            spark,
            args.catalog,
            args.schema,
            row,
        ),
    )
    print(
        f"Champion rolled back from version {result.previous_version} "
        f"to version {result.target_version}."
    )


if __name__ == "__main__":
    main()
