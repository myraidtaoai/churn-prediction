"""Promote an approved Candidate model to Champion."""

from __future__ import annotations

from datetime import datetime, timezone

import mlflow
from common import base_parser, get_spark, table
from inference import INFERENCE_CONTRACT_TAG, INFERENCE_CONTRACT_VERSION
from mlflow.exceptions import MlflowException
from promotion_policy import ModelMetrics, evaluate_promotion
from pyspark.sql import functions as F

spark = get_spark()

parser = base_parser("Evaluate and promote the Candidate model.")
parser.add_argument("--model-name", required=True)
args = parser.parse_args()

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

try:
    candidate_model_version = client.get_model_version_by_alias(
        args.model_name,
        "Candidate",
    )
except MlflowException as exc:
    raise RuntimeError(
        "The registered model does not have a Candidate alias. "
        "Run candidate training before promotion."
    ) from exc

candidate_version = str(candidate_model_version.version)

try:
    champion_model_version = client.get_model_version_by_alias(
        args.model_name,
        "Champion",
    )
    champion_version = str(champion_model_version.version)
except MlflowException as exc:
    if exc.error_code not in {"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"}:
        raise
    champion_model_version = None
    champion_version = None

candidate_contract = (candidate_model_version.tags or {}).get(
    INFERENCE_CONTRACT_TAG,
    "",
)
champion_contract = (
    (champion_model_version.tags or {}).get(INFERENCE_CONTRACT_TAG, "")
    if champion_model_version is not None
    else ""
)
contract_upgrade = (
    candidate_contract == INFERENCE_CONTRACT_VERSION
    and champion_contract != INFERENCE_CONTRACT_VERSION
)

candidate_row = (
    spark.table(
        table(args.catalog, args.schema, "model_candidate_metrics")
    )
    .where(F.col("model_version") == str(candidate_version))
    .orderBy(F.col("trained_at").desc())
    .first()
)

if candidate_row is None:
    raise RuntimeError(
        f"Metrics were not found for Candidate version {candidate_version}."
    )

candidate_metrics = ModelMetrics(
    pr_auc=float(candidate_row["test_pr_auc"]),
    recall=float(candidate_row["test_recall"]),
)

champion_metrics = None

if champion_version is not None:
    champion_row = (
        spark.table(
            table(args.catalog, args.schema, "model_validation_metrics")
        )
        .where(F.col("model_version") == str(champion_version))
        .orderBy(F.col("trained_at").desc())
        .first()
    )

    if champion_row is None:
        raise RuntimeError(
            f"Champion alias points to version {champion_version}, "
            "but matching Champion metrics were not found."
        )

    champion_metrics = ModelMetrics(
        pr_auc=float(champion_row["test_pr_auc"]),
        recall=float(champion_row["test_recall"]),
    )

decision = evaluate_promotion(
    candidate=candidate_metrics,
    champion=champion_metrics,
    allow_equivalent_contract_upgrade=contract_upgrade,
)

evaluated_at = datetime.now(timezone.utc)

if decision.approved:
    client.set_registered_model_alias(
        args.model_name,
        "Champion",
        str(candidate_version),
    )

    current_champion = candidate_row.asDict(recursive=True)
    current_champion["model_alias"] = "Champion"
    current_champion["promoted_at"] = evaluated_at

    (
        spark.createDataFrame([current_champion])
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(
            table(args.catalog, args.schema, "model_validation_metrics")
        )
    )

audit_row = {
    "model_name": args.model_name,
    "candidate_version": str(candidate_version),
    "previous_champion_version": str(champion_version or ""),
    "candidate_inference_contract": candidate_contract,
    "previous_champion_inference_contract": champion_contract,
    "contract_upgrade": contract_upgrade,
    "candidate_pr_auc": candidate_metrics.pr_auc,
    "candidate_recall": candidate_metrics.recall,
    "champion_pr_auc": (
        champion_metrics.pr_auc if champion_metrics is not None else None
    ),
    "champion_recall": (
        champion_metrics.recall if champion_metrics is not None else None
    ),
    "decision": "APPROVED" if decision.approved else "REJECTED",
    "decision_reason": decision.reason,
    "evaluated_at": evaluated_at,
}

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
    .saveAsTable(
        table(args.catalog, args.schema, "model_promotion_history")
    )
)

print(
    f"Promotion decision for Candidate version {candidate_version}: "
    f"{audit_row['decision']} — {decision.reason}"
)
