"""Generate dashboard-ready churn-risk scores using the Champion model."""

from __future__ import annotations

import _path_helper  # noqa: F401 — adds churn_pipeline/ to sys.path

from datetime import datetime, timezone

import mlflow
import mlflow.models
import mlflow.pyfunc
import json

from common import base_parser, get_spark, table
from inference import INFERENCE_CONTRACT_TAG, INFERENCE_CONTRACT_VERSION
from ops.run_logger import log_run
from mlflow.exceptions import MlflowException
from mlflow_compat import create_spark_udf_with_runtime_compat
from pyspark.sql import functions as F

spark = get_spark()
_run_started = datetime.now(timezone.utc)
parser = base_parser("Score all customers with the registered Champion model.")
parser.add_argument("--model-name", required=True)
parser.add_argument("--scoring-run-id", required=True)
args = parser.parse_args()
source_df = spark.table(table(args.catalog, args.schema, "telco_silver"))

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

try:
    champion_model_version = client.get_model_version_by_alias(
        args.model_name,
        "Champion",
    )
except MlflowException as exc:
    raise RuntimeError(
        "The registered model does not have a Champion alias. "
        "Promote an approved Candidate before scoring."
    ) from exc

champion_version = str(champion_model_version.version)
champion_contract = (champion_model_version.tags or {}).get(
    INFERENCE_CONTRACT_TAG,
    "",
)
if champion_contract != INFERENCE_CONTRACT_VERSION:
    raise RuntimeError(
        f"Champion version {champion_version} uses the legacy inference "
        "contract. Run the model pipeline to train and promote a distributed-"
        "inference-compatible Candidate before scoring."
    )

model_uri = f"models:/{args.model_name}/{champion_version}"
model_info = mlflow.models.get_model_info(model_uri)
signature = model_info.signature
if signature is None or signature.inputs is None or signature.outputs is None:
    raise RuntimeError(
        f"Champion version {champion_version} does not have a complete model signature."
    )

feature_columns = signature.inputs.input_names()
output_columns = set(signature.outputs.input_names())
required_outputs = {"churn_probability", "churn_prediction"}
if not required_outputs.issubset(output_columns):
    raise RuntimeError(
        f"Champion version {champion_version} does not provide the required "
        f"outputs: {sorted(required_outputs)}."
    )

missing_features = sorted(set(feature_columns) - set(source_df.columns))
if missing_features:
    raise RuntimeError(
        "The scoring dataset is missing model features: " + ", ".join(missing_features)
    )

metric_row = (
    spark.table(table(args.catalog, args.schema, "model_candidate_metrics"))
    .where(F.col("model_version") == champion_version)
    .orderBy(F.col("trained_at").desc())
    .first()
)
if metric_row is None:
    raise RuntimeError(
        f"Champion alias points to version {champion_version}, "
        "but matching Champion metrics were not found. "
        "Scoring was stopped to prevent using the wrong threshold."
    )
classification_threshold = float(metric_row["classification_threshold"])
udf_options = {
    "result_type": ("struct<churn_probability:double,churn_prediction:long>"),
    "env_manager": "local",
}
prediction_udf = create_spark_udf_with_runtime_compat(
    spark,
    model_uri,
    **udf_options,
)
prediction_input = F.struct(
    *(F.col(column).alias(column) for column in feature_columns)
)
predicted_df = source_df.withColumn(
    "_prediction",
    prediction_udf(prediction_input),
)
probability = F.col("_prediction.churn_probability")
risk_segment = (
    F.when(probability >= classification_threshold, F.lit("High"))
    .when(
        probability >= classification_threshold * 0.6,
        F.lit("Medium"),
    )
    .otherwise(F.lit("Low"))
)
scored_at = datetime.now(timezone.utc)

scores_df = predicted_df.select(
    F.lit(args.scoring_run_id).alias("scoring_run_id"),
    F.col("customer_id"),
    probability.alias("churn_probability"),
    F.col("_prediction.churn_prediction").cast("int").alias("churn_prediction"),
    F.lit(classification_threshold).alias("classification_threshold"),
    risk_segment.alias("risk_segment"),
    F.col("contract"),
    F.col("tenure"),
    F.col("monthly_charges"),
    (F.col("monthly_charges") * 12 * probability).alias("annual_revenue_at_risk"),
    F.lit(metric_row["selected_algorithm"]).alias("selected_algorithm"),
    F.lit(args.model_name).alias("model_name"),
    F.lit(champion_version).alias("model_version"),
    F.lit(str(champion_model_version.run_id or "")).alias("model_run_id"),
    F.lit(scored_at).alias("scored_at"),
)
history_table = table(
    args.catalog,
    args.schema,
    "prediction_history",
)

scores_df.createOrReplaceTempView("new_churn_predictions")

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {history_table}
    USING DELTA
    AS
    SELECT *
    FROM new_churn_predictions
    WHERE 1 = 0
    """
)

spark.sql(
    f"""
    MERGE INTO {history_table} AS target
    USING new_churn_predictions AS source
      ON target.scoring_run_id = source.scoring_run_id
     AND target.customer_id = source.customer_id
    WHEN NOT MATCHED THEN INSERT *
    """
)

current_scores_view = table(
    args.catalog,
    args.schema,
    "current_customer_churn_scores",
)

spark.sql(
    f"""
    CREATE OR REPLACE VIEW {current_scores_view} AS
    WITH scoring_runs AS (
        SELECT
            scoring_run_id,
            MAX(scored_at) AS latest_scored_at
        FROM {history_table}
        GROUP BY scoring_run_id
    ),
    latest_run AS (
        SELECT scoring_run_id
        FROM scoring_runs
        ORDER BY latest_scored_at DESC, scoring_run_id DESC
        LIMIT 1
    )
    SELECT history.*
    FROM {history_table} AS history
    INNER JOIN latest_run
      ON history.scoring_run_id = latest_run.scoring_run_id
    """
)

_scored_count = spark.table(current_scores_view).count()
_summary = {
    "status": "completed",
    "scoring_run_id": args.scoring_run_id,
    "model_version": champion_version,
    "customers_scored": _scored_count,
}
print(json.dumps(_summary, sort_keys=True))

log_run(
    spark=spark,
    catalog=args.catalog,
    schema=args.schema,
    task_name="batch_score",
    run_id=args.scoring_run_id,
    status="succeeded",
    started_at=_run_started,
    finished_at=datetime.now(timezone.utc),
    output_summary=_summary,
)
