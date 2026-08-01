"""Generate dashboard-ready churn-risk scores using the Champion model."""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd
from common import base_parser, get_spark, table
from mlflow.exceptions import MlflowException
from model_utils import assign_risk_segments
from pyspark.sql import functions as F

spark = get_spark()
parser = base_parser("Score all customers with the registered Champion model.")
parser.add_argument("--model-name", required=True)
parser.add_argument("--scoring-run-id", required=True)
args = parser.parse_args()
pdf = spark.table(table(args.catalog, args.schema, "telco_silver")).toPandas()

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

model = mlflow.sklearn.load_model(
    f"models:/{args.model_name}/{champion_version}"
)

feature_columns = list(model.feature_names_in_)
probability = model.predict_proba(pdf[feature_columns])[:, 1]

metric_row = (
    spark.table(
        table(args.catalog, args.schema, "model_validation_metrics")
    )
    .where(F.col("model_version") == str(champion_version))
    .first()
)
if metric_row is None:
    raise RuntimeError(
        f"Champion alias points to version {champion_version}, "
        "but matching Champion metrics were not found. "
        "Scoring was stopped to prevent using the wrong threshold."
    )
classification_threshold = float(metric_row["classification_threshold"])
risk_segment = assign_risk_segments(probability, classification_threshold)

scores = pd.DataFrame(
    {
        "scoring_run_id": args.scoring_run_id,
        "customer_id": pdf["customer_id"],
        "churn_probability": probability,
        "churn_prediction": (
            probability >= classification_threshold
        ).astype(int),
        "classification_threshold": classification_threshold,
        "risk_segment": risk_segment,
        "contract": pdf["contract"],
        "tenure": pdf["tenure"],
        "monthly_charges": pdf["monthly_charges"],
        "annual_revenue_at_risk": (
            pdf["monthly_charges"] * 12 * probability
        ),
        "selected_algorithm": metric_row["selected_algorithm"],
        "model_name": args.model_name,
        "model_version": champion_version,
        "model_run_id": str(champion_model_version.run_id or ""),
        "scored_at": pd.Timestamp.now(tz="UTC"),
    }
)

scores_df = spark.createDataFrame(scores)
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
