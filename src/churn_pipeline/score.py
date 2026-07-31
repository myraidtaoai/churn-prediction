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
        "customer_id": pdf["customer_id"],
        "churn_probability": probability,
        "churn_prediction": (probability >= classification_threshold).astype(int),
        "risk_segment": risk_segment,
        "contract": pdf["contract"],
        "tenure": pdf["tenure"],
        "monthly_charges": pdf["monthly_charges"],
        "annual_revenue_at_risk": pdf["monthly_charges"] * 12 * probability,
        "selected_algorithm": metric_row["selected_algorithm"],
        "model_version": str(champion_version),
        "scored_at": pd.Timestamp.now(tz="UTC"),
    }
)

spark.createDataFrame(scores).write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(table(args.catalog, args.schema, "customer_churn_scores"))
