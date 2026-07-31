"""Generate dashboard-ready churn-risk scores using the Champion model."""

from __future__ import annotations

import mlflow.sklearn
import pandas as pd
from common import base_parser, get_spark, table
from model_utils import assign_risk_segments

spark = get_spark()
parser = base_parser("Score all customers with the registered Champion model.")
parser.add_argument("--model-name", required=True)
args = parser.parse_args()
pdf = spark.table(table(args.catalog, args.schema, "telco_silver")).toPandas()
mlflow.set_registry_uri("databricks-uc")
model = mlflow.sklearn.load_model(f"models:/{args.model_name}@Champion")
feature_columns = list(model.feature_names_in_)
probability = model.predict_proba(pdf[feature_columns])[:, 1]
metric_row = spark.table(table(args.catalog, args.schema, "model_validation_metrics")).first()
if metric_row is None:
    raise RuntimeError("model_validation_metrics is empty; run training before scoring.")
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
        "model_version": metric_row["model_version"],
        "scored_at": pd.Timestamp.now(tz="UTC"),
    }
)

spark.createDataFrame(scores).write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(table(args.catalog, args.schema, "customer_churn_scores"))
