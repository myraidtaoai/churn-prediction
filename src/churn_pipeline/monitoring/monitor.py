"""Drift monitoring: compare current features and predictions against baseline.

Reads the training dataset (baseline) and the latest gold feature snapshot
plus the latest prediction scores.  Computes PSI and KS statistics per
feature, plus prediction distribution drift.  Writes results to
``drift_metrics`` Delta table.

Usage (standalone Databricks job)::

    python monitor.py --catalog main --schema dev_churn
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import _path_helper  # noqa: F401 — adds churn_pipeline/ to sys.path
import numpy as np
from common import base_parser, get_spark, table
from drift import ks_statistic, psi, psi_categorical
from ops.run_logger import log_run
from pyspark.sql import functions as F

spark = get_spark()

parser = base_parser("Compute feature and prediction drift metrics.")
args = parser.parse_args()

_run_started = datetime.now(timezone.utc)


# ── Resolve baseline and current datasets ───────────────────────────

training_table = table(args.catalog, args.schema, "training_dataset")
gold_table = table(args.catalog, args.schema, "gold_feature_snapshot")
scores_table = table(args.catalog, args.schema, "customer_churn_scores")

baseline_df = spark.table(training_table)
if baseline_df.limit(1).count() == 0:
    summary = {"status": "skipped", "reason": "training_dataset is empty"}
    print(json.dumps(summary, sort_keys=True))
    log_run(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
        task_name="drift_monitor",
        run_id=datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        status="skipped",
        started_at=_run_started,
        finished_at=datetime.now(timezone.utc),
        output_summary=summary,
    )
    raise SystemExit(0)

# Latest gold snapshot.
latest_snapshot_date = (
    spark.table(gold_table)
    .select(F.max("snapshot_date").alias("max_date"))
    .collect()[0]["max_date"]
)

if latest_snapshot_date is None:
    summary = {"status": "skipped", "reason": "no gold snapshots found"}
    print(json.dumps(summary, sort_keys=True))
    log_run(
        spark=spark,
        catalog=args.catalog,
        schema=args.schema,
        task_name="drift_monitor",
        run_id=datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        status="skipped",
        started_at=_run_started,
        finished_at=datetime.now(timezone.utc),
        output_summary=summary,
    )
    raise SystemExit(0)

current_df = spark.table(gold_table).filter(
    F.col("snapshot_date") == latest_snapshot_date
)

# ── Identify feature columns ───────────────────────────────────────

EXCLUDE_COLS = {
    "customer_id",
    "churn",
    "churn_label",
    "snapshot_date",
    "_transformed_at",
    "_built_at",
}

baseline_pdf = baseline_df.toPandas()
current_pdf = current_df.toPandas()

feature_columns = [
    c
    for c in baseline_pdf.columns
    if c not in EXCLUDE_COLS and c in current_pdf.columns
]

categorical_cols = (
    baseline_pdf[feature_columns]
    .select_dtypes(include=["object", "string", "category"])
    .columns.tolist()
)
numeric_cols = [c for c in feature_columns if c not in categorical_cols]


# ── Compute drift per feature ──────────────────────────────────────

run_ts = datetime.now(timezone.utc).isoformat()
metrics_rows: list[dict] = []

for col in numeric_cols:
    base_vals = baseline_pdf[col].dropna().values
    curr_vals = current_pdf[col].dropna().values

    psi_val = psi(base_vals, curr_vals)
    ks_val = ks_statistic(base_vals, curr_vals)

    alert = "none"
    if psi_val > 0.25:
        alert = "significant"
    elif psi_val > 0.10:
        alert = "moderate"

    metrics_rows.append(
        {
            "snapshot_date": str(latest_snapshot_date),
            "feature_name": col,
            "feature_type": "numeric",
            "psi": round(psi_val, 6),
            "ks_statistic": round(ks_val, 6),
            "baseline_mean": round(float(np.nanmean(base_vals)), 6)
            if len(base_vals)
            else None,
            "current_mean": round(float(np.nanmean(curr_vals)), 6)
            if len(curr_vals)
            else None,
            "baseline_count": len(base_vals),
            "current_count": len(curr_vals),
            "alert_level": alert,
            "computed_at": run_ts,
        }
    )

for col in categorical_cols:
    base_vals = baseline_pdf[col].dropna().values
    curr_vals = current_pdf[col].dropna().values

    psi_val = psi_categorical(base_vals, curr_vals)

    alert = "none"
    if psi_val > 0.25:
        alert = "significant"
    elif psi_val > 0.10:
        alert = "moderate"

    metrics_rows.append(
        {
            "snapshot_date": str(latest_snapshot_date),
            "feature_name": col,
            "feature_type": "categorical",
            "psi": round(psi_val, 6),
            "ks_statistic": None,
            "baseline_mean": None,
            "current_mean": None,
            "baseline_count": len(base_vals),
            "current_count": len(curr_vals),
            "alert_level": alert,
            "computed_at": run_ts,
        }
    )


# ── Prediction drift ───────────────────────────────────────────────

try:
    scores_df = spark.table(scores_table)
    if scores_df.limit(1).count() > 0:
        score_vals = (
            scores_df.select("churn_probability")
            .toPandas()["churn_probability"]
            .dropna()
            .values
        )
        # Use training churn_label proportions as baseline prediction proxy.
        if "churn_label" in baseline_pdf.columns:
            base_preds = baseline_pdf["churn_label"].astype(float).dropna().values
        else:
            base_preds = np.array([])

        if len(score_vals) > 0 and len(base_preds) > 0:
            pred_psi = psi(base_preds, score_vals)
            pred_ks = ks_statistic(base_preds, score_vals)

            alert = "none"
            if pred_psi > 0.25:
                alert = "significant"
            elif pred_psi > 0.10:
                alert = "moderate"

            metrics_rows.append(
                {
                    "snapshot_date": str(latest_snapshot_date),
                    "feature_name": "_prediction_churn_probability",
                    "feature_type": "prediction",
                    "psi": round(pred_psi, 6),
                    "ks_statistic": round(pred_ks, 6),
                    "baseline_mean": round(float(np.nanmean(base_preds)), 6),
                    "current_mean": round(float(np.nanmean(score_vals)), 6),
                    "baseline_count": len(base_preds),
                    "current_count": len(score_vals),
                    "alert_level": alert,
                    "computed_at": run_ts,
                }
            )
except Exception:
    pass  # Scores table may not exist yet; feature drift still valuable.


# ── Write to drift_metrics ──────────────────────────────────────────

if not metrics_rows:
    summary = {"status": "skipped", "reason": "no features to compare"}
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(0)

drift_df = spark.createDataFrame(metrics_rows)
drift_target = table(args.catalog, args.schema, "drift_metrics")

# Overwrite this snapshot_date partition (idempotent re-runs).
(
    drift_df.write.format("delta")
    .mode("overwrite")
    .option("replaceWhere", f"snapshot_date = '{latest_snapshot_date}'")
    .saveAsTable(drift_target)
)

# ── Summary ─────────────────────────────────────────────────────────

significant = sum(1 for r in metrics_rows if r["alert_level"] == "significant")
moderate = sum(1 for r in metrics_rows if r["alert_level"] == "moderate")
summary = {
    "status": "completed",
    "snapshot_date": str(latest_snapshot_date),
    "features_monitored": len(metrics_rows),
    "significant_drift": significant,
    "moderate_drift": moderate,
}

print(json.dumps(summary, sort_keys=True))

log_run(
    spark=spark,
    catalog=args.catalog,
    schema=args.schema,
    task_name="drift_monitor",
    run_id=datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
    status="succeeded",
    started_at=_run_started,
    finished_at=datetime.now(timezone.utc),
    output_summary=summary,
)
