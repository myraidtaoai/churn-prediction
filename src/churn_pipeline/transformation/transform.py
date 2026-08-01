"""Create cleaned Silver from Bronze using MERGE (idempotent).

The Silver table is keyed on ``(customer_id, snapshot_date)``.  Running the
same snapshot date twice updates existing rows but never creates duplicates.
The Gold ``churn_summary`` aggregate is rebuilt from the current Silver state
after every merge.

Data quality is enforced via the declarative ``quality`` framework.  Bad rows
are quarantined with their violation reason rather than failing the batch.
A fail-severity rule (e.g. zero rows surviving) still aborts the run.
"""

from __future__ import annotations

import _path_helper  # noqa: F401 — adds churn_pipeline/ to sys.path

import argparse
import json
from datetime import date, datetime, timezone

from common import get_spark, table
from pyspark.sql import functions as F
from quality import SILVER_RULES, apply_quality_rules

spark = get_spark()

# ── CLI ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Transform Bronze into Silver (MERGE) and rebuild Gold."
)
parser.add_argument("--catalog", required=True)
parser.add_argument("--schema", required=True)
parser.add_argument(
    "--snapshot-date",
    default="",
    help=(
        "Business date for this snapshot (YYYY-MM-DD).  Defaults to the "
        "current UTC date.  Used as part of the Silver primary key."
    ),
)
parser.add_argument(
    "--run-id",
    default="",
    help="Pipeline run ID for quality-metric lineage.  Defaults to a timestamp.",
)
args = parser.parse_args()

snapshot_date = (
    date.fromisoformat(args.snapshot_date)
    if args.snapshot_date.strip()
    else datetime.now(timezone.utc).date()
)
run_id = args.run_id.strip() or f"transform-{snapshot_date.isoformat()}"

# ── Read Bronze ─────────────────────────────────────────────────────────
bronze = spark.table(table(args.catalog, args.schema, "telco_bronze"))

required_columns = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}
missing_columns = sorted(required_columns.difference(bronze.columns))
if missing_columns:
    raise ValueError(f"Bronze table is missing required columns: {missing_columns}")

# ── Build Silver DataFrame ──────────────────────────────────────────────
silver = bronze.select(
    F.trim(F.col("customerID")).alias("customer_id"),
    F.lit(snapshot_date).cast("date").alias("snapshot_date"),
    F.trim(F.col("gender")).alias("gender"),
    F.col("SeniorCitizen").cast("int").alias("senior_citizen"),
    F.trim(F.col("Partner")).alias("partner"),
    F.trim(F.col("Dependents")).alias("dependents"),
    F.col("tenure").cast("int").alias("tenure"),
    F.trim(F.col("PhoneService")).alias("phone_service"),
    F.trim(F.col("MultipleLines")).alias("multiple_lines"),
    F.trim(F.col("InternetService")).alias("internet_service"),
    F.trim(F.col("OnlineSecurity")).alias("online_security"),
    F.trim(F.col("OnlineBackup")).alias("online_backup"),
    F.trim(F.col("DeviceProtection")).alias("device_protection"),
    F.trim(F.col("TechSupport")).alias("tech_support"),
    F.trim(F.col("StreamingTV")).alias("streaming_tv"),
    F.trim(F.col("StreamingMovies")).alias("streaming_movies"),
    F.trim(F.col("Contract")).alias("contract"),
    F.trim(F.col("PaperlessBilling")).alias("paperless_billing"),
    F.trim(F.col("PaymentMethod")).alias("payment_method"),
    F.col("MonthlyCharges").cast("double").alias("monthly_charges"),
    F.when(F.trim(F.col("TotalCharges")) == "", None)
    .otherwise(F.col("TotalCharges").cast("double"))
    .alias("total_charges"),
    F.initcap(F.trim(F.col("Churn"))).alias("churn"),
    F.when(F.lower(F.trim(F.col("Churn"))) == "yes", F.lit(1))
    .when(F.lower(F.trim(F.col("Churn"))) == "no", F.lit(0))
    .otherwise(F.lit(None).cast("int"))
    .alias("churn_label"),
    F.lit(datetime.now(timezone.utc)).cast("timestamp").alias("_transformed_at"),
)

# ── Data quality — quarantine bad rows, continue with the rest ─────────
# Replaces the previous raise-on-error validation.  Bad rows are preserved
# in telco_quarantine with their violation reason; good rows proceed to
# Silver.  A zero-row result still aborts (FAIL severity).
silver, quality_metrics = apply_quality_rules(
    df=silver,
    rules=SILVER_RULES,
    spark=spark,
    catalog=args.catalog,
    schema=args.schema,
    run_id=run_id,
    stage="silver",
)

# Duplicate customer_id check — this is structural, not a row-level rule.
if silver.groupBy("customer_id").count().filter(F.col("count") > 1).limit(1).count():
    raise ValueError("Duplicate customerID values found after trimming.")

# FAIL if quality left zero rows — something is fundamentally wrong.
clean_count = silver.count()
if clean_count == 0:
    raise ValueError(
        "All rows were quarantined or filtered.  Check telco_quarantine "
        "and data_quality_metrics for details."
    )

quarantined_count = sum(
    m.violating_rows for m in quality_metrics if m.severity == "quarantine"
)

# ── Write Silver via MERGE ──────────────────────────────────────────────
silver_table = table(args.catalog, args.schema, "telco_silver")
table_exists = spark.catalog.tableExists(silver_table)

if not table_exists:
    # First run — create the table directly.
    silver.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(silver_table)
    merge_status = "created"
    row_count = silver.count()
else:
    # Ensure the existing table has the columns the MERGE needs.
    existing_cols = {f.name for f in spark.table(silver_table).schema.fields}
    required_cols = set(silver.columns)
    missing_cols = required_cols - existing_cols

    if missing_cols:
        # Schema evolved — recreate the table so all columns are present.
        silver.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(silver_table)
        merge_status = "recreated"
        row_count = silver.count()
    else:
        # Subsequent runs — MERGE on (customer_id, snapshot_date).
        staging_view = "_silver_staging"
        silver.createOrReplaceTempView(staging_view)

        key_cols = {"customer_id", "snapshot_date"}
        update_cols = [c for c in silver.columns if c not in key_cols]
        set_clause = ", ".join(
            f"target.`{c}` = source.`{c}`" for c in update_cols
        )
        insert_cols = ", ".join(f"`{c}`" for c in silver.columns)
        insert_vals = ", ".join(f"source.`{c}`" for c in silver.columns)

        merge_sql = f"""
        MERGE INTO {silver_table} AS target
        USING {staging_view} AS source
        ON target.customer_id = source.customer_id
           AND target.snapshot_date = source.snapshot_date
        WHEN MATCHED THEN UPDATE SET {set_clause}
        WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
        """
        spark.sql(merge_sql)
        merge_status = "merged"
        row_count = silver.count()

# ── Gold aggregate ──────────────────────────────────────────────────────
current_silver = spark.table(silver_table)
summary = current_silver.groupBy("contract", "internet_service", "payment_method").agg(
    F.count("*").alias("customers"),
    F.sum("churn_label").alias("churned_customers"),
    F.round(F.avg("churn_label"), 4).alias("churn_rate"),
    F.avg("monthly_charges").alias("avg_monthly_charge"),
    F.avg("tenure").alias("avg_tenure"),
)
summary.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(table(args.catalog, args.schema, "churn_summary"))

print(
    json.dumps(
        {
            "status": merge_status,
            "snapshot_date": snapshot_date.isoformat(),
            "silver_rows": row_count,
            "quarantined_rows": quarantined_count,
            "quality_rules_evaluated": len(quality_metrics),
        },
        sort_keys=True,
    )
)
