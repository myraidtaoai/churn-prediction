"""Create cleaned Silver and dashboard-ready Gold churn tables."""

from __future__ import annotations

from common import get_spark, parse_catalog_schema, table
from pyspark.sql import functions as F

spark = get_spark()
args = parse_catalog_schema()
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

silver = (
    bronze.select(
        F.trim(F.col("customerID")).alias("customer_id"),
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
    )
)

if silver.filter(
    F.col("customer_id").isNull() | (F.length(F.col("customer_id")) == 0)
).limit(1).count():
    raise ValueError("customerID contains a missing or blank value.")
if silver.groupBy("customer_id").count().filter(F.col("count") > 1).limit(1).count():
    raise ValueError("Duplicate customerID values found after trimming.")
if silver.filter(F.col("churn_label").isNull()).limit(1).count():
    raise ValueError("Churn must contain only Yes or No values.")
if silver.filter(F.col("senior_citizen").isNull()).limit(1).count():
    raise ValueError("SeniorCitizen contains non-numeric or missing values.")
if silver.filter(F.col("monthly_charges").isNull()).limit(1).count():
    raise ValueError("MonthlyCharges contains non-numeric or missing values.")
if silver.filter(F.col("tenure").isNull() | (F.col("tenure") < 0)).limit(1).count():
    raise ValueError("tenure contains invalid values.")
if silver.filter(
    F.col("total_charges").isNull() & (F.col("tenure") != 0)
).limit(1).count():
    raise ValueError("TotalCharges is blank or non-numeric for a nonzero-tenure row.")

silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    table(args.catalog, args.schema, "telco_silver")
)

summary = (
    silver.groupBy("contract", "internet_service", "payment_method")
    .agg(
        F.count("*").alias("customers"),
        F.sum("churn_label").alias("churned_customers"),
        F.round(F.avg("churn_label"), 4).alias("churn_rate"),
        F.avg("monthly_charges").alias("avg_monthly_charge"),
        F.avg("tenure").alias("avg_tenure"),
    )
)
summary.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    table(args.catalog, args.schema, "churn_summary")
)
