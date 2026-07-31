"""Load the uploaded Telco CSV into the Bronze Delta table."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

from common import get_spark, table
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

spark = get_spark()

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", required=True)
parser.add_argument("--schema", required=True)
parser.add_argument("--landing-file", required=True)
parser.add_argument("--bundled-source-file", required=True)
args = parser.parse_args()

source_columns = [
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
]
source_schema = StructType(
    [StructField(name, StringType(), True) for name in source_columns]
)

try:
    landing_file = Path(args.landing_file)
    bundled_source_file = Path(args.bundled_source_file)
    if bundled_source_file.is_file():
        # Python driver file I/O supports both /Workspace and /Volumes on
        # serverless compute. Spark then reads only from the managed Volume.
        shutil.copyfile(bundled_source_file, landing_file)
    elif not landing_file.is_file():
        raise FileNotFoundError(
            "Neither the landing file nor the bundle-packaged source exists. "
            f"Checked {landing_file} and {bundled_source_file}."
        )

    actual_columns = spark.read.option("header", True).csv(args.landing_file).columns
    if actual_columns != source_columns:
        raise ValueError(
            "Unexpected CSV columns. "
            f"Expected {source_columns}, received {actual_columns}."
        )
    bronze = (
        spark.read.option("header", True)
        .option("mode", "FAILFAST")
        .schema(source_schema)
        .csv(args.landing_file)
        .withColumn("_ingested_at", F.lit(datetime.now(timezone.utc)).cast("timestamp"))
        # input_file_name() is unsupported with Unity Catalog on serverless.
        # The metadata column provides the same lineage information.
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )
    if bronze.limit(1).count() == 0:
        raise ValueError("The landing CSV is empty; Bronze table was not replaced.")
except Exception as exc:
    raise RuntimeError(
        "Unable to seed or read the landing CSV from "
        f"{args.landing_file}. Original error: {exc}"
    ) from exc

(
    bronze.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(table(args.catalog, args.schema, "telco_bronze"))
)
