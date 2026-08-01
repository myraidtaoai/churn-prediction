"""Load the uploaded Telco CSV into the Bronze Delta table (idempotent seed).

This is the one-time historical seed path.  It copies the bundled CSV into
the managed landing Volume and writes it into ``telco_bronze`` as the initial
customer snapshot.  Re-running it when the table already contains rows from
the same source file is a safe no-op.

The event-based incremental path is ``ingest_events.py``.
"""

from __future__ import annotations

import _path_helper  # noqa: F401 — adds churn_pipeline/ to sys.path

import argparse
import hashlib
import json
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


def _record_hash(*values: str | None) -> str:
    """Deterministic SHA-256 over all column values in a row."""
    payload = "|".join(v if v is not None else "" for v in values)
    return hashlib.sha256(payload.encode()).hexdigest()


# Register the hash function as a UDF so each Bronze row gets a stable key.
_hash_udf = F.udf(_record_hash, StringType())

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
    ingestion_ts = datetime.now(timezone.utc)
    bronze = (
        spark.read.option("header", True)
        .option("mode", "FAILFAST")
        .schema(source_schema)
        .csv(args.landing_file)
        .withColumn(
            "_record_hash",
            _hash_udf(*[F.col(c) for c in source_columns]),
        )
        .withColumn("_ingested_at", F.lit(ingestion_ts).cast("timestamp"))
        # input_file_name() is unsupported with Unity Catalog on serverless.
        # The metadata column provides the same lineage information.
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_source_type", F.lit("csv_seed"))
    )
    if bronze.limit(1).count() == 0:
        raise ValueError("The landing CSV is empty; Bronze table was not written.")
except Exception as exc:
    raise RuntimeError(
        "Unable to seed or read the landing CSV from "
        f"{args.landing_file}. Original error: {exc}"
    ) from exc

bronze_table = table(args.catalog, args.schema, "telco_bronze")

# ── Idempotency: skip if Bronze already has rows from this source file ──
table_exists = spark.catalog.tableExists(bronze_table)
if table_exists:
    bronze_df = spark.table(bronze_table)
    has_source_type = "_source_type" in bronze_df.columns
    existing_count = (
        bronze_df.filter(F.col("_source_type") == "csv_seed").count()
        if has_source_type
        else bronze_df.count()
    )
    if existing_count > 0:
        summary = {
            "status": "skipped",
            "reason": "Bronze already contains csv_seed rows",
            "existing_rows": existing_count,
        }
        print(json.dumps(summary, sort_keys=True))
    else:
        # Table exists (from events, perhaps) but no CSV seed yet — append.
        bronze.write.format("delta").mode("append").saveAsTable(bronze_table)
        print(
            json.dumps({"status": "appended", "rows": bronze.count()}, sort_keys=True)
        )
else:
    # First run — create the table.
    bronze.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(bronze_table)
    print(json.dumps({"status": "created", "rows": bronze.count()}, sort_keys=True))
