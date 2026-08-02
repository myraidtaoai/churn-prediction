"""Incrementally ingest generated customer events into Bronze via Auto Loader.

Reads JSONL files from the managed landing Volume using Structured Streaming
with ``cloudFiles`` (Auto Loader) and an ``availableNow`` trigger.  This gives
incremental, checkpointed, exactly-once file ingestion with batch operational
semantics — no continuously running cluster, but full streaming guarantees.

Idempotency comes from the checkpoint: re-running the job after a successful
completion is a no-op because Auto Loader has already advanced past the consumed
files.  Re-running after a mid-job crash replays only the uncommitted micro-batch.

Events land in ``telco_events_bronze`` as an append-only table, deduplicated on
``event_id`` within each batch.  Cross-batch dedup is handled by a post-load
MERGE or by downstream consumers keying on ``event_id``.

Design decisions documented in docs/architecture.md:
  - ``availableNow=True`` — batch semantics, streaming machinery.
  - Directory listing (not file notification) — correct at this volume.
  - ``schemaEvolutionMode=rescue`` — additive changes auto-accepted.
"""

from __future__ import annotations

import _path_helper  # noqa: F401 — adds churn_pipeline/ to sys.path

import argparse
import json
from datetime import datetime, timezone

from common import get_spark, table
from contracts import build_spark_schema
from pyspark.sql import functions as F

spark = get_spark()

parser = argparse.ArgumentParser(
    description="Ingest events from the landing Volume into Bronze via Auto Loader."
)
parser.add_argument("--catalog", required=True)
parser.add_argument("--schema", required=True)
parser.add_argument(
    "--events-path",
    required=True,
    help="Volume path to the events directory (e.g. /Volumes/.../landing/events).",
)
parser.add_argument(
    "--checkpoint-path",
    required=True,
    help="Volume path for the Auto Loader checkpoint.",
)
args = parser.parse_args()

# ── Expected event schema ──────────────────────────────────────────────
# Built from the shared contract (contracts.py) so producer and consumer
# cannot drift.  Auto Loader uses this as the base schema; unexpected
# new fields go to ``_rescued_data`` via rescue mode.
EVENT_SCHEMA = build_spark_schema()

# ── Auto Loader read stream ───────────────────────────────────────────
events = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", f"{args.checkpoint_path}/schema")
    # Rescue mode: unexpected fields go to _rescued_data instead of
    # failing the batch.  This is the schema-evolution policy — additive
    # upstream changes are auto-accepted; breaking changes fail loudly
    # at the downstream contract check.
    .option("cloudFiles.schemaEvolutionMode", "rescue")
    # Directory listing, not file notification.  At ~7K customers and
    # one file/day, listing cost is negligible.  Switch to file
    # notification above ~10K files/batch.
    .option("cloudFiles.useNotifications", "false")
    .schema(EVENT_SCHEMA)
    .load(args.events_path)
)

# ── Enrich with ingestion metadata ────────────────────────────────────
ingestion_ts = datetime.now(timezone.utc)
enriched = (
    events
    # Parse the ISO timestamp string into a proper timestamp column.
    .withColumn(
        "event_ts",
        F.to_timestamp(F.col("event_timestamp"), "yyyy-MM-dd'T'HH:mm:ss'Z'"),
    )
    .withColumn("ingestion_timestamp", F.lit(ingestion_ts).cast("timestamp"))
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .withColumn("_source_type", F.lit("event_generator"))
    # Within-batch dedup on event_id.  Cross-batch dedup is not needed
    # because the generator is immutable (same config = same file) and
    # Auto Loader's checkpoint ensures each file is consumed exactly once.
    .dropDuplicates(["event_id"])
)

# ── Write to Bronze ──────────────────────────────────────────────────
bronze_table = table(args.catalog, args.schema, "telco_events_bronze")

query = (
    enriched.writeStream.format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{args.checkpoint_path}/bronze")
    # Merge schema so the _rescued_data column is auto-created when
    # rescue mode encounters unexpected fields.
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(bronze_table)
)

# availableNow runs all available micro-batches then stops.
query.awaitTermination()

# ── Report ────────────────────────────────────────────────────────────
row_count = spark.table(bronze_table).count()
rescued_count = 0
if "_rescued_data" in spark.table(bronze_table).columns:
    rescued_count = (
        spark.table(bronze_table).filter(F.col("_rescued_data").isNotNull()).count()
    )

print(
    json.dumps(
        {
            "status": "completed",
            "bronze_table": bronze_table,
            "total_rows": row_count,
            "rescued_rows": rescued_count,
            "checkpoint": args.checkpoint_path,
        },
        sort_keys=True,
    )
)
